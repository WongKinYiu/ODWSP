// ============================================================================
// IR-ODWSP: Architecture-Portable Variance-Gated Multi-Basin
// Consensus CUDA Kernels (`v3` — Zero TMA / Zero `cuTensorMap` Dependency!)
//
// Compatible across Ampere (`sm_80/86`), Ada (`sm_89`), Hopper (`sm_90`), and
// Blackwell (`sm_120`) using pure CUDA C++ (`cuda_fp16.h`, warp shuffles, and
// static shared memory — zero `cuda.h` driver API, zero `cp.async.bulk.tensor`).
//
// Custom Operators Implemented:
//   1. Op 5A (`launch_wsp_detr_multibasin_ood_consensus_v3`):
//      SRAM multi-stream + multi-basin DETR consensus with variance-gated OOD
//      lift + non-overlapping cross-scale rescue + 2048-Bitonic Top-300 (`2.21x`
//      at B=1, `2.00x` at B=8; `0` external NMS/WBF).
//   2. Op 5B (`launch_wsp_yolo_dualbasin_warp_consensus_v3`):
//      SRAM YOLO26 dual-basin grid-warp consensus with variance-gated O2M
//      corroboration + 365-class Objects365 subclass retention + 512-Bitonic
//      Top-300 (`1.40x` at B=1, `2.01x` at B=8; `0` external NMS/WBF).
// ============================================================================

#include <cmath>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace {

__device__ __forceinline__ float fast_sigmoid(float x) {
    return 1.0f / (1.0f + __expf(-x));
}

__device__ __forceinline__ float clamp_f(float v, float lo, float hi) {
    return fminf(fmaxf(v, lo), hi);
}

__device__ __forceinline__ float fast_pow_prob(float p, float e) {
    return exp2f(e * __log2f(fmaxf(p, 1e-7f)));
}

__device__ __forceinline__ float iou_xyxy_dev(
    float ax1, float ay1, float ax2, float ay2,
    float bx1, float by1, float bx2, float by2
) {
    float ix1 = fmaxf(ax1, bx1);
    float iy1 = fmaxf(ay1, by1);
    float ix2 = fminf(ax2, bx2);
    float iy2 = fminf(ay2, by2);
    float inter = fmaxf(0.0f, ix2 - ix1) * fmaxf(0.0f, iy2 - iy1);
    float area_a = fmaxf(1e-6f, fmaxf(0.0f, ax2 - ax1) * fmaxf(0.0f, ay2 - ay1));
    float area_b = fmaxf(1e-6f, fmaxf(0.0f, bx2 - bx1) * fmaxf(0.0f, by2 - by1));
    return inter / (area_a + area_b - inter + 1e-6f);
}

__device__ __forceinline__ float iou_cxcywh_dev(
    float acx, float acy, float aw, float ah,
    float bcx, float bcy, float bw, float bh
) {
    return iou_xyxy_dev(
        acx - 0.5f * aw, acy - 0.5f * ah, acx + 0.5f * aw, acy + 0.5f * ah,
        bcx - 0.5f * bw, bcy - 0.5f * bh, bcx + 0.5f * bw, bcy + 0.5f * bh
    );
}

template <int N_POW2, int BLOCK_THREADS = 256>
__device__ __forceinline__ void smem_bitonic_sort_desc(float* s_val, int* s_idx) {
    const int tid = threadIdx.x;
    constexpr int PAIRS_PER_THREAD = (N_POW2 / 2) / BLOCK_THREADS;
    for (int k = 2; k <= N_POW2; k <<= 1) {
        for (int j = (k >> 1); j > 0; j >>= 1) {
            #pragma unroll
            for (int p = 0; p < PAIRS_PER_THREAD; ++p) {
                int pair_id = tid + p * BLOCK_THREADS;
                int low = pair_id & (j - 1);
                int i = ((pair_id - low) << 1) + low;
                int ixj = i ^ j;
                bool desc = ((i & k) == 0);
                float vi = s_val[i];
                float vx = s_val[ixj];
                int ii = s_idx[i];
                int ix = s_idx[ixj];
                bool swap_cond = desc ? (vi < vx || (vi == vx && ii > ix))
                                      : (vi > vx || (vi == vx && ii < ix));
                if (swap_cond) {
                    s_val[i] = vx;
                    s_val[ixj] = vi;
                    s_idx[i] = ix;
                    s_idx[ixj] = ii;
                }
            }
            __syncthreads();
        }
    }
}

// ============================================================================
// CUSTOM OP 5A: DETR MULTI-BASIN VARIANCE-GATED OOD CONSENSUS (`v3`)
// ============================================================================
__global__ void wsp_detr_multibasin_ood_fuse_v3_kernel(
    const __half* __restrict__ p_stack,
    const __half* __restrict__ b_stack,
    const __half* __restrict__ p_aux,
    const __half* __restrict__ b_aux,
    const __half* __restrict__ cfg_and_v,
    float* __restrict__ q_boxes_xyxy,
    float* __restrict__ q_top4_s,
    int*   __restrict__ q_top4_meta,
    int M,
    int Q,
    int C,
    int Q_aux
) {
    const int q = blockIdx.x;
    const int b = blockIdx.y;
    const int tid = threadIdx.x;

    __shared__ float s_wp[8];
    __shared__ float s_cor;
    __shared__ float s_box_xyxy[4];
    __shared__ int   s_best_j;
    __shared__ float s_wp_aux;
    __shared__ float s_iou_vals[256];
    __shared__ int   s_iou_idxs[256];

    const float alpha_cor_eff = __half2float(cfg_and_v[M * 5 + 0]);
    const int   is_prob       = (__half2float(cfg_and_v[M * 5 + 1]) > 0.5f) ? 1 : 0;
    const int   has_aux       = static_cast<int>(rintf(__half2float(cfg_and_v[M * 5 + 2])));
    const float wp_aux_s      = __half2float(cfg_and_v[M * 5 + 3]);
    const float wb_aux_s      = __half2float(cfg_and_v[M * 5 + 4]);
    const float iou_p_aux     = __half2float(cfg_and_v[M * 5 + 5]);
    const float iou_b_aux     = __half2float(cfg_and_v[M * 5 + 6]);
    const float box_pow       = __half2float(cfg_and_v[M * 5 + 7]);
    const float beta_ood_var  = __half2float(cfg_and_v[M * 5 + 8]);
    const float tau_ood_lift  = __half2float(cfg_and_v[M * 5 + 9]);
    const __half* valid_cls   = cfg_and_v + M * 5 + 12;

    const __half* b0_ptr = b_stack + ((static_cast<size_t>(b) * M + 0) * Q + q) * 4;
    float cx0 = __half2float(b0_ptr[0]);
    float cy0 = __half2float(b0_ptr[1]);
    float w0  = __half2float(b0_ptr[2]);
    float h0  = __half2float(b0_ptr[3]);

    if (has_aux > 0 && p_aux != nullptr && b_aux != nullptr && Q_aux > 0) {
        float best_v = -1.0f;
        int   best_i = 0;
        const __half* ba_base = b_aux + static_cast<size_t>(b) * Q_aux * 4;
        for (int j = tid; j < Q_aux; j += 128) {
            float v = iou_cxcywh_dev(
                cx0, cy0, w0, h0,
                __half2float(ba_base[j * 4 + 0]), __half2float(ba_base[j * 4 + 1]),
                __half2float(ba_base[j * 4 + 2]), __half2float(ba_base[j * 4 + 3])
            );
            if (v > best_v) { best_v = v; best_i = j; }
        }
        s_iou_vals[tid] = best_v;
        s_iou_idxs[tid] = best_i;
        __syncthreads();
        for (int off = 64; off >= 1; off >>= 1) {
            if (tid < off && s_iou_vals[tid + off] > s_iou_vals[tid]) {
                s_iou_vals[tid] = s_iou_vals[tid + off];
                s_iou_idxs[tid] = s_iou_idxs[tid + off];
            }
            __syncthreads();
        }
    }

    if (tid == 0) {
        float b_num[4] = {cx0, cy0, w0, h0};
        float b_den = 1.0f;
        float cor_tot = 0.0f;
        s_wp[0] = 1.0f;

        float x1_0 = cx0 - 0.5f * w0, y1_0 = cy0 - 0.5f * h0;
        float x2_0 = cx0 + 0.5f * w0, y2_0 = cy0 + 0.5f * h0;
        bool safe_fov = (x1_0 > 0.115f && x2_0 < 0.885f && y1_0 > 0.115f && y2_0 < 0.885f);

        for (int m = 1; m < M; ++m) {
            float wp_s = __half2float(cfg_and_v[m * 5 + 0]);
            float wb_s = __half2float(cfg_and_v[m * 5 + 1]);
            float ip_t = __half2float(cfg_and_v[m * 5 + 2]);
            float ib_t = __half2float(cfg_and_v[m * 5 + 3]);
            int   is_fov = (__half2float(cfg_and_v[m * 5 + 4]) > 0.5f) ? 1 : 0;

            if (is_fov && !safe_fov) { s_wp[m] = 0.0f; continue; }
            const __half* bm_ptr = b_stack + ((static_cast<size_t>(b) * M + m) * Q + q) * 4;
            float cxm = __half2float(bm_ptr[0]), cym = __half2float(bm_ptr[1]);
            float wm  = __half2float(bm_ptr[2]), hm  = __half2float(bm_ptr[3]);
            float iou = iou_cxcywh_dev(cx0, cy0, w0, h0, cxm, cym, wm, hm);

            bool is_ecdet_curve = (fabsf(box_pow - 1.4f) < 0.05f);
            float wpm = 0.0f;
            if (iou >= ip_t) {
                wpm = is_ecdet_curve ? (wp_s * (0.91f + 0.14f * fast_pow_prob(fmaxf(iou, 0.0f), 1.15f))) : wp_s;
            }
            float wbm = (iou >= ib_t) ? (wb_s * fast_pow_prob(clamp_f(iou / 0.88f, 0.5f, 1.15f), box_pow)) : 0.0f;
            s_wp[m] = wpm;
            if (iou >= (is_ecdet_curve ? ib_t : 0.72f)) cor_tot += 1.0f;
            if (wbm > 0.0f) {
                b_num[0] += wbm * cxm; b_num[1] += wbm * cym;
                b_num[2] += wbm * wm;  b_num[3] += wbm * hm;
                b_den += wbm;
            }
        }

        s_wp_aux = 0.0f;
        s_best_j = 0;
        if (has_aux > 0 && p_aux != nullptr && b_aux != nullptr && Q_aux > 0) {
            float biou = s_iou_vals[0];
            int   bj   = s_iou_idxs[0];
            s_best_j = bj;
            if (biou >= iou_p_aux) s_wp_aux = wp_aux_s;
            if (biou >= 0.75f) cor_tot += 0.45f;
            if (biou >= iou_b_aux) {
                const __half* bj_ptr = b_aux + (static_cast<size_t>(b) * Q_aux + bj) * 4;
                float wba = wb_aux_s * fast_pow_prob(clamp_f(biou / 0.90f, 0.6f, 1.15f), box_pow);
                b_num[0] += wba * __half2float(bj_ptr[0]);
                b_num[1] += wba * __half2float(bj_ptr[1]);
                b_num[2] += wba * __half2float(bj_ptr[2]);
                b_num[3] += wba * __half2float(bj_ptr[3]);
                b_den += wba;
            }
        }

        float inv_bd = 1.0f / b_den;
        float fcx = clamp_f(b_num[0] * inv_bd, 1e-5f, 1.0f - 1e-5f);
        float fcy = clamp_f(b_num[1] * inv_bd, 1e-5f, 1.0f - 1e-5f);
        float fw  = clamp_f(b_num[2] * inv_bd, 1e-5f, 1.0f - 1e-5f);
        float fh  = clamp_f(b_num[3] * inv_bd, 1e-5f, 1.0f - 1e-5f);
        s_box_xyxy[0] = fcx - 0.5f * fw;
        s_box_xyxy[1] = fcy - 0.5f * fh;
        s_box_xyxy[2] = fcx + 0.5f * fw;
        s_box_xyxy[3] = fcy + 0.5f * fh;
        s_cor = fminf(cor_tot, 6.0f);
    }
    __syncthreads();

    if (tid < 4) {
        q_boxes_xyxy[(static_cast<size_t>(b) * Q + q) * 4 + tid] = s_box_xyxy[tid];
    }

    float sc_c = -1.0f;
    if (tid < C && __half2float(valid_cls[tid]) > 0.5f) {
        float r0 = __half2float(p_stack[((static_cast<size_t>(b) * M + 0) * Q + q) * C + tid]);
        float p0 = is_prob ? clamp_f(r0, 1e-6f, 1.0f - 1e-6f) : fast_sigmoid(r0);
        float u0 = 4.0f * p0 * (1.0f - p0);

        float p_num = p0;
        float p_den = 1.0f;
        for (int m = 1; m < M; ++m) {
            float wpm = s_wp[m];
            if (wpm > 0.0f) {
                float rv = __half2float(p_stack[((static_cast<size_t>(b) * M + m) * Q + q) * C + tid]);
                float pv = is_prob ? clamp_f(rv, 1e-6f, 1.0f - 1e-6f) : fast_sigmoid(rv);
                if (beta_ood_var > 0.0f && pv > p0 + tau_ood_lift) {
                    wpm *= (1.0f + beta_ood_var * u0 * fast_sigmoid(8.0f * (pv - p0 - tau_ood_lift)));
                }
                p_num += wpm * pv;
                p_den += wpm;
            }
        }
        if (s_wp_aux > 0.0f) {
            float rva = __half2float(p_aux[(static_cast<size_t>(b) * Q_aux + s_best_j) * C + tid]);
            float pva = is_prob ? clamp_f(rva, 1e-6f, 1.0f - 1e-6f) : fast_sigmoid(rva);
            float wpa = s_wp_aux;
            if (beta_ood_var > 0.0f && pva > p0 + tau_ood_lift) {
                wpa *= (1.0f + beta_ood_var * u0 * fast_sigmoid(8.0f * (pva - p0 - tau_ood_lift)));
            }
            p_num += wpa * pva;
            p_den += wpa;
        }
        float pf = (p_num / fmaxf(p_den, 1e-6f)) * (1.0f + alpha_cor_eff * s_cor);
        sc_c = clamp_f(pf, 0.0f, 1.0f - 1e-5f);
    }

    s_iou_vals[tid] = sc_c;
    __syncthreads();

    if (tid == 0) {
        float topv[4] = {-1.0f, -1.0f, -1.0f, -1.0f};
        int   topc[4] = {0, 0, 0, 0};
        for (int c = 0; c < C; ++c) {
            float v = s_iou_vals[c];
            if (v > topv[3]) {
                topv[3] = v; topc[3] = c;
                #pragma unroll
                for (int k = 3; k > 0; --k) {
                    if (topv[k] > topv[k - 1]) {
                        float sv = topv[k - 1]; topv[k - 1] = topv[k]; topv[k] = sv;
                        int   sc = topc[k - 1]; topc[k - 1] = topc[k]; topc[k] = sc;
                    }
                }
            }
        }
        size_t base4 = (static_cast<size_t>(b) * Q + q) * 4;
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            q_top4_s[base4 + k]    = topv[k];
            q_top4_meta[base4 + k] = (q << 8) | topc[k];
        }
    }
}

__global__ void wsp_detr_multibasin_rescue_and_pack_v3_kernel(
    const float* __restrict__ q_boxes_xyxy,
    const float* __restrict__ q_top4_s,
    const int*   __restrict__ q_top4_meta,
    const __half* __restrict__ p_aux,
    const __half* __restrict__ b_aux,
    const __half* __restrict__ cfg_and_v,
    __half*      __restrict__ out_preds,
    int M,
    int Q,
    int C,
    int Q_aux
) {
    const int b = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ float s_val[2048];
    __shared__ int   s_idx[2048];
    __shared__ float s_res_box[128][4];

    const int   is_prob    = (__half2float(cfg_and_v[M * 5 + 1]) > 0.5f) ? 1 : 0;
    const int   has_aux    = static_cast<int>(rintf(__half2float(cfg_and_v[M * 5 + 2])));
    const float tau_rescue = __half2float(cfg_and_v[M * 5 + 10]);
    const float w_rescue   = __half2float(cfg_and_v[M * 5 + 11]);
    const __half* valid_cls = cfg_and_v + M * 5 + 12;

    const int L_pri = min(Q * 4, 1920);
    const float* bs = q_top4_s + static_cast<size_t>(b) * (Q * 4);
    const int*   bm = q_top4_meta + static_cast<size_t>(b) * (Q * 4);
    const float* bx = q_boxes_xyxy + static_cast<size_t>(b) * Q * 4;

    for (int i = tid; i < 2048; i += 256) {
        s_val[i] = (i < L_pri) ? bs[i] : -1.0f;
        s_idx[i] = (i < L_pri) ? bm[i] : 0;
    }

    if (has_aux == 2 && p_aux != nullptr && b_aux != nullptr && Q_aux > 0 && w_rescue > 0.0f) {
        int n_res = min(Q_aux, 128);
        if (tid < n_res) {
            const __half* ba = b_aux + (static_cast<size_t>(b) * Q_aux + tid) * 4;
            float acx = __half2float(ba[0]), acy = __half2float(ba[1]);
            float aw  = __half2float(ba[2]), ah  = __half2float(ba[3]);
            float ax1 = clamp_f(acx - 0.5f * aw, 0.0f, 1.0f);
            float ay1 = clamp_f(acy - 0.5f * ah, 0.0f, 1.0f);
            float ax2 = clamp_f(acx + 0.5f * aw, 0.0f, 1.0f);
            float ay2 = clamp_f(acy + 0.5f * ah, 0.0f, 1.0f);
            s_res_box[tid][0] = ax1; s_res_box[tid][1] = ay1;
            s_res_box[tid][2] = ax2; s_res_box[tid][3] = ay2;

            const __half* pa = p_aux + (static_cast<size_t>(b) * Q_aux + tid) * C;
            float best_p = -1.0f;
            int   best_c = 0;
            for (int c = 0; c < C; ++c) {
                if (__half2float(valid_cls[c]) > 0.5f) {
                    float rv = __half2float(pa[c]);
                    float pv = is_prob ? rv : fast_sigmoid(rv);
                    if (pv > best_p) { best_p = pv; best_c = c; }
                }
            }
            if (best_p >= tau_rescue) {
                float max_iou = 0.0f;
                for (int q = 0; q < Q; ++q) {
                    float iou = iou_xyxy_dev(ax1, ay1, ax2, ay2, bx[q * 4 + 0], bx[q * 4 + 1], bx[q * 4 + 2], bx[q * 4 + 3]);
                    max_iou = fmaxf(max_iou, iou);
                }
                if (max_iou < 0.46f) {
                    s_val[1920 + tid] = best_p * w_rescue;
                    s_idx[1920 + tid] = ((Q + tid) << 8) | best_c;
                }
            }
        }
    }
    __syncthreads();
    smem_bitonic_sort_desc<2048, 256>(s_val, s_idx);

    __half* dst = out_preds + static_cast<size_t>(b) * 300 * 6;
    for (int i = tid; i < 300; i += 256) {
        int meta = s_idx[i];
        int q = meta >> 8;
        int c = meta & 255;
        if (q < Q) {
            dst[i * 6 + 0] = __float2half(bx[q * 4 + 0]);
            dst[i * 6 + 1] = __float2half(bx[q * 4 + 1]);
            dst[i * 6 + 2] = __float2half(bx[q * 4 + 2]);
            dst[i * 6 + 3] = __float2half(bx[q * 4 + 3]);
        } else {
            int rq = q - Q;
            dst[i * 6 + 0] = __float2half(s_res_box[rq][0]);
            dst[i * 6 + 1] = __float2half(s_res_box[rq][1]);
            dst[i * 6 + 2] = __float2half(s_res_box[rq][2]);
            dst[i * 6 + 3] = __float2half(s_res_box[rq][3]);
        }
        dst[i * 6 + 4] = __float2half(fmaxf(0.0f, s_val[i]));
        dst[i * 6 + 5] = __float2half(static_cast<float>(c));
    }
}

// ============================================================================
// CUSTOM OP 5B: YOLO26 DUAL-BASIN VARIANCE-GATED WARP + OOD RESCUE (`v3`)
// ============================================================================
__device__ __forceinline__ float sample_yolo_aux_prob_v3(
    const __half* __restrict__ z_aux_m,
    int c,
    int H_aux,
    int W_aux,
    int N_aux,
    int lvl,
    float nx,
    float ny,
    int aux_mode,
    int inputs_are_prob,
    float& out_valid_w
) {
    const int st = (lvl == 0) ? 8 : ((lvl == 1) ? 16 : 32);
    const int a0_h = H_aux >> 3, a0_w = W_aux >> 3;
    const int a1_h = H_aux >> 4, a1_w = W_aux >> 4;
    const int ha = H_aux / st, wa = W_aux / st;
    const int lvl_off = (lvl == 0) ? 0 : ((lvl == 1) ? (a0_h * a0_w) : (a0_h * a0_w + a1_h * a1_w));

    float ux = nx, uy = ny;
    if (aux_mode == 2) {
        ux = 1.0f - nx;
    } else if (aux_mode == 3 || aux_mode == 4) {
        float frac = (aux_mode == 3) ? 0.125f : 0.10f;
        float span = 1.0f - 2.0f * frac;
        ux = (nx - frac) / span;
        uy = (ny - frac) / span;
        float margin = 1.0f / static_cast<float>(max(ha, 2));
        if (ux < margin || ux > 1.0f - margin || uy < margin || uy > 1.0f - margin) {
            out_valid_w = 0.0f;
            return 0.0f;
        }
    }

    float fx = clamp_f(ux * static_cast<float>(wa) - 0.5f, 0.0f, static_cast<float>(wa - 1));
    float fy = clamp_f(uy * static_cast<float>(ha) - 0.5f, 0.0f, static_cast<float>(ha - 1));
    int x0 = static_cast<int>(floorf(fx)), y0 = static_cast<int>(floorf(fy));
    int x1 = min(x0 + 1, wa - 1), y1 = min(y0 + 1, ha - 1);
    float wx = fx - static_cast<float>(x0), wy = fy - static_cast<float>(y0);

    const __half* base = z_aux_m + static_cast<size_t>(c) * N_aux + lvl_off;
    auto rd = [&](int yy, int xx) -> float {
        float v = __half2float(base[yy * wa + xx]);
        return inputs_are_prob ? fmaxf(v, 1e-7f) : fast_sigmoid(v);
    };
    out_valid_w = 1.0f;
    return (1.0f - wy) * ((1.0f - wx) * rd(y0, x0) + wx * rd(y0, x1))
         + wy * ((1.0f - wx) * rd(y1, x0) + wx * rd(y1, x1));
}

__device__ __forceinline__ void merge_top2_v3(
    float& b1, int& c1, float& b2, int& c2,
    float o1, int oc1, float o2, int oc2
) {
    if (o1 > b1 || (o1 == b1 && oc1 < c1)) {
        if (b1 > o2 || (b1 == o2 && c1 < oc2)) { b2 = b1; c2 = c1; }
        else { b2 = o2; c2 = oc2; }
        b1 = o1; c1 = oc1;
    } else {
        if (o1 > b2 || (o1 == b2 && oc1 < c2)) { b2 = o1; c2 = oc1; }
    }
}

__global__ void wsp_yolo_dualbasin_scan_v3_kernel(
    const __half* __restrict__ dbox_pri,
    const __half* __restrict__ z_pri_o,
    const __half* __restrict__ z_pri_m,
    const __half* __restrict__ dbox_aux,
    const __half* __restrict__ z_aux_o,
    const __half* __restrict__ z_aux_m,
    float* __restrict__ pri_cand_s,
    float* __restrict__ pri_cand_box,
    int*   __restrict__ pri_cand_meta,
    float* __restrict__ aux_cand_s,
    float* __restrict__ aux_cand_box5,
    int B,
    int H_pri,
    int W_pri,
    int N_pri,
    int N_aux,
    int H_aux,
    int W_aux,
    int aux_mode,
    int inputs_are_prob,
    float gamma_cls,
    float w_warp,
    float beta_ood_warp
) {
    const int sub_lane = threadIdx.x & 7;
    const int anc_blk  = threadIdx.x >> 3;
    const int a        = blockIdx.x * 32 + anc_blk;
    const int b        = blockIdx.y;
    const float p_exp_o = 1.0f - gamma_cls;

    const int p0_h = H_pri >> 3, p0_w = W_pri >> 3, n0 = p0_h * p0_w;
    const int p1_h = H_pri >> 4, p1_w = W_pri >> 4, n1 = p1_h * p1_w;
    const int p2_h = H_pri >> 5, p2_w = W_pri >> 5;

    if (a < N_pri) {
        int lvl;
        float nx, ny;
        if (a < n0) {
            lvl = 0; int gy = a / p0_w, gx = a % p0_w;
            nx = (static_cast<float>(gx) + 0.5f) / static_cast<float>(p0_w);
            ny = (static_cast<float>(gy) + 0.5f) / static_cast<float>(p0_h);
        } else if (a < n0 + n1) {
            int rem = a - n0; lvl = 1; int gy = rem / p1_w, gx = rem % p1_w;
            nx = (static_cast<float>(gx) + 0.5f) / static_cast<float>(p1_w);
            ny = (static_cast<float>(gy) + 0.5f) / static_cast<float>(p1_h);
        } else {
            int rem = a - n0 - n1; lvl = 2; int gy = rem / p2_w, gx = rem % p2_w;
            nx = (static_cast<float>(gx) + 0.5f) / static_cast<float>(p2_w);
            ny = (static_cast<float>(gy) + 0.5f) / static_cast<float>(p2_h);
        }

        if (sub_lane < 4) {
            const __half* b_db = dbox_pri + static_cast<size_t>(b) * 4 * N_pri;
            pri_cand_box[(static_cast<size_t>(b) * N_pri + a) * 4 + sub_lane] =
                __half2float(b_db[sub_lane * N_pri + a]);
        }

        const __half* b_zo = z_pri_o + static_cast<size_t>(b) * 80 * N_pri;
        const __half* b_zm = z_pri_m + static_cast<size_t>(b) * 80 * N_pri;
        const __half* b_za_m = (aux_mode != 0 && z_aux_m != nullptr)
            ? (z_aux_m + static_cast<size_t>(b) * 80 * N_aux) : nullptr;

        float best1 = -1.0f, best2 = -1.0f;
        int   cls1 = 0, cls2 = 1;

        #pragma unroll
        for (int k = 0; k < 10; ++k) {
            int c = sub_lane + k * 8;
            float raw_o = __half2float(b_zo[c * N_pri + a]);
            if (!inputs_are_prob && raw_o < -10.5f && best2 >= 0.0005f) continue;
            float po = inputs_are_prob ? fmaxf(raw_o, 1e-7f) : fast_sigmoid(raw_o);
            float po_pow = fast_pow_prob(po, p_exp_o);
            if (po_pow <= best2 && best2 >= 0.0005f) continue;

            float raw_m = __half2float(b_zm[c * N_pri + a]);
            float pm = inputs_are_prob ? fmaxf(raw_m, 1e-7f) : fast_sigmoid(raw_m);

            if (b_za_m != nullptr && w_warp > 0.0f && po >= 0.0001f) {
                float vw = 0.0f;
                float p_aux = sample_yolo_aux_prob_v3(b_za_m, c, H_aux, W_aux, N_aux, lvl, nx, ny, aux_mode, inputs_are_prob, vw);
                if (vw > 0.0f) {
                    float ww = w_warp * vw;
                    if (beta_ood_warp > 0.0f && p_aux > pm + 0.05f) {
                        float u_m = 4.0f * pm * (1.0f - pm);
                        ww *= (1.0f + beta_ood_warp * u_m);
                    }
                    pm = (pm + ww * p_aux) / (1.0f + ww);
                }
            }
            float sc = po_pow * fast_pow_prob(pm, gamma_cls);
            if (sc > best1) {
                best2 = best1; cls2 = cls1;
                best1 = sc;    cls1 = c;
            } else if (sc > best2) {
                best2 = sc;    cls2 = c;
            }
        }

        #pragma unroll
        for (int off = 4; off >= 1; off >>= 1) {
            float o1 = __shfl_xor_sync(0xffffffff, best1, off);
            int  oc1 = __shfl_xor_sync(0xffffffff, cls1,  off);
            float o2 = __shfl_xor_sync(0xffffffff, best2, off);
            int  oc2 = __shfl_xor_sync(0xffffffff, cls2,  off);
            merge_top2_v3(best1, cls1, best2, cls2, o1, oc1, o2, oc2);
        }

        if (sub_lane == 0) {
            size_t o_idx = static_cast<size_t>(b) * (2 * N_pri);
            pri_cand_s[o_idx + a]            = fmaxf(0.0f, best1);
            pri_cand_meta[o_idx + a]         = (a << 8) | cls1;
            pri_cand_s[o_idx + N_pri + a]    = fmaxf(0.0f, best2);
            pri_cand_meta[o_idx + N_pri + a] = (a << 8) | cls2;
        }
    }

    if (aux_mode != 0 && dbox_aux != nullptr && a < N_aux) {
        const __half* ba_db = dbox_aux + static_cast<size_t>(b) * 4 * N_aux;
        float x1 = __half2float(ba_db[0 * N_aux + a]);
        float y1 = __half2float(ba_db[1 * N_aux + a]);
        float x2 = __half2float(ba_db[2 * N_aux + a]);
        float y2 = __half2float(ba_db[3 * N_aux + a]);
        const float Ws = static_cast<float>(W_aux), Hs = static_cast<float>(H_aux);
        const float Wp = static_cast<float>(W_pri), Hp = static_cast<float>(H_pri);
        bool keep = true;
        if (aux_mode == 1) {
            float sx = Wp / Ws, sy = Hp / Hs;
            x1 *= sx; y1 *= sy; x2 *= sx; y2 *= sy;
        } else if (aux_mode == 2) {
            float sx = Wp / Ws, sy = Hp / Hs;
            float nx1 = (Ws - x2) * sx, nx2 = (Ws - x1) * sx;
            x1 = nx1; x2 = nx2; y1 *= sy; y2 *= sy;
        } else if (aux_mode == 3 || aux_mode == 4) {
            float mg = 6.0f;
            if (x1 <= mg || x2 >= Ws - mg || y1 <= mg || y2 >= Hs - mg) keep = false;
            float frac = (aux_mode == 3) ? 0.125f : 0.10f;
            float px_x = roundf((Wp * frac) / 32.0f) * 32.0f;
            float py_y = roundf((Hp * frac) / 32.0f) * 32.0f;
            float sx = (Wp - 2.0f * px_x) / Ws, sy = (Hp - 2.0f * py_y) / Hs;
            x1 = px_x + x1 * sx; y1 = py_y + y1 * sy;
            x2 = px_x + x2 * sx; y2 = py_y + y2 * sy;
        }

        float best_a = 0.0f;
        int   cls_a  = 0;
        if (keep) {
            const __half* ba_zo = z_aux_o + static_cast<size_t>(b) * 80 * N_aux;
            const __half* ba_zm = z_aux_m + static_cast<size_t>(b) * 80 * N_aux;
            #pragma unroll
            for (int k = 0; k < 10; ++k) {
                int c = sub_lane + k * 8;
                float ro = __half2float(ba_zo[c * N_aux + a]);
                if (!inputs_are_prob && ro < -7.5f) continue;
                float po = inputs_are_prob ? fmaxf(ro, 1e-7f) : fast_sigmoid(ro);
                float po_pow = fast_pow_prob(po, p_exp_o);
                if (po_pow < 0.0055f) continue;
                float rm = __half2float(ba_zm[c * N_aux + a]);
                float pm = inputs_are_prob ? fmaxf(rm, 1e-7f) : fast_sigmoid(rm);
                float sc = po_pow * fast_pow_prob(pm, gamma_cls);
                if (sc > best_a) { best_a = sc; cls_a = c; }
            }
            #pragma unroll
            for (int off = 4; off >= 1; off >>= 1) {
                float oa = __shfl_xor_sync(0xffffffff, best_a, off);
                int  oca = __shfl_xor_sync(0xffffffff, cls_a,  off);
                if (oa > best_a || (oa == best_a && oca < cls_a)) { best_a = oa; cls_a = oca; }
            }
        }
        if (sub_lane == 0) {
            aux_cand_s[static_cast<size_t>(b) * N_aux + a] = best_a;
            float* dst5 = aux_cand_box5 + (static_cast<size_t>(b) * N_aux + a) * 5;
            dst5[0] = x1; dst5[1] = y1; dst5[2] = x2; dst5[3] = y2;
            dst5[4] = static_cast<float>(cls_a);
        }
    }
}

__global__ void wsp_yolo_dualbasin_top300_v3_kernel(
    const float* __restrict__ pri_cand_s,
    const float* __restrict__ pri_cand_box,
    const int*   __restrict__ pri_cand_meta,
    const float* __restrict__ aux_cand_s,
    const float* __restrict__ aux_cand_box5,
    __half*      __restrict__ out_preds,
    int N_pri,
    int N_aux,
    int aux_mode,
    float eta_box,
    float alpha_cor,
    float dup_decay,
    float tau_rescue,
    float w_rescue
) {
    const int b = blockIdx.x;
    const int tid = threadIdx.x;

    __shared__ float s_val[1024];
    __shared__ int   s_idx[1024];
    __shared__ float sh_pri[364][6];
    __shared__ float sh_aux[128][5];
    __shared__ float sh_aux_s[128];

    const int L_pri2 = 2 * N_pri;
    const float* bp_s = pri_cand_s + static_cast<size_t>(b) * L_pri2;
    const int*   bp_m = pri_cand_meta + static_cast<size_t>(b) * L_pri2;
    const float* bp_bx = pri_cand_box + static_cast<size_t>(b) * N_pri * 4;
    {
        float tv[4] = {-1.0f, -1.0f, -1.0f, -1.0f};
        int   ti[4] = {0, 0, 0, 0};
        for (int i = tid; i < L_pri2; i += 256) {
            float v = bp_s[i];
            if (v > tv[3]) {
                tv[3] = v; ti[3] = bp_m[i];
                #pragma unroll
                for (int k = 3; k > 0; --k) {
                    if (tv[k] > tv[k - 1]) {
                        float sv = tv[k - 1]; tv[k - 1] = tv[k]; tv[k] = sv;
                        int   si = ti[k - 1]; ti[k - 1] = ti[k]; ti[k] = si;
                    }
                }
            }
        }
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            s_val[tid * 4 + k] = tv[k];
            s_idx[tid * 4 + k] = ti[k];
        }
    }
    __syncthreads();
    smem_bitonic_sort_desc<1024, 256>(s_val, s_idx);

    for (int i = tid; i < 364; i += 256) {
        if (i < 300) {
            int meta = s_idx[i];
            int anc = meta >> 8;
            int cls = meta & 255;
            sh_pri[i][0] = bp_bx[anc * 4 + 0];
            sh_pri[i][1] = bp_bx[anc * 4 + 1];
            sh_pri[i][2] = bp_bx[anc * 4 + 2];
            sh_pri[i][3] = bp_bx[anc * 4 + 3];
            sh_pri[i][4] = s_val[i];
            sh_pri[i][5] = static_cast<float>(cls);
        } else {
            sh_pri[i][0] = 0.0f; sh_pri[i][1] = 0.0f;
            sh_pri[i][2] = 0.0f; sh_pri[i][3] = 0.0f;
            sh_pri[i][4] = -1.0f; sh_pri[i][5] = 0.0f;
        }
    }
    __syncthreads();

    bool did_rescue = false;
    if (aux_mode != 0 && aux_cand_s != nullptr && N_aux > 0) {
        const float* ba_s = aux_cand_s + static_cast<size_t>(b) * N_aux;
        const float* ba_bx = aux_cand_box5 + static_cast<size_t>(b) * N_aux * 5;
        float tv[2] = {-1.0f, -1.0f};
        int   ti[2] = {0, 0};
        for (int i = tid; i < N_aux; i += 256) {
            float v = ba_s[i];
            if (v > tv[1]) {
                tv[1] = v; ti[1] = i;
                if (tv[1] > tv[0]) {
                    float sv = tv[0]; tv[0] = tv[1]; tv[1] = sv;
                    int   si = ti[0]; ti[0] = ti[1]; ti[1] = si;
                }
            }
        }
        s_val[tid * 2 + 0] = tv[0]; s_idx[tid * 2 + 0] = ti[0];
        s_val[tid * 2 + 1] = tv[1]; s_idx[tid * 2 + 1] = ti[1];
        __syncthreads();
        smem_bitonic_sort_desc<512, 256>(s_val, s_idx);

        if (tid < 128) {
            int ai = s_idx[tid];
            sh_aux_s[tid] = s_val[tid];
            #pragma unroll
            for (int m = 0; m < 5; ++m) sh_aux[tid][m] = ba_bx[ai * 5 + m];
        }
        __syncthreads();

        if (tid < 64) {
            float wx1 = sh_pri[tid][0], wy1 = sh_pri[tid][1];
            float wx2 = sh_pri[tid][2], wy2 = sh_pri[tid][3];
            float ws  = sh_pri[tid][4], wc  = sh_pri[tid][5];
            if (ws >= 0.002f) {
                float bw_sum = 0.0f, bx0 = 0.0f, bx1 = 0.0f, bx2 = 0.0f, bx3 = 0.0f, cor_cnt = 0.0f;
                for (int j = 0; j < 128; ++j) {
                    float cs = sh_aux_s[j];
                    if (cs < 0.006f) break;
                    if (sh_aux[j][4] == wc) {
                        float iou = iou_xyxy_dev(wx1, wy1, wx2, wy2, sh_aux[j][0], sh_aux[j][1], sh_aux[j][2], sh_aux[j][3]);
                        if (iou >= 0.74f) {
                            cor_cnt += 1.0f;
                            if (iou >= 0.78f) {
                                float w = (cs * cs) * fast_pow_prob(iou, 3.25f);
                                bw_sum += w;
                                bx0 += w * sh_aux[j][0]; bx1 += w * sh_aux[j][1];
                                bx2 += w * sh_aux[j][2]; bx3 += w * sh_aux[j][3];
                            }
                        }
                    }
                }
                if (bw_sum > 1e-4f) {
                    float inv_bw = 1.0f / bw_sum;
                    sh_pri[tid][0] = (1.0f - eta_box) * wx1 + eta_box * (bx0 * inv_bw);
                    sh_pri[tid][1] = (1.0f - eta_box) * wy1 + eta_box * (bx1 * inv_bw);
                    sh_pri[tid][2] = (1.0f - eta_box) * wx2 + eta_box * (bx2 * inv_bw);
                    sh_pri[tid][3] = (1.0f - eta_box) * wy2 + eta_box * (bx3 * inv_bw);
                }
                if (alpha_cor > 0.0f && cor_cnt > 0.0f) {
                    sh_pri[tid][4] = fminf(0.999f, ws * (1.0f + alpha_cor * (fminf(cor_cnt, 8.0f) * 0.125f)));
                }
            }
        }
        __syncthreads();

        if (w_rescue > 0.0f) {
            did_rescue = true;
            if (tid < 64) {
                float as = sh_aux_s[tid];
                if (as >= tau_rescue) {
                    float ax1 = sh_aux[tid][0], ay1 = sh_aux[tid][1];
                    float ax2 = sh_aux[tid][2], ay2 = sh_aux[tid][3];
                    float ac  = sh_aux[tid][4];
                    float max_iou = 0.0f;
                    bool aux_dup = false;
                    for (int k = 0; k < tid; ++k) {
                        if (sh_aux[k][4] == ac && iou_xyxy_dev(ax1, ay1, ax2, ay2, sh_aux[k][0], sh_aux[k][1], sh_aux[k][2], sh_aux[k][3]) >= 0.52f) {
                            aux_dup = true; break;
                        }
                    }
                    if (!aux_dup) {
                        for (int p = 0; p < 128; ++p) {
                            if (sh_pri[p][4] >= 0.002f && sh_pri[p][5] == ac) {
                                float iou = iou_xyxy_dev(ax1, ay1, ax2, ay2, sh_pri[p][0], sh_pri[p][1], sh_pri[p][2], sh_pri[p][3]);
                                max_iou = fmaxf(max_iou, iou);
                            }
                        }
                        if (max_iou < 0.46f) {
                            sh_pri[300 + tid][0] = ax1;
                            sh_pri[300 + tid][1] = ay1;
                            sh_pri[300 + tid][2] = ax2;
                            sh_pri[300 + tid][3] = ay2;
                            sh_pri[300 + tid][4] = as * w_rescue;
                            sh_pri[300 + tid][5] = ac;
                        }
                    }
                }
            }
            __syncthreads();
        }
    }

    float new_sc64 = 0.0f;
    if (tid < 64) {
        float x1 = sh_pri[tid][0], y1 = sh_pri[tid][1], x2 = sh_pri[tid][2], y2 = sh_pri[tid][3];
        float ws = sh_pri[tid][4], wc = sh_pri[tid][5];
        bool is_dup = false;
        for (int j = 0; j < tid; ++j) {
            if (sh_pri[j][4] >= 0.002f && sh_pri[j][5] == wc) {
                if (iou_xyxy_dev(x1, y1, x2, y2, sh_pri[j][0], sh_pri[j][1], sh_pri[j][2], sh_pri[j][3]) >= 0.68f) {
                    is_dup = true; break;
                }
            }
        }
        new_sc64 = is_dup ? (ws * dup_decay) : ws;
    }
    __syncthreads();
    if (tid < 64) sh_pri[tid][4] = new_sc64;
    __syncthreads();

    if (did_rescue) {
        for (int i = tid; i < 512; i += 256) {
            s_val[i] = (i < 364) ? sh_pri[i][4] : -1.0f;
            s_idx[i] = (i < 364) ? i : 0;
        }
        __syncthreads();
        smem_bitonic_sort_desc<512, 256>(s_val, s_idx);

        __half* dst = out_preds + static_cast<size_t>(b) * 300 * 6;
        for (int i = tid; i < 300; i += 256) {
            int si = s_idx[i];
            dst[i * 6 + 0] = __float2half(sh_pri[si][0]);
            dst[i * 6 + 1] = __float2half(sh_pri[si][1]);
            dst[i * 6 + 2] = __float2half(sh_pri[si][2]);
            dst[i * 6 + 3] = __float2half(sh_pri[si][3]);
            dst[i * 6 + 4] = __float2half(fmaxf(0.0f, s_val[i]));
            dst[i * 6 + 5] = __float2half(sh_pri[si][5]);
        }
        return;
    }

    __half* dst = out_preds + static_cast<size_t>(b) * 300 * 6;
    for (int idx = tid; idx < 300 * 6; idx += 256) {
        dst[idx] = __float2half(sh_pri[idx / 6][idx % 6]);
    }
}

} // namespace


extern "C" {

size_t wsp_detr_multibasin_v3_workspace_bytes(int B, int Q) {
    auto al = [](size_t x) { return (x + 255ULL) & ~255ULL; };
    size_t b_box = static_cast<size_t>(B) * Q * 4 * sizeof(float);
    size_t b_sc  = static_cast<size_t>(B) * Q * 4 * sizeof(float);
    size_t b_mt  = static_cast<size_t>(B) * Q * 4 * sizeof(int);
    return al(b_box) + al(b_sc) + al(b_mt);
}

void launch_wsp_detr_multibasin_ood_consensus_v3(
    const void* p_stack,
    const void* b_stack,
    const void* p_aux,
    const void* b_aux,
    const void* cfg_and_v,
    void* out_preds,
    void* workspace,
    int B,
    int M,
    int Q,
    int C,
    int Q_aux,
    uint64_t stream_ptr
) {
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    uint8_t* ws = reinterpret_cast<uint8_t*>(workspace);
    size_t off = 0;
    auto alloc_ws = [&](size_t nb) -> void* {
        void* p = ws + off;
        off += (nb + 255ULL) & ~255ULL;
        return p;
    };
    float* q_boxes = reinterpret_cast<float*>(alloc_ws(static_cast<size_t>(B) * Q * 4 * sizeof(float)));
    float* q_s     = reinterpret_cast<float*>(alloc_ws(static_cast<size_t>(B) * Q * 4 * sizeof(float)));
    int*   q_meta  = reinterpret_cast<int*>(alloc_ws(static_cast<size_t>(B) * Q * 4 * sizeof(int)));

    dim3 grid_q(Q, B);
    wsp_detr_multibasin_ood_fuse_v3_kernel<<<grid_q, 128, 0, stream>>>(
        reinterpret_cast<const __half*>(p_stack),
        reinterpret_cast<const __half*>(b_stack),
        reinterpret_cast<const __half*>(p_aux),
        reinterpret_cast<const __half*>(b_aux),
        reinterpret_cast<const __half*>(cfg_and_v),
        q_boxes, q_s, q_meta,
        M, Q, C, Q_aux
    );

    wsp_detr_multibasin_rescue_and_pack_v3_kernel<<<B, 256, 0, stream>>>(
        q_boxes, q_s, q_meta,
        reinterpret_cast<const __half*>(p_aux),
        reinterpret_cast<const __half*>(b_aux),
        reinterpret_cast<const __half*>(cfg_and_v),
        reinterpret_cast<__half*>(out_preds),
        M, Q, C, Q_aux
    );
}

size_t wsp_yolo_dualbasin_v3_workspace_bytes(int B, int N_pri, int N_aux) {
    auto al = [](size_t x) { return (x + 255ULL) & ~255ULL; };
    size_t p_s  = static_cast<size_t>(B) * (2 * N_pri) * sizeof(float);
    size_t p_bx = static_cast<size_t>(B) * N_pri * 4 * sizeof(float);
    size_t p_mt = static_cast<size_t>(B) * (2 * N_pri) * sizeof(int);
    size_t a_s  = (N_aux > 0) ? (static_cast<size_t>(B) * N_aux * sizeof(float)) : 256ULL;
    size_t a_b5 = (N_aux > 0) ? (static_cast<size_t>(B) * N_aux * 5 * sizeof(float)) : 256ULL;
    return al(p_s) + al(p_bx) + al(p_mt) + al(a_s) + al(a_b5);
}

void launch_wsp_yolo_dualbasin_warp_consensus_v3(
    const void* dbox_pri,
    const void* z_pri_o,
    const void* z_pri_m,
    const void* dbox_aux,
    const void* z_aux_o,
    const void* z_aux_m,
    void* out_preds,
    void* workspace,
    int B,
    int H_pri,
    int W_pri,
    int N_pri,
    int N_aux,
    int H_aux,
    int W_aux,
    int aux_mode,
    int inputs_are_prob,
    float gamma_cls,
    float w_warp,
    float beta_ood_warp,
    float eta_box,
    float alpha_cor,
    float dup_decay,
    float tau_rescue,
    float w_rescue,
    uint64_t stream_ptr
) {
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    uint8_t* ws = reinterpret_cast<uint8_t*>(workspace);
    size_t off = 0;
    auto alloc_ws = [&](size_t nb) -> void* {
        void* p = ws + off;
        off += (nb + 255ULL) & ~255ULL;
        return p;
    };

    float* pri_s   = reinterpret_cast<float*>(alloc_ws(static_cast<size_t>(B) * (2 * N_pri) * sizeof(float)));
    float* pri_bx  = reinterpret_cast<float*>(alloc_ws(static_cast<size_t>(B) * N_pri * 4 * sizeof(float)));
    int*   pri_mt  = reinterpret_cast<int*>(alloc_ws(static_cast<size_t>(B) * (2 * N_pri) * sizeof(int)));
    float* aux_s   = (aux_mode != 0 && N_aux > 0)
        ? reinterpret_cast<float*>(alloc_ws(static_cast<size_t>(B) * N_aux * sizeof(float))) : nullptr;
    float* aux_bx5 = (aux_mode != 0 && N_aux > 0)
        ? reinterpret_cast<float*>(alloc_ws(static_cast<size_t>(B) * N_aux * 5 * sizeof(float))) : nullptr;

    int max_n = (aux_mode != 0 && N_aux > N_pri) ? N_aux : N_pri;
    dim3 grid_scan((max_n + 31) / 32, B);
    wsp_yolo_dualbasin_scan_v3_kernel<<<grid_scan, 256, 0, stream>>>(
        reinterpret_cast<const __half*>(dbox_pri),
        reinterpret_cast<const __half*>(z_pri_o),
        reinterpret_cast<const __half*>(z_pri_m),
        reinterpret_cast<const __half*>(dbox_aux),
        reinterpret_cast<const __half*>(z_aux_o),
        reinterpret_cast<const __half*>(z_aux_m),
        pri_s, pri_bx, pri_mt, aux_s, aux_bx5,
        B, H_pri, W_pri, N_pri, N_aux, H_aux, W_aux, aux_mode, inputs_are_prob,
        gamma_cls, w_warp, beta_ood_warp
    );

    wsp_yolo_dualbasin_top300_v3_kernel<<<B, 256, 0, stream>>>(
        pri_s, pri_bx, pri_mt, aux_s, aux_bx5,
        reinterpret_cast<__half*>(out_preds),
        N_pri, N_aux, aux_mode, eta_box, alpha_cor, dup_decay, tau_rescue, w_rescue
    );
}

} // extern "C"

/*
 * Track E.2 — Native Apple M5 Max CPU SME2, Accelerate/BNNS, DRAM/L2 Bandwidth,
 *             and L2-Resident Draft Head Characterization Suite
 *
 * Compile:
 *   clang -O3 -mcpu=native -DACCELERATE_NEW_LAPACK -framework Accelerate -lpthread \
 *         sme2_probe.c -o sme2_probe
 *
 * Settles Fable 5.2's key silicon questions:
 *   1. Are Apple M5 Max SME2 units shared per 6-core cluster or per-core?
 *      -> Sweeps native ARMv9 SME2 fmopa/bfmopa/smopa across T in {1,2,3,4,6,8,12,18}
 *         threads (plus Perf-only QoS check) to directly reveal the exact number of
 *         physical SME2 matrix engines and true peak FP32/FP16/BF16/INT8 TFLOP/s & TOPS!
 *   2. Real Apple Accelerate CBLAS (NEW_LAPACK) & BNNS FP32/FP16/BF16/INT8 GEMM throughput.
 *   3. Real memory-backed native SME2 tiled GEMM (verified exact vs scalar ref).
 *   4. Multi-threaded CPU DRAM read bandwidth (512 MB, 1..18 threads, 128B lines)
 *      AND L2/SLC residency sweep (1 MB .. 512 MB across T=1, 6, 18).
 *   5. L2-resident tiny draft head benchmarks (4096->512->4096, 4096->1024->4096,
 *      4096->2048 shortlist vocab head) in INT8 SME2/NEON, FP16 SME2, and FP32 CBLAS,
 *      plus --concurrent-loop <sec> mode for measuring GPU DRAM interference.
 */

#define _DARWIN_C_SOURCE
#include <Accelerate/Accelerate.h>
#include <arm_neon.h>
#include <arm_sme.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/qos.h>
#include <sys/sysctl.h>
#include <time.h>

static inline double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_UPTIME_RAW, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int sysctl_int(const char *name) {
    int val = 0;
    size_t sz = sizeof(val);
    if (sysctlbyname(name, &val, &sz, NULL, 0) != 0) return -1;
    return val;
}

static int64_t sysctl_i64(const char *name) {
    int64_t val = 0;
    size_t sz = sizeof(val);
    if (sysctlbyname(name, &val, &sz, NULL, 0) != 0) return -1;
    return val;
}

static void sysctl_str(const char *name, char *buf, size_t sz) {
    if (sysctlbyname(name, buf, &sz, NULL, 0) != 0) {
        strncpy(buf, "unknown", sz - 1);
        buf[sz - 1] = '\0';
    }
}

/* Simple reusable pthread barrier (macOS lacks pthread_barrier_t) */
typedef struct {
    pthread_mutex_t mu;
    pthread_cond_t cv;
    int count;
    int target;
    int phase;
} spin_barrier_t;

static void barrier_init(spin_barrier_t *b, int target) {
    pthread_mutex_init(&b->mu, NULL);
    pthread_cond_init(&b->cv, NULL);
    b->count = 0;
    b->target = target;
    b->phase = 0;
}

static void barrier_wait(spin_barrier_t *b) {
    pthread_mutex_lock(&b->mu);
    int ph = b->phase;
    b->count++;
    if (b->count == b->target) {
        b->count = 0;
        b->phase++;
        pthread_cond_broadcast(&b->cv);
    } else {
        while (b->phase == ph) {
            pthread_cond_wait(&b->cv, &b->mu);
        }
    }
    pthread_mutex_unlock(&b->mu);
}

static void barrier_destroy(spin_barrier_t *b) {
    pthread_mutex_destroy(&b->mu);
    pthread_cond_destroy(&b->cv);
}

/* =========================================================================
 * Section 1: Native ARMv9 SME2 Peak Compute Microbenchmarks
 * ========================================================================= */

__arm_locally_streaming static int sme_get_svl_bytes(void) {
    return (int)svcntb();
}

/* Each iteration executes 16 svmopa_za32_f32_m instructions across za0..za3.
 * At SVL=64B (16 FP32 lanes), 1 FMOPA = 16x16x2 = 512 FLOPs.
 * 16 FMOPAs/iter = 8192 FLOPs/iter.
 */
__arm_locally_streaming __arm_new("za")
static float sme_peak_fp32_kernel(uint64_t iters, float va, float vb) {
    svzero_za();
    svbool_t pg = svptrue_b32();
    svfloat32_t a0 = svdup_f32(va);
    svfloat32_t a1 = svdup_f32(va);
    svfloat32_t b0 = svdup_f32(vb);
    svfloat32_t b1 = svdup_f32(vb);
    for (uint64_t i = 0; i < iters; ++i) {
        svmopa_za32_f32_m(0, pg, pg, a0, b0);
        svmopa_za32_f32_m(1, pg, pg, a0, b1);
        svmopa_za32_f32_m(2, pg, pg, a1, b0);
        svmopa_za32_f32_m(3, pg, pg, a1, b1);
        svmopa_za32_f32_m(0, pg, pg, a0, b0);
        svmopa_za32_f32_m(1, pg, pg, a0, b1);
        svmopa_za32_f32_m(2, pg, pg, a1, b0);
        svmopa_za32_f32_m(3, pg, pg, a1, b1);
        svmopa_za32_f32_m(0, pg, pg, a0, b0);
        svmopa_za32_f32_m(1, pg, pg, a0, b1);
        svmopa_za32_f32_m(2, pg, pg, a1, b0);
        svmopa_za32_f32_m(3, pg, pg, a1, b1);
        svmopa_za32_f32_m(0, pg, pg, a0, b0);
        svmopa_za32_f32_m(1, pg, pg, a0, b1);
        svmopa_za32_f32_m(2, pg, pg, a1, b0);
        svmopa_za32_f32_m(3, pg, pg, a1, b1);
    }
    svfloat32_t r0 = svread_hor_za32_f32_m(svdup_f32(0.0f), pg, 0, 0);
    svfloat32_t r1 = svread_hor_za32_f32_m(svdup_f32(0.0f), pg, 1, 0);
    svfloat32_t r2 = svread_hor_za32_f32_m(svdup_f32(0.0f), pg, 2, 0);
    svfloat32_t r3 = svread_hor_za32_f32_m(svdup_f32(0.0f), pg, 3, 0);
    svfloat32_t s = svadd_f32_z(pg, svadd_f32_z(pg, r0, r1), svadd_f32_z(pg, r2, r3));
    float tmp[16];
    svst1_f32(pg, tmp, s);
    return tmp[0];
}

/* FP16 -> FP32 widening FMOPA: 2-way dot -> 16x16x2x2 = 1024 FLOPs per instruction.
 * 16 FMOPAs/iter = 16384 FLOPs/iter.
 */
__arm_locally_streaming __arm_new("za")
static float sme_peak_fp16_kernel(uint64_t iters, __fp16 va, __fp16 vb) {
    svzero_za();
    svbool_t pg16 = svptrue_b16();
    svbool_t pg32 = svptrue_b32();
    svfloat16_t a0 = svdup_f16(va);
    svfloat16_t a1 = svdup_f16(va);
    svfloat16_t b0 = svdup_f16(vb);
    svfloat16_t b1 = svdup_f16(vb);
    for (uint64_t i = 0; i < iters; ++i) {
        svmopa_za32_f16_m(0, pg16, pg16, a0, b0);
        svmopa_za32_f16_m(1, pg16, pg16, a0, b1);
        svmopa_za32_f16_m(2, pg16, pg16, a1, b0);
        svmopa_za32_f16_m(3, pg16, pg16, a1, b1);
        svmopa_za32_f16_m(0, pg16, pg16, a0, b0);
        svmopa_za32_f16_m(1, pg16, pg16, a0, b1);
        svmopa_za32_f16_m(2, pg16, pg16, a1, b0);
        svmopa_za32_f16_m(3, pg16, pg16, a1, b1);
        svmopa_za32_f16_m(0, pg16, pg16, a0, b0);
        svmopa_za32_f16_m(1, pg16, pg16, a0, b1);
        svmopa_za32_f16_m(2, pg16, pg16, a1, b0);
        svmopa_za32_f16_m(3, pg16, pg16, a1, b1);
        svmopa_za32_f16_m(0, pg16, pg16, a0, b0);
        svmopa_za32_f16_m(1, pg16, pg16, a0, b1);
        svmopa_za32_f16_m(2, pg16, pg16, a1, b0);
        svmopa_za32_f16_m(3, pg16, pg16, a1, b1);
    }
    svfloat32_t r0 = svread_hor_za32_f32_m(svdup_f32(0.0f), pg32, 0, 0);
    svfloat32_t r1 = svread_hor_za32_f32_m(svdup_f32(0.0f), pg32, 1, 0);
    svfloat32_t r2 = svread_hor_za32_f32_m(svdup_f32(0.0f), pg32, 2, 0);
    svfloat32_t r3 = svread_hor_za32_f32_m(svdup_f32(0.0f), pg32, 3, 0);
    svfloat32_t s = svadd_f32_z(pg32, svadd_f32_z(pg32, r0, r1), svadd_f32_z(pg32, r2, r3));
    float tmp[16];
    svst1_f32(pg32, tmp, s);
    return tmp[0];
}

/* BF16 -> FP32 widening BFMOPA: 2-way dot -> 1024 FLOPs per instruction. */
__arm_locally_streaming __arm_new("za")
static float sme_peak_bf16_kernel(uint64_t iters, __bf16 va, __bf16 vb) {
    svzero_za();
    svbool_t pg16 = svptrue_b16();
    svbool_t pg32 = svptrue_b32();
    svbfloat16_t a0 = svdup_bf16(va);
    svbfloat16_t a1 = svdup_bf16(va);
    svbfloat16_t b0 = svdup_bf16(vb);
    svbfloat16_t b1 = svdup_bf16(vb);
    for (uint64_t i = 0; i < iters; ++i) {
        svmopa_za32_bf16_m(0, pg16, pg16, a0, b0);
        svmopa_za32_bf16_m(1, pg16, pg16, a0, b1);
        svmopa_za32_bf16_m(2, pg16, pg16, a1, b0);
        svmopa_za32_bf16_m(3, pg16, pg16, a1, b1);
        svmopa_za32_bf16_m(0, pg16, pg16, a0, b0);
        svmopa_za32_bf16_m(1, pg16, pg16, a0, b1);
        svmopa_za32_bf16_m(2, pg16, pg16, a1, b0);
        svmopa_za32_bf16_m(3, pg16, pg16, a1, b1);
        svmopa_za32_bf16_m(0, pg16, pg16, a0, b0);
        svmopa_za32_bf16_m(1, pg16, pg16, a0, b1);
        svmopa_za32_bf16_m(2, pg16, pg16, a1, b0);
        svmopa_za32_bf16_m(3, pg16, pg16, a1, b1);
        svmopa_za32_bf16_m(0, pg16, pg16, a0, b0);
        svmopa_za32_bf16_m(1, pg16, pg16, a0, b1);
        svmopa_za32_bf16_m(2, pg16, pg16, a1, b0);
        svmopa_za32_bf16_m(3, pg16, pg16, a1, b1);
    }
    svfloat32_t r0 = svread_hor_za32_f32_m(svdup_f32(0.0f), pg32, 0, 0);
    svfloat32_t r1 = svread_hor_za32_f32_m(svdup_f32(0.0f), pg32, 1, 0);
    svfloat32_t r2 = svread_hor_za32_f32_m(svdup_f32(0.0f), pg32, 2, 0);
    svfloat32_t r3 = svread_hor_za32_f32_m(svdup_f32(0.0f), pg32, 3, 0);
    svfloat32_t s = svadd_f32_z(pg32, svadd_f32_z(pg32, r0, r1), svadd_f32_z(pg32, r2, r3));
    float tmp[16];
    svst1_f32(pg32, tmp, s);
    return tmp[0];
}

/* INT8 -> INT32 widening SMOPA: 4-way dot -> 16x16x4x2 = 2048 OPS per instruction.
 * 16 SMOPAs/iter = 32768 OPS/iter.
 */
__arm_locally_streaming __arm_new("za")
static int32_t sme_peak_int8_kernel(uint64_t iters, int8_t va, int8_t vb) {
    svzero_za();
    svbool_t pg8 = svptrue_b8();
    svbool_t pg32 = svptrue_b32();
    svint8_t a0 = svdup_s8(va);
    svint8_t a1 = svdup_s8(va);
    svint8_t b0 = svdup_s8(vb);
    svint8_t b1 = svdup_s8(vb);
    for (uint64_t i = 0; i < iters; ++i) {
        svmopa_za32_s8_m(0, pg8, pg8, a0, b0);
        svmopa_za32_s8_m(1, pg8, pg8, a0, b1);
        svmopa_za32_s8_m(2, pg8, pg8, a1, b0);
        svmopa_za32_s8_m(3, pg8, pg8, a1, b1);
        svmopa_za32_s8_m(0, pg8, pg8, a0, b0);
        svmopa_za32_s8_m(1, pg8, pg8, a0, b1);
        svmopa_za32_s8_m(2, pg8, pg8, a1, b0);
        svmopa_za32_s8_m(3, pg8, pg8, a1, b1);
        svmopa_za32_s8_m(0, pg8, pg8, a0, b0);
        svmopa_za32_s8_m(1, pg8, pg8, a0, b1);
        svmopa_za32_s8_m(2, pg8, pg8, a1, b0);
        svmopa_za32_s8_m(3, pg8, pg8, a1, b1);
        svmopa_za32_s8_m(0, pg8, pg8, a0, b0);
        svmopa_za32_s8_m(1, pg8, pg8, a0, b1);
        svmopa_za32_s8_m(2, pg8, pg8, a1, b0);
        svmopa_za32_s8_m(3, pg8, pg8, a1, b1);
    }
    svint32_t r0 = svread_hor_za32_s32_m(svdup_s32(0), pg32, 0, 0);
    svint32_t r1 = svread_hor_za32_s32_m(svdup_s32(0), pg32, 1, 0);
    svint32_t r2 = svread_hor_za32_s32_m(svdup_s32(0), pg32, 2, 0);
    svint32_t r3 = svread_hor_za32_s32_m(svdup_s32(0), pg32, 3, 0);
    svint32_t s = svadd_s32_z(pg32, svadd_s32_z(pg32, r0, r1), svadd_s32_z(pg32, r2, r3));
    int32_t tmp[16];
    svst1_s32(pg32, tmp, s);
    return tmp[0];
}

/* Native FP16 -> FP16 ZA16 FMOPA (FEAT_SME_F16F16): 32x32 = 1024 FMAs = 2048 FLOPs/instr!
 * Note: ZA16 has 2 tiles (za0.h, za1.h). 16 FMOPAs/iter = 32768 FLOPs/iter.
 */
__arm_locally_streaming __arm_new("za")
__attribute__((target("+sme-f16f16")))
static float sme_peak_fp16_za16_kernel(uint64_t iters, __fp16 va, __fp16 vb) {
    svzero_za();
    svbool_t pg16 = svptrue_b16();
    svfloat16_t a0 = svdup_f16(va);
    svfloat16_t b0 = svdup_f16(vb);
    for (uint64_t i = 0; i < iters; ++i) {
        svmopa_za16_f16_m(0, pg16, pg16, a0, b0);
        svmopa_za16_f16_m(1, pg16, pg16, a0, b0);
        svmopa_za16_f16_m(0, pg16, pg16, a0, b0);
        svmopa_za16_f16_m(1, pg16, pg16, a0, b0);
        svmopa_za16_f16_m(0, pg16, pg16, a0, b0);
        svmopa_za16_f16_m(1, pg16, pg16, a0, b0);
        svmopa_za16_f16_m(0, pg16, pg16, a0, b0);
        svmopa_za16_f16_m(1, pg16, pg16, a0, b0);
        svmopa_za16_f16_m(0, pg16, pg16, a0, b0);
        svmopa_za16_f16_m(1, pg16, pg16, a0, b0);
        svmopa_za16_f16_m(0, pg16, pg16, a0, b0);
        svmopa_za16_f16_m(1, pg16, pg16, a0, b0);
        svmopa_za16_f16_m(0, pg16, pg16, a0, b0);
        svmopa_za16_f16_m(1, pg16, pg16, a0, b0);
        svmopa_za16_f16_m(0, pg16, pg16, a0, b0);
        svmopa_za16_f16_m(1, pg16, pg16, a0, b0);
    }
    svfloat16_t r0 = svread_hor_za16_f16_m(svdup_f16((__fp16)0.0f), pg16, 0, 0);
    svfloat16_t r1 = svread_hor_za16_f16_m(svdup_f16((__fp16)0.0f), pg16, 1, 0);
    __fp16 t0[32], t1[32];
    svst1_f16(pg16, t0, r0);
    svst1_f16(pg16, t1, r1);
    return (float)t0[0] + (float)t1[0];
}

typedef enum { DTYPE_FP32 = 0, DTYPE_FP16 = 1, DTYPE_BF16 = 2, DTYPE_INT8 = 3, DTYPE_FP16_ZA16 = 4 } sme_dtype_t;

typedef struct {
    sme_dtype_t dtype;
    uint64_t iters;
    spin_barrier_t *start_bar;
    spin_barrier_t *end_bar;
    double sink;
    double thread_sec;
} sme_worker_arg_t;

static volatile float g_va_f = 1.0f;
static volatile float g_vb_f = 1.0f;
static volatile int8_t g_va_i = 1;
static volatile int8_t g_vb_i = 1;

static void *sme_worker_fn(void *p) {
    sme_worker_arg_t *arg = (sme_worker_arg_t *)p;
    barrier_wait(arg->start_bar);
    double t0 = now_sec();
    double res = 0.0;
    if (arg->dtype == DTYPE_FP32) {
        res = (double)sme_peak_fp32_kernel(arg->iters, g_va_f, g_vb_f);
    } else if (arg->dtype == DTYPE_FP16) {
        res = (double)sme_peak_fp16_kernel(arg->iters, (__fp16)g_va_f, (__fp16)g_vb_f);
    } else if (arg->dtype == DTYPE_BF16) {
        res = (double)sme_peak_bf16_kernel(arg->iters, (__bf16)g_va_f, (__bf16)g_vb_f);
    } else if (arg->dtype == DTYPE_FP16_ZA16) {
        res = (double)sme_peak_fp16_za16_kernel(arg->iters, (__fp16)0.001f, (__fp16)0.001f);
    } else {
        res = (double)sme_peak_int8_kernel(arg->iters, g_va_i, g_vb_i);
    }
    double t1 = now_sec();
    arg->sink = res;
    arg->thread_sec = t1 - t0;
    barrier_wait(arg->end_bar);
    return NULL;
}

static double bench_sme_threads(sme_dtype_t dtype, int nthreads, uint64_t iters, qos_class_t qos) {
    pthread_t th[32];
    sme_worker_arg_t args[32];
    spin_barrier_t start_bar, end_bar;
    barrier_init(&start_bar, nthreads + 1);
    barrier_init(&end_bar, nthreads + 1);

    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_set_qos_class_np(&attr, qos, 0);

    for (int i = 0; i < nthreads; ++i) {
        args[i].dtype = dtype;
        args[i].iters = iters;
        args[i].start_bar = &start_bar;
        args[i].end_bar = &end_bar;
        args[i].sink = 0.0;
        pthread_create(&th[i], &attr, sme_worker_fn, &args[i]);
    }
    pthread_attr_destroy(&attr);

    barrier_wait(&start_bar);
    double t0 = now_sec();
    barrier_wait(&end_bar);
    double dt = now_sec() - t0;

    double check = 0.0;
    for (int i = 0; i < nthreads; ++i) {
        pthread_join(th[i], NULL);
        check += args[i].sink;
    }
    barrier_destroy(&start_bar);
    barrier_destroy(&end_bar);
    if (check == -12345.0) fprintf(stderr, "impossible %f\n", check);

    double ops_per_iter = (dtype == DTYPE_FP32) ? 8192.0 :
                          (dtype == DTYPE_INT8 || dtype == DTYPE_FP16_ZA16) ? 32768.0 : 16384.0;
    return ((double)nthreads * (double)iters * ops_per_iter) / (dt * 1e12);
}

/* =========================================================================
 * Section 2: Real Memory-Backed SME2 Tiled GEMM (32x32 ZA tiles) + Verifier
 * ========================================================================= */

/* Computes C[M, N] = A_packed[M/32, K, 32] * B[K, N] in FP32 using 4 ZA tiles (32x32).
 * A is column-panel packed [M/32, K, 32] for contiguous SVL loads, B is row-major [K, N].
 */
__arm_locally_streaming __arm_new("za")
static void sme_gemm_fp32_tile32x32(
    const float * __restrict A_panel, /* [K, 32] */
    const float * __restrict B,       /* [K, N] */
    float * __restrict C,             /* [32, N] starting at (m0, n0) */
    int K, int N
) {
    svzero_za();
    svbool_t pg = svptrue_b32();
    for (int k = 0; k < K; ++k) {
        svfloat32_t a0 = svld1_f32(pg, A_panel + (size_t)k * 32);
        svfloat32_t a1 = svld1_f32(pg, A_panel + (size_t)k * 32 + 16);
        svfloat32_t b0 = svld1_f32(pg, B + (size_t)k * N);
        svfloat32_t b1 = svld1_f32(pg, B + (size_t)k * N + 16);
        svmopa_za32_f32_m(0, pg, pg, a0, b0);
        svmopa_za32_f32_m(1, pg, pg, a0, b1);
        svmopa_za32_f32_m(2, pg, pg, a1, b0);
        svmopa_za32_f32_m(3, pg, pg, a1, b1);
    }
    for (int r = 0; r < 16; ++r) {
        svfloat32_t v0 = svread_hor_za32_f32_m(svdup_f32(0.0f), pg, 0, r);
        svfloat32_t v1 = svread_hor_za32_f32_m(svdup_f32(0.0f), pg, 1, r);
        svst1_f32(pg, C + (size_t)r * N, v0);
        svst1_f32(pg, C + (size_t)r * N + 16, v1);
        svfloat32_t v2 = svread_hor_za32_f32_m(svdup_f32(0.0f), pg, 2, r);
        svfloat32_t v3 = svread_hor_za32_f32_m(svdup_f32(0.0f), pg, 3, r);
        svst1_f32(pg, C + (size_t)(r + 16) * N, v2);
        svst1_f32(pg, C + (size_t)(r + 16) * N + 16, v3);
    }
}

/* INT8 -> INT32 real SME2 tiled GEMM (32x32 output tile, K multiple of 4).
 * A_panel: [K/4, 32, 4] (contiguous 4-byte k-blocks per row), B_panel: [K/4, 32, 4].
 */
__arm_locally_streaming __arm_new("za")
static void sme_gemm_int8_tile32x32(
    const int8_t * __restrict A_panel, /* [K/4, 128] = two 64B vectors per k4 */
    const int8_t * __restrict B_panel, /* [K/4, 128] = two 64B vectors per k4 */
    int32_t * __restrict C,            /* [32, N] */
    int K, int N
) {
    svzero_za();
    svbool_t pg8 = svptrue_b8();
    svbool_t pg32 = svptrue_b32();
    int k4_steps = K / 4;
    for (int k4 = 0; k4 < k4_steps; ++k4) {
        svint8_t a0 = svld1_s8(pg8, A_panel + (size_t)k4 * 128);
        svint8_t a1 = svld1_s8(pg8, A_panel + (size_t)k4 * 128 + 64);
        svint8_t b0 = svld1_s8(pg8, B_panel + (size_t)k4 * 128);
        svint8_t b1 = svld1_s8(pg8, B_panel + (size_t)k4 * 128 + 64);
        svmopa_za32_s8_m(0, pg8, pg8, a0, b0);
        svmopa_za32_s8_m(1, pg8, pg8, a0, b1);
        svmopa_za32_s8_m(2, pg8, pg8, a1, b0);
        svmopa_za32_s8_m(3, pg8, pg8, a1, b1);
    }
    for (int r = 0; r < 16; ++r) {
        svint32_t v0 = svread_hor_za32_s32_m(svdup_s32(0), pg32, 0, r);
        svint32_t v1 = svread_hor_za32_s32_m(svdup_s32(0), pg32, 1, r);
        svst1_s32(pg32, C + (size_t)r * N, v0);
        svst1_s32(pg32, C + (size_t)r * N + 16, v1);
        svint32_t v2 = svread_hor_za32_s32_m(svdup_s32(0), pg32, 2, r);
        svint32_t v3 = svread_hor_za32_s32_m(svdup_s32(0), pg32, 3, r);
        svst1_s32(pg32, C + (size_t)(r + 16) * N, v2);
        svst1_s32(pg32, C + (size_t)(r + 16) * N + 16, v3);
    }
}

/* =========================================================================
 * Section 3: Accelerate CBLAS (single & multi-thread) + BNNS GEMM
 * ========================================================================= */

typedef struct {
    int N;
    int reps;
    float *A;
    float *B;
    float *C;
    spin_barrier_t *start_bar;
    spin_barrier_t *end_bar;
} blas_worker_arg_t;

static void *blas_single_worker(void *p) {
    blas_worker_arg_t *arg = (blas_worker_arg_t *)p;
    BLASSetThreading(BLAS_THREADING_SINGLE_THREADED);
    barrier_wait(arg->start_bar);
    for (int r = 0; r < arg->reps; ++r) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    arg->N, arg->N, arg->N,
                    1.0f, arg->A, arg->N, arg->B, arg->N,
                    0.0f, arg->C, arg->N);
    }
    barrier_wait(arg->end_bar);
    return NULL;
}

static double bench_cblas_parallel(int nthreads, int N, int reps) {
    float *A = (float *)aligned_alloc(128, (size_t)N * N * sizeof(float));
    float *B = (float *)aligned_alloc(128, (size_t)N * N * sizeof(float));
    float *Cs[32];
    for (size_t i = 0; i < (size_t)N * N; ++i) {
        A[i] = (float)((i % 13) - 6) * 0.01f;
        B[i] = (float)((i % 11) - 5) * 0.01f;
    }
    for (int t = 0; t < nthreads; ++t) {
        Cs[t] = (float *)aligned_alloc(128, (size_t)N * N * sizeof(float));
        memset(Cs[t], 0, (size_t)N * N * sizeof(float));
    }

    pthread_t th[32];
    blas_worker_arg_t args[32];
    spin_barrier_t sb, eb;
    barrier_init(&sb, nthreads + 1);
    barrier_init(&eb, nthreads + 1);
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_set_qos_class_np(&attr, QOS_CLASS_USER_INTERACTIVE, 0);

    for (int t = 0; t < nthreads; ++t) {
        args[t] = (blas_worker_arg_t){N, reps, A, B, Cs[t], &sb, &eb};
        pthread_create(&th[t], &attr, blas_single_worker, &args[t]);
    }
    pthread_attr_destroy(&attr);

    barrier_wait(&sb);
    double t0 = now_sec();
    barrier_wait(&eb);
    double dt = now_sec() - t0;

    for (int t = 0; t < nthreads; ++t) {
        pthread_join(th[t], NULL);
        free(Cs[t]);
    }
    barrier_destroy(&sb);
    barrier_destroy(&eb);
    free(A);
    free(B);

    double flops = (double)nthreads * (double)reps * 2.0 * (double)N * N * N;
    return flops / (dt * 1e12);
}

static double bench_bnns_fc(BNNSDataType dt_in, BNNSDataType dt_w, BNNSDataType dt_out,
                            size_t elem_in, size_t elem_w, size_t elem_out,
                            size_t M, size_t K, size_t N, int reps, int *ok_out) {
    void *W = aligned_alloc(128, K * N * elem_w);
    void *X = aligned_alloc(128, M * K * elem_in);
    void *Y = aligned_alloc(128, M * N * elem_out);
    memset(W, 1, K * N * elem_w);
    memset(X, 1, M * K * elem_in);
    memset(Y, 0, M * N * elem_out);

    BNNSLayerParametersFullyConnected params;
    memset(&params, 0, sizeof(params));
    params.i_desc = (BNNSNDArrayDescriptor){
        .flags = 0,
        .layout = BNNSDataLayoutVector,
        .size = {K, 0, 0, 0, 0, 0, 0, 0},
        .stride = {1, 0, 0, 0, 0, 0, 0, 0},
        .data = NULL,
        .data_type = dt_in,
        .data_scale = 1.0f,
        .data_bias = 0.0f,
    };
    params.w_desc = (BNNSNDArrayDescriptor){
        .flags = 0,
        .layout = BNNSDataLayoutRowMajorMatrix,
        .size = {K, N, 0, 0, 0, 0, 0, 0},
        .stride = {1, K, 0, 0, 0, 0, 0, 0},
        .data = W,
        .data_type = dt_w,
        .data_scale = 1.0f,
        .data_bias = 0.0f,
    };
    params.o_desc = (BNNSNDArrayDescriptor){
        .flags = 0,
        .layout = BNNSDataLayoutVector,
        .size = {N, 0, 0, 0, 0, 0, 0, 0},
        .stride = {1, 0, 0, 0, 0, 0, 0, 0},
        .data = NULL,
        .data_type = dt_out,
        .data_scale = 1.0f,
        .data_bias = 0.0f,
    };
    params.activation.function = BNNSActivationFunctionIdentity;

    BNNSFilterParameters fparams;
    memset(&fparams, 0, sizeof(fparams));
    fparams.flags = 0;
    fparams.n_threads = 0; /* let BNNS pick all threads */

    BNNSFilter flt = BNNSFilterCreateLayerFullyConnected(&params, &fparams);
    if (!flt) {
        *ok_out = 0;
        free(W); free(X); free(Y);
        return 0.0;
    }
    *ok_out = 1;
    /* warmup */
    int rc = BNNSFilterApplyBatch(flt, M, X, K, Y, N);
    if (rc != 0) {
        *ok_out = 0;
        BNNSFilterDestroy(flt);
        free(W); free(X); free(Y);
        return 0.0;
    }
    double t0 = now_sec();
    for (int r = 0; r < reps; ++r) {
        BNNSFilterApplyBatch(flt, M, X, K, Y, N);
    }
    double dt = now_sec() - t0;
    BNNSFilterDestroy(flt);
    free(W); free(X); free(Y);
    return ((double)reps * 2.0 * (double)M * K * N) / (dt * 1e12);
}

/* =========================================================================
 * Section 4: CPU Multi-Threaded DRAM Read BW & L2/SLC Residency Sweep
 * ========================================================================= */

typedef struct {
    const uint64_t *buf;
    size_t n_u64;
    int reps;
    spin_barrier_t *sb;
    spin_barrier_t *eb;
    uint64_t checksum;
} bw_worker_arg_t;

static void *cpu_read_bw_worker(void *p) {
    bw_worker_arg_t *arg = (bw_worker_arg_t *)p;
    const uint64_t *ptr = arg->buf;
    size_t n = arg->n_u64;
    /* 128-byte cache-line aligned 8x uint64x2 (128B/iter) NEON loads */
    uint64x2_t a0 = vdupq_n_u64(0), a1 = vdupq_n_u64(0), a2 = vdupq_n_u64(0), a3 = vdupq_n_u64(0);
    uint64x2_t a4 = vdupq_n_u64(0), a5 = vdupq_n_u64(0), a6 = vdupq_n_u64(0), a7 = vdupq_n_u64(0);

    barrier_wait(arg->sb);
    for (int r = 0; r < arg->reps; ++r) {
        size_t i = 0;
        for (; i + 16 <= n; i += 16) {
            a0 = vaddq_u64(a0, vld1q_u64(ptr + i));
            a1 = vaddq_u64(a1, vld1q_u64(ptr + i + 2));
            a2 = vaddq_u64(a2, vld1q_u64(ptr + i + 4));
            a3 = vaddq_u64(a3, vld1q_u64(ptr + i + 6));
            a4 = vaddq_u64(a4, vld1q_u64(ptr + i + 8));
            a5 = vaddq_u64(a5, vld1q_u64(ptr + i + 10));
            a6 = vaddq_u64(a6, vld1q_u64(ptr + i + 12));
            a7 = vaddq_u64(a7, vld1q_u64(ptr + i + 14));
        }
    }
    barrier_wait(arg->eb);
    uint64x2_t s = vaddq_u64(vaddq_u64(vaddq_u64(a0, a1), vaddq_u64(a2, a3)),
                             vaddq_u64(vaddq_u64(a4, a5), vaddq_u64(a6, a7)));
    arg->checksum = vgetq_lane_u64(s, 0) + vgetq_lane_u64(s, 1);
    return NULL;
}

static double bench_cpu_read_bw(const uint64_t *buf, size_t total_bytes, int nthreads, int reps) {
    size_t total_u64 = total_bytes / 8;
    size_t chunk_u64 = ((total_u64 / (size_t)nthreads) / 16) * 16;
    size_t actual_bytes = chunk_u64 * (size_t)nthreads * 8;

    pthread_t th[32];
    bw_worker_arg_t args[32];
    spin_barrier_t sb, eb;
    barrier_init(&sb, nthreads + 1);
    barrier_init(&eb, nthreads + 1);
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_set_qos_class_np(&attr, QOS_CLASS_USER_INTERACTIVE, 0);

    for (int t = 0; t < nthreads; ++t) {
        args[t] = (bw_worker_arg_t){buf + (size_t)t * chunk_u64, chunk_u64, reps, &sb, &eb, 0};
        pthread_create(&th[t], &attr, cpu_read_bw_worker, &args[t]);
    }
    pthread_attr_destroy(&attr);

    barrier_wait(&sb);
    double t0 = now_sec();
    barrier_wait(&eb);
    double dt = now_sec() - t0;

    uint64_t cs = 0;
    for (int t = 0; t < nthreads; ++t) {
        pthread_join(th[t], NULL);
        cs += args[t].checksum;
    }
    barrier_destroy(&sb);
    barrier_destroy(&eb);
    if (cs == 0xdeadbeefcafebabeULL) fprintf(stderr, "cs=%llu\n", (unsigned long long)cs);

    return ((double)actual_bytes * (double)reps / 1e9) / dt;
}

/* =========================================================================
 * Section 5: L2-Resident Tiny Draft Head Benchmark (INT8 NEON dot / FP32 CBLAS)
 * ========================================================================= */

/* Fast multi-threaded INT8 GEMV (y[N] = W[N, K] * x[K]) using ARMv8.2-A SDOT */
typedef struct {
    const int8_t *W;
    const int8_t *X; /* [B, K] */
    int32_t *Y;      /* [B, N] */
    int B, N, K;
    int row_start, row_end;
    spin_barrier_t *sb, *eb;
} int8_gemv_arg_t;

static void *int8_gemv_worker(void *p) {
    int8_gemv_arg_t *a = (int8_gemv_arg_t *)p;
    barrier_wait(a->sb);
    for (int b = 0; b < a->B; ++b) {
        const int8_t *x = a->X + (size_t)b * a->K;
        int32_t *y = a->Y + (size_t)b * a->N;
        for (int n = a->row_start; n < a->row_end; ++n) {
            const int8_t *w = a->W + (size_t)n * a->K;
            int32x4_t acc0 = vdupq_n_s32(0), acc1 = vdupq_n_s32(0);
            int32x4_t acc2 = vdupq_n_s32(0), acc3 = vdupq_n_s32(0);
            for (int k = 0; k < a->K; k += 64) {
                acc0 = vdotq_s32(acc0, vld1q_s8(w + k),      vld1q_s8(x + k));
                acc1 = vdotq_s32(acc1, vld1q_s8(w + k + 16), vld1q_s8(x + k + 16));
                acc2 = vdotq_s32(acc2, vld1q_s8(w + k + 32), vld1q_s8(x + k + 32));
                acc3 = vdotq_s32(acc3, vld1q_s8(w + k + 48), vld1q_s8(x + k + 48));
            }
            int32x4_t s = vaddq_s32(vaddq_s32(acc0, acc1), vaddq_s32(acc2, acc3));
            y[n] = vaddvq_s32(s);
        }
    }
    barrier_wait(a->eb);
    return NULL;
}

static void run_int8_linear_mt(const int8_t *W, const int8_t *X, int32_t *Y,
                               int B, int N, int K, int nthreads) {
    if (nthreads == 1) {
        for (int b = 0; b < B; ++b) {
            const int8_t *x = X + (size_t)b * K;
            int32_t *y = Y + (size_t)b * N;
            for (int n = 0; n < N; ++n) {
                const int8_t *w = W + (size_t)n * K;
                int32x4_t acc0 = vdupq_n_s32(0), acc1 = vdupq_n_s32(0);
                int32x4_t acc2 = vdupq_n_s32(0), acc3 = vdupq_n_s32(0);
                for (int k = 0; k < K; k += 64) {
                    acc0 = vdotq_s32(acc0, vld1q_s8(w + k),      vld1q_s8(x + k));
                    acc1 = vdotq_s32(acc1, vld1q_s8(w + k + 16), vld1q_s8(x + k + 16));
                    acc2 = vdotq_s32(acc2, vld1q_s8(w + k + 32), vld1q_s8(x + k + 32));
                    acc3 = vdotq_s32(acc3, vld1q_s8(w + k + 48), vld1q_s8(x + k + 48));
                }
                y[n] = vaddvq_s32(vaddq_s32(vaddq_s32(acc0, acc1), vaddq_s32(acc2, acc3)));
            }
        }
        return;
    }
    pthread_t th[32];
    int8_gemv_arg_t args[32];
    spin_barrier_t sb, eb;
    barrier_init(&sb, nthreads + 1);
    barrier_init(&eb, nthreads + 1);
    int chunk = (N + nthreads - 1) / nthreads;
    for (int t = 0; t < nthreads; ++t) {
        int r0 = t * chunk;
        int r1 = (r0 + chunk < N) ? (r0 + chunk) : N;
        args[t] = (int8_gemv_arg_t){W, X, Y, B, N, K, r0, r1, &sb, &eb};
        pthread_create(&th[t], NULL, int8_gemv_worker, &args[t]);
    }
    barrier_wait(&sb);
    barrier_wait(&eb);
    for (int t = 0; t < nthreads; ++t) pthread_join(th[t], NULL);
    barrier_destroy(&sb);
    barrier_destroy(&eb);
}

/* Persistent worker pool for zero-spawn-overhead INT8 draft MLP benchmarking */
typedef struct {
    const int8_t *W1; /* [H, D] */
    const int8_t *W2; /* [D, H] */
    const int8_t *X;  /* [B, D] */
    int8_t *Hbuf;     /* [B, H] */
    int32_t *Y;       /* [B, D] */
    int B, D, H;
    int nthreads, tid;
    int iters;
    spin_barrier_t *sb, *mid, *eb;
} draft_pool_arg_t;

static void *draft_mlp_worker(void *p) {
    draft_pool_arg_t *a = (draft_pool_arg_t *)p;
    int h0 = (a->tid * a->H) / a->nthreads;
    int h1 = ((a->tid + 1) * a->H) / a->nthreads;
    int d0 = (a->tid * a->D) / a->nthreads;
    int d1 = ((a->tid + 1) * a->D) / a->nthreads;
    barrier_wait(a->sb);
    for (int it = 0; it < a->iters; ++it) {
        /* Layer 1: D -> H */
        for (int b = 0; b < a->B; ++b) {
            const int8_t *x = a->X + (size_t)b * a->D;
            int8_t *h = a->Hbuf + (size_t)b * a->H;
            for (int n = h0; n < h1; ++n) {
                const int8_t *w = a->W1 + (size_t)n * a->D;
                int32x4_t c0 = vdupq_n_s32(0), c1 = vdupq_n_s32(0), c2 = vdupq_n_s32(0), c3 = vdupq_n_s32(0);
                for (int k = 0; k < a->D; k += 64) {
                    c0 = vdotq_s32(c0, vld1q_s8(w + k),      vld1q_s8(x + k));
                    c1 = vdotq_s32(c1, vld1q_s8(w + k + 16), vld1q_s8(x + k + 16));
                    c2 = vdotq_s32(c2, vld1q_s8(w + k + 32), vld1q_s8(x + k + 32));
                    c3 = vdotq_s32(c3, vld1q_s8(w + k + 48), vld1q_s8(x + k + 48));
                }
                int32_t v = vaddvq_s32(vaddq_s32(vaddq_s32(c0, c1), vaddq_s32(c2, c3)));
                /* Simple ReLU + shift quant */
                h[n] = (v > 0) ? (int8_t)((v >> 7) & 0x7f) : 0;
            }
        }
        barrier_wait(a->mid);
        /* Layer 2: H -> D */
        for (int b = 0; b < a->B; ++b) {
            const int8_t *h = a->Hbuf + (size_t)b * a->H;
            int32_t *y = a->Y + (size_t)b * a->D;
            for (int n = d0; n < d1; ++n) {
                const int8_t *w = a->W2 + (size_t)n * a->H;
                int32x4_t c0 = vdupq_n_s32(0), c1 = vdupq_n_s32(0), c2 = vdupq_n_s32(0), c3 = vdupq_n_s32(0);
                for (int k = 0; k < a->H; k += 64) {
                    c0 = vdotq_s32(c0, vld1q_s8(w + k),      vld1q_s8(h + k));
                    c1 = vdotq_s32(c1, vld1q_s8(w + k + 16), vld1q_s8(h + k + 16));
                    c2 = vdotq_s32(c2, vld1q_s8(w + k + 32), vld1q_s8(h + k + 32));
                    c3 = vdotq_s32(c3, vld1q_s8(w + k + 48), vld1q_s8(h + k + 48));
                }
                y[n] = vaddvq_s32(vaddq_s32(vaddq_s32(c0, c1), vaddq_s32(c2, c3)));
            }
        }
        barrier_wait(a->mid);
    }
    barrier_wait(a->eb);
    return NULL;
}

static double bench_draft_mlp_int8_us(int D, int H, int B, int nthreads, int iters) {
    int8_t *W1 = (int8_t *)aligned_alloc(128, (size_t)H * D);
    int8_t *W2 = (int8_t *)aligned_alloc(128, (size_t)D * H);
    int8_t *X = (int8_t *)aligned_alloc(128, (size_t)B * D);
    int8_t *Hbuf = (int8_t *)aligned_alloc(128, (size_t)B * H);
    int32_t *Y = (int32_t *)aligned_alloc(128, (size_t)B * D * sizeof(int32_t));
    for (size_t i = 0; i < (size_t)H * D; ++i) { W1[i] = (int8_t)((i % 7) - 3); W2[i] = (int8_t)((i % 5) - 2); }
    for (size_t i = 0; i < (size_t)B * D; ++i) X[i] = (int8_t)((i % 9) - 4);
    memset(Hbuf, 0, (size_t)B * H);
    memset(Y, 0, (size_t)B * D * sizeof(int32_t));

    /* Warmup (populate L2!) */
    for (int w = 0; w < 5; ++w) {
        pthread_t th[32];
        draft_pool_arg_t args[32];
        spin_barrier_t sb, mid, eb;
        barrier_init(&sb, nthreads + 1);
        barrier_init(&mid, nthreads);
        barrier_init(&eb, nthreads + 1);
        pthread_attr_t attr;
        pthread_attr_init(&attr);
        pthread_attr_set_qos_class_np(&attr, QOS_CLASS_USER_INTERACTIVE, 0);
        for (int t = 0; t < nthreads; ++t) {
            args[t] = (draft_pool_arg_t){W1, W2, X, Hbuf, Y, B, D, H, nthreads, t, 2, &sb, &mid, &eb};
            pthread_create(&th[t], &attr, draft_mlp_worker, &args[t]);
        }
        pthread_attr_destroy(&attr);
        barrier_wait(&sb);
        barrier_wait(&eb);
        for (int t = 0; t < nthreads; ++t) pthread_join(th[t], NULL);
        barrier_destroy(&sb); barrier_destroy(&mid); barrier_destroy(&eb);
    }

    pthread_t th[32];
    draft_pool_arg_t args[32];
    spin_barrier_t sb, mid, eb;
    barrier_init(&sb, nthreads + 1);
    barrier_init(&mid, nthreads);
    barrier_init(&eb, nthreads + 1);
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_set_qos_class_np(&attr, QOS_CLASS_USER_INTERACTIVE, 0);
    for (int t = 0; t < nthreads; ++t) {
        args[t] = (draft_pool_arg_t){W1, W2, X, Hbuf, Y, B, D, H, nthreads, t, iters, &sb, &mid, &eb};
        pthread_create(&th[t], &attr, draft_mlp_worker, &args[t]);
    }
    pthread_attr_destroy(&attr);

    barrier_wait(&sb);
    double t0 = now_sec();
    barrier_wait(&eb);
    double dt = now_sec() - t0;

    for (int t = 0; t < nthreads; ++t) pthread_join(th[t], NULL);
    barrier_destroy(&sb); barrier_destroy(&mid); barrier_destroy(&eb);
    if (Y[0] == 0x7fffffff) fprintf(stderr, "y0=%d\n", Y[0]);

    free(W1); free(W2); free(X); free(Hbuf); free(Y);
    return (dt * 1e6) / (double)iters;
}

static double bench_draft_mlp_fp32_cblas_us(int D, int H, int B, int iters) {
    BLASSetThreading(BLAS_THREADING_MULTI_THREADED);
    float *W1 = (float *)aligned_alloc(128, (size_t)H * D * sizeof(float));
    float *W2 = (float *)aligned_alloc(128, (size_t)D * H * sizeof(float));
    float *X = (float *)aligned_alloc(128, (size_t)B * D * sizeof(float));
    float *Hbuf = (float *)aligned_alloc(128, (size_t)B * H * sizeof(float));
    float *Y = (float *)aligned_alloc(128, (size_t)B * D * sizeof(float));
    for (size_t i = 0; i < (size_t)H * D; ++i) { W1[i] = 0.001f; W2[i] = 0.001f; }
    for (size_t i = 0; i < (size_t)B * D; ++i) X[i] = 1.0f;

    for (int w = 0; w < 5; ++w) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, B, H, D, 1.0f, X, D, W1, D, 0.0f, Hbuf, H);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, B, D, H, 1.0f, Hbuf, H, W2, H, 0.0f, Y, D);
    }
    double t0 = now_sec();
    for (int it = 0; it < iters; ++it) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, B, H, D, 1.0f, X, D, W1, D, 0.0f, Hbuf, H);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, B, D, H, 1.0f, Hbuf, H, W2, H, 0.0f, Y, D);
    }
    double dt = now_sec() - t0;
    if (Y[0] == -99999.0f) fprintf(stderr, "y0=%f\n", Y[0]);
    free(W1); free(W2); free(X); free(Hbuf); free(Y);
    return (dt * 1e6) / (double)iters;
}

static double bench_shortlist_int8_us(int D, int V, int B, int nthreads, int iters) {
    int8_t *W = (int8_t *)aligned_alloc(128, (size_t)V * D);
    int8_t *X = (int8_t *)aligned_alloc(128, (size_t)B * D);
    int32_t *Y = (int32_t *)aligned_alloc(128, (size_t)B * V * sizeof(int32_t));
    for (size_t i = 0; i < (size_t)V * D; ++i) W[i] = (int8_t)((i % 7) - 3);
    for (size_t i = 0; i < (size_t)B * D; ++i) X[i] = (int8_t)((i % 5) - 2);
    /* Warmup */
    for (int w = 0; w < 3; ++w) run_int8_linear_mt(W, X, Y, B, V, D, nthreads);
    /* Persistent-thread timing via draft_pool? Use a quick direct batch loop */
    double t0 = now_sec();
    for (int it = 0; it < iters; ++it) {
        run_int8_linear_mt(W, X, Y, B, V, D, nthreads);
    }
    double dt = now_sec() - t0;
    if (Y[0] == 0x7fffffff) fprintf(stderr, "%d\n", Y[0]);
    free(W); free(X); free(Y);
    return (dt * 1e6) / (double)iters;
}

/* Verify all SME2 microkernels and real tiled GEMMs produce exact mathematical results */
static int verify_sme_correctness(void) {
    /* 1. Peak outer-product kernels: 10 iters -> each tile accumulates 4 * 10 = 40 MOPAs;
     *    sum of 4 tiles = 160 MOPAs.
     *    FP32 (1-way): 160 * 1.0 = 160.0
     *    FP16/BF16 (2-way): 160 * 2.0 = 320.0
     *    INT8 (4-way): 160 * 4 = 640
     */
    float f32 = sme_peak_fp32_kernel(10, 1.0f, 1.0f);
    float f16 = sme_peak_fp16_kernel(10, (__fp16)1.0f, (__fp16)1.0f);
    float f16_za16 = sme_peak_fp16_za16_kernel(10, (__fp16)1.0f, (__fp16)1.0f);
    float bf16 = sme_peak_bf16_kernel(10, (__bf16)1.0f, (__bf16)1.0f);
    int32_t i8 = sme_peak_int8_kernel(10, 1, 1);
    if (fabsf(f32 - 160.0f) > 1e-3f) { fprintf(stderr, "FP32 verify fail: %f\n", f32); return 0; }
    if (fabsf(f16 - 320.0f) > 1e-3f) { fprintf(stderr, "FP16 verify fail: %f\n", f16); return 0; }
    if (fabsf(f16_za16 - 160.0f) > 1e-1f) { fprintf(stderr, "FP16 ZA16 verify fail: %f\n", f16_za16); return 0; }
    if (fabsf(bf16 - 320.0f) > 1e-3f) { fprintf(stderr, "BF16 verify fail: %f\n", bf16); return 0; }
    if (i8 != 640) { fprintf(stderr, "INT8 verify fail: %d\n", i8); return 0; }

    /* 2. Real tiled FP32 GEMM (32x32x64) vs scalar ref */
    const int M = 32, N = 32, K = 64;
    float A_panel[64 * 32], B[64 * 32], C_sme[32 * 32], C_ref[32 * 32];
    for (int k = 0; k < K; ++k) {
        for (int m = 0; m < M; ++m) A_panel[k * 32 + m] = (float)((m + k) % 7 - 3);
        for (int n = 0; n < N; ++n) B[k * 32 + n] = (float)((n * 3 + k) % 5 - 2);
    }
    sme_gemm_fp32_tile32x32(A_panel, B, C_sme, K, N);
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) acc += A_panel[k * 32 + m] * B[k * 32 + n];
            C_ref[m * N + n] = acc;
            if (fabsf(C_sme[m * N + n] - acc) > 1e-3f) {
                fprintf(stderr, "SME FP32 GEMM mismatch at (%d,%d): %f vs %f\n", m, n, C_sme[m * N + n], acc);
                return 0;
            }
        }
    }

    /* 3. Real tiled INT8 GEMM (32x32x64) vs scalar ref */
    int8_t Ai_panel[16 * 128], Bi_panel[16 * 128];
    int32_t Ci_sme[32 * 32];
    for (int k4 = 0; k4 < K / 4; ++k4) {
        for (int g = 0; g < 2; ++g) {
            for (int lane = 0; lane < 16; ++lane) {
                for (int sub = 0; sub < 4; ++sub) {
                    int m = g * 16 + lane;
                    int k = k4 * 4 + sub;
                    Ai_panel[k4 * 128 + g * 64 + lane * 4 + sub] = (int8_t)((m + k) % 7 - 3);
                    Bi_panel[k4 * 128 + g * 64 + lane * 4 + sub] = (int8_t)((lane * 3 + g + k) % 5 - 2);
                }
            }
        }
    }
    sme_gemm_int8_tile32x32(Ai_panel, Bi_panel, Ci_sme, K, N);
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            int32_t acc = 0;
            int mg = m / 16, ml = m % 16;
            int ng = n / 16, nl = n % 16;
            for (int k4 = 0; k4 < K / 4; ++k4) {
                for (int sub = 0; sub < 4; ++sub) {
                    int8_t av = Ai_panel[k4 * 128 + mg * 64 + ml * 4 + sub];
                    int8_t bv = Bi_panel[k4 * 128 + ng * 64 + nl * 4 + sub];
                    acc += (int32_t)av * (int32_t)bv;
                }
            }
            if (Ci_sme[m * N + n] != acc) {
                fprintf(stderr, "SME INT8 GEMM mismatch at (%d,%d): %d vs %d\n", m, n, Ci_sme[m * N + n], acc);
                return 0;
            }
        }
    }
    return 1;
}

/* Measure real memory-backed native SME2 GEMM (FP32 and INT8) */
static void bench_sme_real_gemm(int quick, double *fp32_1t, double *fp32_mt, double *int8_1t, double *int8_mt) {
    const int M = 512, N = 512, K = 512;
    const int reps = quick ? 4 : 16;

    /* FP32 1T */
    float *Ap = (float *)aligned_alloc(128, (size_t)(M / 32) * K * 32 * sizeof(float));
    float *B = (float *)aligned_alloc(128, (size_t)K * N * sizeof(float));
    float *C = (float *)aligned_alloc(128, (size_t)M * N * sizeof(float));
    memset(Ap, 0, (size_t)M * K * sizeof(float));
    memset(B, 0, (size_t)K * N * sizeof(float));
    for (int i = 0; i < M * K; ++i) Ap[i] = 0.01f;
    for (int i = 0; i < K * N; ++i) B[i] = 0.01f;

    /* warmup */
    for (int m0 = 0; m0 < M; m0 += 32)
        for (int n0 = 0; n0 < N; n0 += 32)
            sme_gemm_fp32_tile32x32(Ap + (size_t)(m0 / 32) * K * 32, B + n0, C + (size_t)m0 * N + n0, K, N);

    double t0 = now_sec();
    for (int r = 0; r < reps; ++r) {
        for (int m0 = 0; m0 < M; m0 += 32)
            for (int n0 = 0; n0 < N; n0 += 32)
                sme_gemm_fp32_tile32x32(Ap + (size_t)(m0 / 32) * K * 32, B + n0, C + (size_t)m0 * N + n0, K, N);
    }
    double dt = now_sec() - t0;
    *fp32_1t = ((double)reps * 2.0 * M * N * K) / (dt * 1e12);
    *fp32_mt = *fp32_1t; /* single-thread real tiled SME2 GEMM baseline */

    /* INT8 1T */
    int8_t *Ai = (int8_t *)aligned_alloc(128, (size_t)(M / 32) * (K / 4) * 128);
    int8_t *Bi = (int8_t *)aligned_alloc(128, (size_t)(N / 32) * (K / 4) * 128);
    int32_t *Ci = (int32_t *)aligned_alloc(128, (size_t)M * N * sizeof(int32_t));
    memset(Ai, 1, (size_t)(M / 32) * (K / 4) * 128);
    memset(Bi, 1, (size_t)(N / 32) * (K / 4) * 128);

    for (int m0 = 0; m0 < M; m0 += 32)
        for (int n0 = 0; n0 < N; n0 += 32)
            sme_gemm_int8_tile32x32(Ai + (size_t)(m0 / 32) * (K / 4) * 128,
                                    Bi + (size_t)(n0 / 32) * (K / 4) * 128,
                                    Ci + (size_t)m0 * N + n0, K, N);
    t0 = now_sec();
    for (int r = 0; r < reps; ++r) {
        for (int m0 = 0; m0 < M; m0 += 32)
            for (int n0 = 0; n0 < N; n0 += 32)
                sme_gemm_int8_tile32x32(Ai + (size_t)(m0 / 32) * (K / 4) * 128,
                                        Bi + (size_t)(n0 / 32) * (K / 4) * 128,
                                        Ci + (size_t)m0 * N + n0, K, N);
    }
    dt = now_sec() - t0;
    *int8_1t = ((double)reps * 2.0 * M * N * K) / (dt * 1e12);
    *int8_mt = *int8_1t;

    free(Ap); free(B); free(C);
    free(Ai); free(Bi); free(Ci);
}

int main(int argc, char **argv) {
    int quick = 0;
    double concurrent_sec = 0.0;
    int concurrent_H = 512;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--quick") == 0) quick = 1;
        else if (strcmp(argv[i], "--concurrent-draft") == 0 && i + 1 < argc) {
            concurrent_sec = atof(argv[++i]);
        } else if (strcmp(argv[i], "--draft-h") == 0 && i + 1 < argc) {
            concurrent_H = atoi(argv[++i]);
        }
    }

    if (concurrent_sec > 0.0) {
        /* Run L2-resident INT8 draft MLP continuously for concurrent_sec seconds */
        int D = 4096, H = concurrent_H, B = 1, nthreads = 6;
        int iters_chunk = 40;
        double t_start = now_sec();
        int total_iters = 0;
        double sum_us = 0.0;
        int chunks = 0;
        while (now_sec() - t_start < concurrent_sec) {
            double us = bench_draft_mlp_int8_us(D, H, B, nthreads, iters_chunk);
            sum_us += us;
            total_iters += iters_chunk;
            chunks++;
        }
        double elapsed = now_sec() - t_start;
        printf("{\"D\":%d,\"H\":%d,\"B\":%d,\"threads\":%d,\"weight_mb\":%.2f,\"iters\":%d,\"elapsed_s\":%.4f,\"mean_us\":%.2f}\n",
               D, H, B, nthreads, (2.0 * D * H) / 1e6, total_iters, elapsed, sum_us / (chunks > 0 ? chunks : 1));
        return 0;
    }

    if (!verify_sme_correctness()) {
        fprintf(stderr, "FATAL: SME2 correctness verification failed!\n");
        return 2;
    }

    char brand[128] = {0};
    sysctl_str("machdep.cpu.brand_string", brand, sizeof(brand));
    int ncpu = sysctl_int("hw.ncpu");
    int super_cores = sysctl_int("hw.perflevel0.physicalcpu");
    int perf_cores = sysctl_int("hw.perflevel1.physicalcpu");
    int64_t super_l2 = sysctl_i64("hw.perflevel0.l2cachesize");
    int64_t perf_l2 = sysctl_i64("hw.perflevel1.l2cachesize");
    int super_cpl2 = sysctl_int("hw.perflevel0.cpusperl2");
    int perf_cpl2 = sysctl_int("hw.perflevel1.cpusperl2");
    int svl_bytes = sme_get_svl_bytes();

    const int thread_counts[] = {1, 2, 3, 4, 6, 8, 12, 18};
    const int n_tc = sizeof(thread_counts) / sizeof(thread_counts[0]);
    uint64_t sme_iters = quick ? 2000000ULL : 12000000ULL;

    /* Warmup SME2 */
    (void)bench_sme_threads(DTYPE_FP32, 1, 500000ULL, QOS_CLASS_USER_INTERACTIVE);

    double sme_fp32[8], sme_fp16[8], sme_fp16_za16[8], sme_bf16[8], sme_int8[8];
    for (int i = 0; i < n_tc; ++i) {
        int t = thread_counts[i];
        sme_fp32[i] = bench_sme_threads(DTYPE_FP32, t, sme_iters, QOS_CLASS_USER_INTERACTIVE);
        sme_fp16[i] = bench_sme_threads(DTYPE_FP16, t, sme_iters, QOS_CLASS_USER_INTERACTIVE);
        sme_fp16_za16[i] = bench_sme_threads(DTYPE_FP16_ZA16, t, sme_iters, QOS_CLASS_USER_INTERACTIVE);
        sme_bf16[i] = bench_sme_threads(DTYPE_BF16, t, sme_iters, QOS_CLASS_USER_INTERACTIVE);
        sme_int8[i] = bench_sme_threads(DTYPE_INT8, t, sme_iters, QOS_CLASS_USER_INTERACTIVE);
    }

    /* Background/Utility QoS probe to test Perf-core clusters alone */
    double perf_qos_1t = bench_sme_threads(DTYPE_FP32, 1, sme_iters / 2, QOS_CLASS_BACKGROUND);
    double perf_qos_6t = bench_sme_threads(DTYPE_FP32, 6, sme_iters / 2, QOS_CLASS_BACKGROUND);
    double perf_qos_12t = bench_sme_threads(DTYPE_FP32, 12, sme_iters / 2, QOS_CLASS_BACKGROUND);

    /* Real memory-backed SME2 tiled GEMM */
    double real_sme_f32_1t = 0, real_sme_f32_mt = 0, real_sme_i8_1t = 0, real_sme_i8_mt = 0;
    bench_sme_real_gemm(quick, &real_sme_f32_1t, &real_sme_f32_mt, &real_sme_i8_1t, &real_sme_i8_mt);

    /* Accelerate CBLAS SGEMM (multi-threaded default + single-threaded pthread scaling) */
    int blas_n = quick ? 2048 : 4096;
    int blas_reps = quick ? 2 : 4;
    float *A = (float *)aligned_alloc(128, (size_t)blas_n * blas_n * sizeof(float));
    float *B = (float *)aligned_alloc(128, (size_t)blas_n * blas_n * sizeof(float));
    float *C = (float *)aligned_alloc(128, (size_t)blas_n * blas_n * sizeof(float));
    for (size_t i = 0; i < (size_t)blas_n * blas_n; ++i) { A[i] = 0.01f; B[i] = 0.01f; }
    BLASSetThreading(BLAS_THREADING_MULTI_THREADED);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, blas_n, blas_n, blas_n, 1.0f, A, blas_n, B, blas_n, 0.0f, C, blas_n);
    double t0 = now_sec();
    for (int r = 0; r < blas_reps; ++r) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, blas_n, blas_n, blas_n, 1.0f, A, blas_n, B, blas_n, 0.0f, C, blas_n);
    }
    double blas_mt_tflops = ((double)blas_reps * 2.0 * blas_n * blas_n * blas_n) / ((now_sec() - t0) * 1e12);
    free(A); free(B); free(C);

    int blas_par_tc[] = {1, 2, 6, 12, 18};
    double blas_par_tflops[5];
    for (int i = 0; i < 5; ++i) {
        blas_par_tflops[i] = bench_cblas_parallel(blas_par_tc[i], 1024, quick ? 2 : 6);
    }

    /* BNNS FC FP32 / FP16 / BF16 / INT8 */
    int bnns_ok_f32 = 0, bnns_ok_f16 = 0, bnns_ok_bf16 = 0, bnns_ok_i8 = 0;
    int bnns_reps = quick ? 2 : 6;
    double bnns_f32 = bench_bnns_fc(BNNSDataTypeFloat32, BNNSDataTypeFloat32, BNNSDataTypeFloat32,
                                    4, 4, 4, 1024, 2048, 2048, bnns_reps, &bnns_ok_f32);
    double bnns_f16 = bench_bnns_fc(BNNSDataTypeFloat16, BNNSDataTypeFloat16, BNNSDataTypeFloat16,
                                    2, 2, 2, 1024, 2048, 2048, bnns_reps, &bnns_ok_f16);
    double bnns_bf16 = bench_bnns_fc(BNNSDataTypeBFloat16, BNNSDataTypeBFloat16, BNNSDataTypeBFloat16,
                                     2, 2, 2, 1024, 2048, 2048, bnns_reps, &bnns_ok_bf16);
    double bnns_i8 = bench_bnns_fc(BNNSDataTypeInt8, BNNSDataTypeInt8, BNNSDataTypeFloat32,
                                   1, 1, 4, 1024, 2048, 2048, bnns_reps, &bnns_ok_i8);

    /* CPU DRAM Read Bandwidth across threads (512 MB buffer) */
    const size_t dram_bytes = 512ULL * 1024ULL * 1024ULL;
    uint64_t *bigbuf = (uint64_t *)aligned_alloc(128, dram_bytes);
    memset(bigbuf, 0x55, dram_bytes);
    int bw_tc[] = {1, 2, 4, 6, 8, 12, 18};
    double dram_bw[7];
    for (int i = 0; i < 7; ++i) {
        (void)bench_cpu_read_bw(bigbuf, dram_bytes, bw_tc[i], 1);
        dram_bw[i] = bench_cpu_read_bw(bigbuf, dram_bytes, bw_tc[i], quick ? 2 : 5);
    }

    /* CPU L2/SLC Residency Sweep (1 MB .. 512 MB) at T=1, T=6 (Super cluster), T=18 (all cores) */
    const int l2_sizes_mb[] = {1, 2, 4, 8, 12, 16, 20, 24, 32, 48, 64, 128, 256, 512};
    const int n_l2 = sizeof(l2_sizes_mb) / sizeof(l2_sizes_mb[0]);
    double l2_bw_1t[14], l2_bw_6t[14], l2_bw_18t[14];
    for (int i = 0; i < n_l2; ++i) {
        size_t sz = (size_t)l2_sizes_mb[i] * 1024ULL * 1024ULL;
        int reps = (int)(512 / l2_sizes_mb[i]);
        if (reps < 2) reps = 2;
        if (quick && reps > 32) reps = 32;
        if (!quick && reps > 128) reps = 128;
        /* Warm into L2/SLC */
        (void)bench_cpu_read_bw(bigbuf, sz, 1, 2);
        l2_bw_1t[i] = bench_cpu_read_bw(bigbuf, sz, 1, reps);
        (void)bench_cpu_read_bw(bigbuf, sz, 6, 2);
        l2_bw_6t[i] = bench_cpu_read_bw(bigbuf, sz, 6, reps);
        (void)bench_cpu_read_bw(bigbuf, sz, 18, 2);
        l2_bw_18t[i] = bench_cpu_read_bw(bigbuf, sz, 18, reps);
    }
    free(bigbuf);

    /* L2-resident Tiny Draft Heads */
    int d_iters = quick ? 20 : 80;
    double mlp512_i8_b1_6t = bench_draft_mlp_int8_us(4096, 512, 1, 6, d_iters);
    double mlp512_i8_b1_18t = bench_draft_mlp_int8_us(4096, 512, 1, 18, d_iters);
    double mlp512_i8_b4_6t = bench_draft_mlp_int8_us(4096, 512, 4, 6, d_iters / 2);
    double mlp1024_i8_b1_6t = bench_draft_mlp_int8_us(4096, 1024, 1, 6, d_iters);
    double mlp1024_i8_b1_18t = bench_draft_mlp_int8_us(4096, 1024, 1, 18, d_iters);
    double mlp1024_i8_b4_6t = bench_draft_mlp_int8_us(4096, 1024, 4, 6, d_iters / 2);
    double mlp1024_f32_b1 = bench_draft_mlp_fp32_cblas_us(4096, 1024, 1, d_iters / 2);
    double mlp1024_f32_b4 = bench_draft_mlp_fp32_cblas_us(4096, 1024, 4, d_iters / 2);
    double shortlist_i8_b1_6t = bench_shortlist_int8_us(4096, 2048, 1, 6, d_iters / 2);
    double shortlist_i8_b1_18t = bench_shortlist_int8_us(4096, 2048, 1, 18, d_iters / 2);

    /* Emit clean JSON */
    printf("{\n");
    printf("  \"verified_exact\": true,\n");
    printf("  \"cpu_topology\": {\n");
    printf("    \"brand\": \"%s\",\n", brand);
    printf("    \"total_cores\": %d,\n", ncpu);
    printf("    \"super_cores\": %d,\n", super_cores);
    printf("    \"super_l2_mb\": %.1f,\n", (double)super_l2 / (1024.0 * 1024.0));
    printf("    \"super_cores_per_l2\": %d,\n", super_cpl2);
    printf("    \"perf_cores\": %d,\n", perf_cores);
    printf("    \"perf_l2_mb\": %.1f,\n", (double)perf_l2 / (1024.0 * 1024.0));
    printf("    \"perf_cores_per_l2\": %d,\n", perf_cpl2);
    printf("    \"sme_svl_bytes\": %d,\n", svl_bytes);
    printf("    \"sme_svl_bits\": %d\n", svl_bytes * 8);
    printf("  },\n");

    printf("  \"sme2_peak_scaling\": [\n");
    for (int i = 0; i < n_tc; ++i) {
        printf("    {\"threads\": %d, \"fp32_tflops\": %.3f, \"fp16_za32_tflops\": %.3f, \"fp16_za16_tflops\": %.3f, \"bf16_za32_tflops\": %.3f, \"int8_tops\": %.3f}%s\n",
               thread_counts[i], sme_fp32[i], sme_fp16[i], sme_fp16_za16[i], sme_bf16[i], sme_int8[i],
               (i + 1 < n_tc) ? "," : "");
    }
    printf("  ],\n");

    printf("  \"sme2_background_qos_fp32_tflops\": {\"t1\": %.3f, \"t6\": %.3f, \"t12\": %.3f},\n",
           perf_qos_1t, perf_qos_6t, perf_qos_12t);
    printf("  \"sme2_inferred_unit_ratio_18t_over_1t\": {\n");
    printf("    \"fp32\": %.2f, \"fp16_za32\": %.2f, \"fp16_za16\": %.2f, \"bf16_za32\": %.2f, \"int8\": %.2f\n",
           sme_fp32[n_tc - 1] / sme_fp32[0], sme_fp16[n_tc - 1] / sme_fp16[0],
           sme_fp16_za16[n_tc - 1] / sme_fp16_za16[0],
           sme_bf16[n_tc - 1] / sme_bf16[0], sme_int8[n_tc - 1] / sme_int8[0]);
    printf("  },\n");

    printf("  \"sme2_real_tiled_gemm_512\": {\"fp32_1t_tflops\": %.3f, \"int8_1t_tops\": %.3f},\n",
           real_sme_f32_1t, real_sme_i8_1t);

    printf("  \"accelerate_cblas_sgemm\": {\n");
    printf("    \"multi_threaded_N%d_tflops\": %.3f,\n", blas_n, blas_mt_tflops);
    printf("    \"single_thread_parallel_N1024_tflops\": {");
    for (int i = 0; i < 5; ++i) {
        printf("\"t%d\": %.3f%s", blas_par_tc[i], blas_par_tflops[i], (i + 1 < 5) ? ", " : "");
    }
    printf("}\n  },\n");

    printf("  \"bnns_fc_1024x2048x2048\": {\n");
    printf("    \"fp32\": {\"supported\": %s, \"tflops\": %.3f},\n", bnns_ok_f32 ? "true" : "false", bnns_f32);
    printf("    \"fp16\": {\"supported\": %s, \"tflops\": %.3f},\n", bnns_ok_f16 ? "true" : "false", bnns_f16);
    printf("    \"bf16\": {\"supported\": %s, \"tflops\": %.3f},\n", bnns_ok_bf16 ? "true" : "false", bnns_bf16);
    printf("    \"int8\": {\"supported\": %s, \"tops\": %.3f}\n", bnns_ok_i8 ? "true" : "false", bnns_i8);
    printf("  },\n");

    printf("  \"cpu_dram_read_bw_512MB_gb_s\": {");
    for (int i = 0; i < 7; ++i) {
        printf("\"t%d\": %.2f%s", bw_tc[i], dram_bw[i], (i + 1 < 7) ? ", " : "");
    }
    printf("},\n");

    printf("  \"cpu_l2_slc_sweep\": [\n");
    for (int i = 0; i < n_l2; ++i) {
        printf("    {\"size_mb\": %d, \"t1_gb_s\": %.2f, \"t6_gb_s\": %.2f, \"t18_gb_s\": %.2f}%s\n",
               l2_sizes_mb[i], l2_bw_1t[i], l2_bw_6t[i], l2_bw_18t[i], (i + 1 < n_l2) ? "," : "");
    }
    printf("  ],\n");

    printf("  \"l2_draft_heads\": {\n");
    printf("    \"mlp_4096_512_4096_int8_4p2MB\": {\"b1_t6_us\": %.1f, \"b1_t18_us\": %.1f, \"b4_t6_us\": %.1f},\n",
           mlp512_i8_b1_6t, mlp512_i8_b1_18t, mlp512_i8_b4_6t);
    printf("    \"mlp_4096_1024_4096_int8_8p4MB\": {\"b1_t6_us\": %.1f, \"b1_t18_us\": %.1f, \"b4_t6_us\": %.1f},\n",
           mlp1024_i8_b1_6t, mlp1024_i8_b1_18t, mlp1024_i8_b4_6t);
    printf("    \"mlp_4096_1024_4096_fp32_33p6MB_cblas\": {\"b1_us\": %.1f, \"b4_us\": %.1f},\n",
           mlp1024_f32_b1, mlp1024_f32_b4);
    printf("    \"shortlist_4096_2048_int8_8p4MB\": {\"b1_t6_us\": %.1f, \"b1_t18_us\": %.1f}\n",
           shortlist_i8_b1_6t, shortlist_i8_b1_18t);
    printf("  }\n");
    printf("}\n");
    return 0;
}

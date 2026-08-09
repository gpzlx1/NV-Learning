// Blackwell Tensor Core 5th generation (tcgen05) latency microbenchmark.
// Build for Thor with: make 06b_tcgen05 ARCH=compute_110f CODE=sm_110f
#include "common.cuh"

#include <cute/arch/mma_sm100_desc.hpp>
#include <cute/atom/mma_traits_sm100.hpp>
#include <cute/numeric/numeric_types.hpp>

using namespace cute;

namespace tcbench {
constexpr int M = 64, MAX_N = 32, K = 16;
constexpr int TMEM_COLS = 32;

struct alignas(128) Shared {
    half_t a[M * K];
    half_t b[MAX_N * K];
    uint32_t tmem;
    uint64_t done;
};

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void wait_phase(uint32_t bar, uint32_t phase) {
    asm volatile("{ .reg .pred p;\n"
                 "L%=:\n"
                 "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
                 "@!p bra.uni L%=;\n}"
                 :: "r"(bar), "r"(phase) : "memory");
}

// MODE 0: independent overwrite (issue interval).
// MODE 1: accumulate into the same TMEM tile (true accumulator dependency).
// MODE 2: one MMA + commit/wait per inner iteration (completion round trip).
template<int N, int MODE>
__global__ void k_tcgen05(int iters, uint64_t* out) {
    __shared__ Shared s;
    for (int i = threadIdx.x; i < M*K; i += blockDim.x) s.a[i] = half_t(1.0f);
    for (int i = threadIdx.x; i < N*K; i += blockDim.x) s.b[i] = half_t(1.0f);
    if (threadIdx.x == 0) {
        s.done = 0;
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
                     :: "r"(smem_u32(&s.done)) : "memory");
    }
    __syncthreads();

    // tcgen05.alloc must be issued uniformly by one full warp.
    if (threadIdx.x < 32) {
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                     :: "r"(smem_u32(&s.tmem)), "r"(TMEM_COLS) : "memory");
    }
    __syncthreads();

    // Construct canonical unswizzled K-major descriptors using CuTe's checked helper.
    auto la = tile_to_shape(UMMA::Layout_K_INTER_Atom<half_t>{}, make_shape(Int<M>{}, Int<K>{}));
    auto lb = tile_to_shape(UMMA::Layout_K_INTER_Atom<half_t>{}, make_shape(Int<N>{}, Int<K>{}));
    auto ta = make_tensor(make_smem_ptr(s.a), la);
    auto tb = make_tensor(make_smem_ptr(s.b), lb);
    uint64_t ad = uint64_t(UMMA::make_umma_desc<UMMA::Major::K>(ta));
    uint64_t bd = uint64_t(UMMA::make_umma_desc<UMMA::Major::K>(tb));
    auto id = UMMA::make_instr_desc<half_t, half_t, float,
                                    M, N, UMMA::Major::K, UMMA::Major::K>();
    uint32_t idesc = static_cast<uint32_t>(id);
    uint32_t bar = smem_u32(&s.done);

    __syncthreads();
    if (threadIdx.x == 0) {
        uint64_t t0, t1;
        uint32_t phase = 0;
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
        for (int i = 0; i < iters; ++i) {
            #pragma unroll
            for (int u = 0; u < UNROLL; ++u) {
                uint32_t accumulate = MODE == 0 ? 0u : 1u;
                uint32_t mask = 0;
                asm volatile(
                    "{ .reg .pred p; setp.ne.b32 p, %4, 0;\n"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, "
                    "{%5,%5,%5,%5}, p; }"
                    :: "r"(s.tmem), "l"(ad), "l"(bd), "r"(idesc), "r"(accumulate), "r"(mask)
                    : "memory");
                if constexpr (MODE == 2) {
                    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                                 :: "r"(bar) : "memory");
                    wait_phase(bar, phase);
                    phase ^= 1;
                }
            }
        }
        if constexpr (MODE != 2) {
            asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                         :: "r"(bar) : "memory");
            wait_phase(bar, phase);
        }
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
        out[0] = t1 - t0;
        out[1] = s.tmem; // observable allocation result; prevents dead-code removal.
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                     :: "r"(s.tmem), "r"(TMEM_COLS) : "memory");
    }
}
} // namespace tcbench

using namespace tcbench;

int main(int argc, char** argv) {
    dev_header(arg_dev(argc, argv));
    int dev = arg_dev(argc, argv);
    cudaDeviceProp p{}; CK(cudaGetDeviceProperties(&p, dev));
    if (p.major != 11 || p.minor != 0) {
        fprintf(stderr, "06b_tcgen05 requires sm_110 (found sm_%d%d)\n", p.major, p.minor);
        return 2;
    }

    sec("Blackwell tcgen05 Tensor Core latency (M64, K16, F16->F32)");
    hdr("周期", "说明");
#define RUN_SHAPE(N_) do { \
    char name[128]; \
    snprintf(name, sizeof name, "M64N%dK16 overwrite 连发 + 最终 commit/wait", N_); \
    rowf(name, slope([&](int n){ k_tcgen05<N_,0><<<1,128>>>(n,g_out); }, 64,256,UNROLL), 1.0, \
         "无 accumulator 依赖: 稳态间隔"); \
    snprintf(name, sizeof name, "M64N%dK16 同一 TMEM accumulator 依赖链", N_); \
    rowf(name, slope([&](int n){ k_tcgen05<N_,1><<<1,128>>>(n,g_out); }, 64,256,UNROLL), 1.0, \
         "D=A*B+D, 最终 commit/wait"); \
    snprintf(name, sizeof name, "M64N%dK16 每条 commit + mbarrier wait", N_); \
    rowf(name, slope([&](int n){ k_tcgen05<N_,2><<<1,128>>>(n,g_out); }, 16,64,UNROLL), 1.0, \
         "端到端完成, 含 commit/wait"); \
} while (0)
    RUN_SHAPE(8);
    RUN_SHAPE(16);
    RUN_SHAPE(32);
#undef RUN_SHAPE

    int rc = sentinel_report();
    CK(cudaFree(g_out));
    return rc;
}

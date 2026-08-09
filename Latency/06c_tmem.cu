// Blackwell Tensor Memory (TMEM) instruction latency microbenchmark.
// Build: make 06c_tmem   Run: ./06c_tmem --dev 0
#include "common.cuh"
#include <cute/atom/mma_traits_sm100.hpp>
#include <cute/numeric/numeric_types.hpp>

using namespace cute;

namespace tmem_bench {
constexpr int COLS = 32;

struct alignas(16) Shared {
    uint32_t tmem;
};

struct alignas(128) CopyShared {
    half_t src[128 * 16];
    uint32_t tmem;
    uint64_t done;
};

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void tmem_alloc(uint32_t* smem_dst) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                 :: "r"(smem_u32(smem_dst)), "r"(COLS) : "memory");
}
__device__ __forceinline__ void tmem_free(uint32_t addr) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                 :: "r"(addr), "r"(COLS) : "memory");
}

__device__ __forceinline__ void wait_phase(uint32_t bar, uint32_t phase) {
    asm volatile("{ .reg .pred p; L%=: "
                 "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1; "
                 "@!p bra.uni L%=; }"
                 :: "r"(bar), "r"(phase) : "memory");
}
__device__ __forceinline__ void tmem_ld(uint32_t addr,
                                        uint32_t& a, uint32_t& b,
                                        uint32_t& c, uint32_t& d) {
    asm volatile("tcgen05.ld.sync.aligned.16x256b.x1.b32 "
                 "{%0,%1,%2,%3}, [%4];"
                 : "=r"(a), "=r"(b), "=r"(c), "=r"(d) : "r"(addr) : "memory");
}
__device__ __forceinline__ void tmem_st(uint32_t addr,
                                        uint32_t a, uint32_t b,
                                        uint32_t c, uint32_t d) {
    asm volatile("tcgen05.st.sync.aligned.16x256b.x1.b32 "
                 "[%0], {%1,%2,%3,%4};"
                 :: "r"(addr), "r"(a), "r"(b), "r"(c), "r"(d) : "memory");
}

enum Mode { LD, ST, ST_WAIT, ST_LD, WAIT_LD, WAIT_ST };

template<Mode MODE>
__global__ void k_tmem(int iters, uint64_t* out, uint32_t runtime_zero) {
    __shared__ Shared s;
    if (threadIdx.x < 32) tmem_alloc(&s.tmem);
    __syncthreads();

    uint32_t a = 0x12345678u + threadIdx.x, b = a + 1, c = a + 2, d = a + 3;
    if (threadIdx.x < 32) {
        tmem_st(s.tmem, a,b,c,d);
        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
    }
    __syncthreads();

    uint32_t addr = s.tmem;
    uint64_t t0 = 0, t1 = 0;
    if (threadIdx.x < 32) {
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
        for (int i = 0; i < iters; ++i) {
            #pragma unroll
            for (int u = 0; u < UNROLL; ++u) {
                if constexpr (MODE == LD) {
                    tmem_ld(addr, a,b,c,d);
                    // Actual runtime_zero is 0, but ptxas cannot prove it. This makes
                    // every load address depend on the preceding load result.
                    addr = s.tmem ^ (a & runtime_zero);
                } else if constexpr (MODE == ST) {
                    tmem_st(s.tmem, a,b,c,d);
                } else if constexpr (MODE == ST_WAIT) {
                    tmem_st(s.tmem, a,b,c,d);
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                } else if constexpr (MODE == ST_LD) {
                    tmem_st(s.tmem, a,b,c,d);
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    tmem_ld(s.tmem, a,b,c,d);
                } else if constexpr (MODE == WAIT_LD) {
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                } else if constexpr (MODE == WAIT_ST) {
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                }
            }
        }
        if constexpr (MODE == ST) asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        out[0] = t1 - t0;
        out[1] = uint64_t(a) + b + c + d + addr;
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
        tmem_free(s.tmem);
    }
}

__global__ void k_alloc(int iters, uint64_t* out) {
    __shared__ Shared s;
    uint64_t t0 = 0, t1 = 0;
    if (threadIdx.x < 32) {
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
        for (int i = 0; i < iters; ++i) {
            tmem_alloc(&s.tmem);
            tmem_free(s.tmem);
        }
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
    }
    if (threadIdx.x == 0) { out[0] = t1-t0; out[1] = s.tmem; }
}

template<bool WAIT_EACH>
__global__ void k_cp(int iters, uint64_t* out) {
    __shared__ CopyShared s;
    for (int i = threadIdx.x; i < 128*16; i += blockDim.x) s.src[i] = half_t(1.0f);
    if (threadIdx.x == 0) {
        s.done = 0;
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
                     :: "r"(smem_u32(&s.done)) : "memory");
    }
    __syncthreads();
    if (threadIdx.x < 32) tmem_alloc(&s.tmem);
    __syncthreads();

    auto layout = tile_to_shape(UMMA::Layout_K_INTER_Atom<half_t>{},
                                make_shape(Int<128>{}, Int<16>{}));
    auto tensor = make_tensor(make_smem_ptr(s.src), layout);
    uint64_t desc = uint64_t(UMMA::make_umma_desc<UMMA::Major::K>(tensor));
    uint32_t bar = smem_u32(&s.done);

    if (threadIdx.x == 0) {
        uint64_t t0, t1; uint32_t phase = 0;
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
        for (int i = 0; i < iters; ++i) {
            #pragma unroll
            for (int u = 0; u < UNROLL; ++u) {
                asm volatile("tcgen05.cp.cta_group::1.128x256b [%0], %1;"
                             :: "r"(s.tmem), "l"(desc) : "memory");
                if constexpr (WAIT_EACH) {
                    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                                 :: "r"(bar) : "memory");
                    wait_phase(bar, phase); phase ^= 1;
                }
            }
        }
        if constexpr (!WAIT_EACH) {
            asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                         :: "r"(bar) : "memory");
            wait_phase(bar, phase);
        }
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
        out[0] = t1-t0; out[1] = s.tmem;
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
        tmem_free(s.tmem);
    }
}
} // namespace tmem_bench

using namespace tmem_bench;

int main(int argc, char** argv) {
    dev_header(arg_dev(argc, argv));
    cudaDeviceProp p{}; CK(cudaGetDeviceProperties(&p, arg_dev(argc, argv)));
    if (p.major != 11 || p.minor != 0) {
        fprintf(stderr, "06c_tmem requires sm_110 (found sm_%d%d)\n", p.major, p.minor);
        return 2;
    }
    sec("Blackwell Tensor Memory (TMEM) latency");
    hdr("周期", "口径");
    rowf("tcgen05.alloc + dealloc (32 columns)",
         slope([&](int n){ k_alloc<<<1,32>>>(n,g_out); }, 16,64,1), 1.0,
         "同一 full warp, 每轮一对");
    rowf("tcgen05.ld 16x256b.x1 (TMEM->register)",
         slope([&](int n){ k_tmem<LD><<<1,32>>>(n,g_out,0); }, 64,256,UNROLL), 1.0,
         "同步 load, 寄存器复用依赖链");
    rowf("tcgen05.st 16x256b.x1 连发",
         slope([&](int n){ k_tmem<ST><<<1,32>>>(n,g_out,0); }, 64,256,UNROLL), 1.0,
         "最终 wait::st");
    rowf("tcgen05.st + wait::st",
         slope([&](int n){ k_tmem<ST_WAIT><<<1,32>>>(n,g_out,0); }, 32,128,UNROLL), 1.0,
         "每条等待完成");
    rowf("TMEM store->load round trip",
         slope([&](int n){ k_tmem<ST_LD><<<1,32>>>(n,g_out,0); }, 32,128,UNROLL), 1.0,
         "st + wait::st + ld");
    rowf("tcgen05.wait::ld (无在飞 load)",
         slope([&](int n){ k_tmem<WAIT_LD><<<1,32>>>(n,g_out,0); }, 64,256,UNROLL), 1.0);
    rowf("tcgen05.wait::st (无在飞 store)",
         slope([&](int n){ k_tmem<WAIT_ST><<<1,32>>>(n,g_out,0); }, 64,256,UNROLL), 1.0);
    rowf("tcgen05.cp 128x256b (SMEM->TMEM) 连发",
         slope([&](int n){ k_cp<false><<<1,128>>>(n,g_out); }, 32,128,UNROLL), 1.0,
         "最终 commit/mbarrier wait");
    rowf("tcgen05.cp 128x256b 每条 commit/wait",
         slope([&](int n){ k_cp<true><<<1,128>>>(n,g_out); }, 8,32,UNROLL), 1.0,
         "端到端完成");
    int rc = sentinel_report();
    CK(cudaFree(g_out));
    return rc;
}

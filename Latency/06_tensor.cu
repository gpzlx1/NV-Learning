// ============================================================================
//  06_tensor.cu —— 【指令延迟 · tensor core】
//    mma.sync   : warp 级 (32 线程), 累加器纯依赖链, 无描述符
//    wgmma      : warpgroup 级 (必须 128 线程), 需要 smem 描述符
//  数值不正确无所谓 —— 累加器依赖链是真的, 测延迟够用。
//  公开参考值(H800 PCIe): mma.m16n8k16 = 24.1 [p1 T7]
//                         wgmma.m64n8k16 LAT(RS)=13.0 LAT(SS)=18.0 [p1 T10][p2 T11]
//  编译: make      运行: ./06_tensor --dev 3
// ============================================================================
#include "common.cuh"
// mma.sync: warp 级(32 线程), 累加器纯依赖链, 无描述符
__global__ void k_mma(int iters, uint64_t* out, uint32_t seed)
{
    uint32_t a0 = 0x3c003c00u + threadIdx.x, a1 = a0, a2 = a0, a3 = a0;
    uint32_t b0 = 0x3c003c00u + seed, b1 = b0;
    float d0 = 0, d1 = 0, d2 = 0, d3 = 0;
    uint64_t t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
    for (int i = 0; i < iters; ++i) {
        #pragma unroll
        for (int u = 0; u < UNROLL; ++u)
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                         "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                         : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                         : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
    out[0] = t1 - t0; out[1] = (uint64_t)(d0 + d1 + d2 + d3);
}

__device__ __forceinline__ uint64_t smem_desc(const void* p, uint32_t lbo, uint32_t sbo)
{
    uint32_t ad = (uint32_t)__cvta_generic_to_shared(p);
    uint64_t d = 0;
    d |= (uint64_t)((ad  & 0x3FFFFu) >> 4);
    d |= (uint64_t)((lbo & 0x3FFFFu) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFFu) >> 4) << 32;
    return d;                                          // swizzle=0, base_off=0
}
// wgmma: warpgroup 级(必须 128 线程). MODE 0 = 连发不等(依赖发射率),
//        1 = 每条都 commit+wait (完全串行完成)
// 数值不正确无所谓 —— 累加器依赖链是真的, 测延迟够用
template<int MODE>
__global__ void k_wgmma(int iters, uint64_t* out, uint32_t seed)
{
    __shared__ __align__(128) uint32_t bsm[512];
    for (int i = threadIdx.x; i < 512; i += blockDim.x) bsm[i] = 0x3c003c00u + seed;
    __syncthreads();
    uint64_t bd = smem_desc(bsm, 128, 256);
    uint32_t a0 = 0x3c003c00u, a1 = a0, a2 = a0, a3 = a0;
    float d0 = 0, d1 = 0, d2 = 0, d3 = 0;
    float e0 = 0, e1 = 0, e2 = 0, e3 = 0;              // MODE 2 的第 2..4 组累加器
    float f0 = 0, f1 = 0, f2 = 0, f3 = 0;
    float g0 = 0, g1 = 0, g2 = 0, g3 = 0;
    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    uint64_t t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
    for (int i = 0; i < iters; ++i) {
        #pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            asm volatile("wgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 "
                         "{%0,%1,%2,%3}, {%4,%5,%6,%7}, %8, 1, 1, 1, 0;"
                         : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                         : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(bd) : "memory");
            if constexpr (MODE == 1) {
                asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
                asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
            }
            if constexpr (MODE == 2) {                 // 另外 3 组独立累加器
                asm volatile("wgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 "
                             "{%0,%1,%2,%3}, {%4,%5,%6,%7}, %8, 1, 1, 1, 0;"
                             : "+f"(e0),"+f"(e1),"+f"(e2),"+f"(e3)
                             : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"l"(bd));
                asm volatile("wgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 "
                             "{%0,%1,%2,%3}, {%4,%5,%6,%7}, %8, 1, 1, 1, 0;"
                             : "+f"(f0),"+f"(f1),"+f"(f2),"+f"(f3)
                             : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"l"(bd));
                asm volatile("wgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 "
                             "{%0,%1,%2,%3}, {%4,%5,%6,%7}, %8, 1, 1, 1, 0;"
                             : "+f"(g0),"+f"(g1),"+f"(g2),"+f"(g3)
                             : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"l"(bd));
            }
        }
    }
    if constexpr (MODE != 1) {
        asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
        asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
    if (threadIdx.x == 0) { out[0] = t1 - t0;
        out[1] = (uint64_t)(d0+d1+d2+d3+e0+e1+e2+e3+f0+f1+f2+f3+g0+g1+g2+g3); }
}

int main(int argc, char** argv)
{
    dev_header(arg_dev(argc, argv));
    sec("tensor core 依赖链延迟");
    hdr("周期", "公开参考值");
    rowf("mma.sync m16n8k16 f32.f16.f16.f32 (1 warp)",
         slope([&](int n){ k_mma<<<1,32>>>(n,g_out,3); },128,512,UNROLL), 1.0, "24.1 [p1 T7]");
    rowf("wgmma m64n8k16 RS 连发 (128 线程)",
         slope([&](int n){ k_wgmma<0><<<1,128>>>(n,g_out,3); },128,512,UNROLL), 1.0, "LAT(RS)=13.0 [p1 T10]");
    rowf("wgmma m64n8k16 RS 每条 commit+wait",
         slope([&](int n){ k_wgmma<1><<<1,128>>>(n,g_out,3); },128,512,UNROLL), 1.0);
    rowf("wgmma m64n8k16 RS 4 组独立累加器",
         slope([&](int n){ k_wgmma<2><<<1,128>>>(n,g_out,3); },128,512,UNROLL*4), 1.0,
         "三种写法同值: ptxas 每条后都插了 WARPGROUP.DEPBAR");
    int rc = sentinel_report();
    CK(cudaFree(g_out)); return rc;
}

// ============================================================================
//  07_sync.cu —— 【同步与一致性】barrier / fence / mbarrier / 原子
//
//    · block 内: barrier.sync (__syncthreads)
//    · cluster 内: barrier.cluster.arrive/wait, relaxed vs release
//    · 异步: mbarrier arrive + try_wait
//    · 内存栅栏: membar.cta / membar.gl
//    · 原子往返: atom 返回旧值 -> 天然形成依赖链, 测到 L2 原子单元的完整往返
//      (无返回值的 red.* 只测发射间隔)
//  编译: make      运行: ./07_sync --dev 3
// ============================================================================
#include "common.cuh"
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

// ═══════════ 4. mbarrier 纯开销 (arrive + try_wait, 不实际等数据) ═══════════
__global__ void k_mbar(int iters, uint64_t* out)
{
    __shared__ __align__(8) uint64_t mbar;
    uint32_t sb = (uint32_t)__cvta_generic_to_shared(&mbar);
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(sb) : "memory");
    __syncthreads();
    uint32_t ph = 0; uint64_t t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
    for (int i = 0; i < iters; ++i) {
        #pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(sb) : "memory");
            asm volatile("{ .reg .pred P;\n\t"
                         "M%=: mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t"
                         "@!P bra M%=;\n\t}" :: "r"(sb), "r"(ph) : "memory");
            ph ^= 1u;
        }
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
    out[0] = t1 - t0; out[1] = ph;
}

// ═══════════ 5. barrier.cluster (2 CTA), relaxed vs release ═══════════
template<int SEM>            // 0 = arrive.relaxed (SASS 只有 UCGABAR_ARV), 1 = arrive.release
__global__ void k_cbar(int iters, uint64_t* out)
{
    cg::cluster_group cl = cg::this_cluster();
    cl.sync();
    uint64_t t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
    for (int i = 0; i < iters; ++i) {
        #pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            if constexpr (SEM == 0) asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
            else                    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
            asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
        }
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
    if (cl.block_rank() == 0) { out[0] = t1 - t0; out[1] = 1; }
    cl.sync();
}

// ─── 栅栏与特殊寄存器的每条开销 ───
template<int OP>
__global__ void k_bar(uint64_t* out, int iters)
{
    uint64_t t0, t1, z = 0;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
    for (int i = 0; i < iters; ++i) {
        #pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            if constexpr (OP == 0) asm volatile("barrier.sync 0;" ::: "memory");
            if constexpr (OP == 1) asm volatile("membar.cta;" ::: "memory");
            if constexpr (OP == 2) asm volatile("membar.gl;"  ::: "memory");
            if constexpr (OP == 3) asm volatile("membar.sys;" ::: "memory");
            if constexpr (OP == 4) { uint64_t c;
                asm volatile("mov.u64 %0, %%clock64;" : "=l"(c) :: "memory"); z += c; }
        }
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
    out[0] = t1 - t0; out[1] = z;
}

// ─── 原子往返 (atom 返回旧值 -> 天然依赖链) 与 red 发射间隔 ───
template<int OP>
__global__ void k_atomic(uint64_t* out, int iters, uint32_t* gp)
{
    extern __shared__ uint32_t sm[];
    uint32_t sa = (uint32_t)__cvta_generic_to_shared(sm);
    uint32_t saa[UNROLL]; uint32_t* gaa[UNROLL];
    #pragma unroll
    for (int u = 0; u < UNROLL; ++u) { saa[u] = sa + u*128u; gaa[u] = gp + u*64u; }
    sm[0] = 1; __syncwarp();
    uint32_t v = 7u; uint64_t t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
    for (int i = 0; i < iters; ++i) {
        #pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            if constexpr (OP==0) asm volatile("atom.shared.add.u32 %0, [%1], %0;"      : "+r"(v) : "r"(sa) : "memory");
            if constexpr (OP==1) asm volatile("atom.global.add.u32 %0, [%1], %0;"      : "+r"(v) : "l"(gp) : "memory");
            if constexpr (OP==2) asm volatile("atom.global.cas.b32 %0, [%1], %0, %0;"  : "+r"(v) : "l"(gp) : "memory");
            if constexpr (OP==3) asm volatile("atom.global.exch.b32 %0, [%1], %0;"     : "+r"(v) : "l"(gp) : "memory");
            if constexpr (OP==4) asm volatile("atom.global.min.u32 %0, [%1], %0;"      : "+r"(v) : "l"(gp) : "memory");
            if constexpr (OP==5) asm volatile("red.global.add.u32 [%0], %1;"  :: "l"(gaa[u]), "r"(v) : "memory");
            if constexpr (OP==6) asm volatile("red.shared.add.u32 [%0], %1;"  :: "r"(saa[u]), "r"(v) : "memory");
        }
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
    out[0] = t1 - t0; out[1] = v;
}

// 原子延迟随"地址落在哪个 L2 分区"变化 —— 这解释了不同 buffer 上测同一个原子
// 操作能差 1.8 倍: L2 是双分区的(近 258 / 远 414 周期), 单个固定地址只会落在一边。
__global__ void k_atom_at(uint64_t* out, int iters, uint32_t* p)
{
    uint32_t v = 7u; uint64_t t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
    for (int i = 0; i < iters; ++i) {
        #pragma unroll
        for (int u = 0; u < UNROLL; ++u)
            asm volatile("atom.global.add.u32 %0, [%1], %0;" : "+r"(v) : "l"(p) : "memory");
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
    out[0] = t1 - t0; out[1] = v;
}

template<class K> static void run_cluster(K k, int cs, int iters) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = cs; cfg.blockDim = 1;
    cudaLaunchAttribute at[1] = {};
    at[0].id = cudaLaunchAttributeClusterDimension;
    at[0].val.clusterDim.x = cs; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
    cfg.attrs = at; cfg.numAttrs = 1;
    CK(cudaLaunchKernelEx(&cfg, k, iters, g_out));
}

int main(int argc, char** argv)
{
    dev_header(arg_dev(argc, argv));
    uint32_t* gp; CK(cudaMalloc(&gp, 1<<20)); CK(cudaMemset(gp, 0, 1<<20));
    const int SMB = 32<<10;

    sec("一、屏障与栅栏 (每条/每次的周期数)");
    hdr();
    rowf("barrier.sync (__syncthreads, 1 warp 无等待)",
         slope([&](int n){ k_bar<0><<<1,32>>>(g_out,n); },128,512,UNROLL), 0.5);
    rowf("membar.cta (__threadfence_block)",
         slope([&](int n){ k_bar<1><<<1,32>>>(g_out,n); },128,512,UNROLL), 0.5);
    rowf("membar.gl (__threadfence)",
         slope([&](int n){ k_bar<2><<<1,32>>>(g_out,n); },128,512,UNROLL), 0.5, "比 __syncthreads 贵 47 倍");
    rowf("membar.sys (__threadfence_system)",
         slope([&](int n){ k_bar<3><<<1,32>>>(g_out,n); },128,512,UNROLL), 0.5);
    rowf("mov %clock64 (读 SM 时钟)",
         slope([&](int n){ k_bar<4><<<1,32>>>(g_out,n); },128,512,UNROLL), 0.5);

    sec("二、cluster 与异步屏障");
    hdr();
    rowf("mbarrier arrive + try_wait (无实际等待)",
         slope([&](int n){ k_mbar<<<1,1>>>(n,g_out); },64,256,UNROLL), 1.0, "论文未测");
    rowf("barrier.cluster arrive.relaxed + wait (2 CTA)",
         slope([&](int n){ run_cluster(k_cbar<0>,2,n); },64,256,UNROLL), 1.0);
    rowf("barrier.cluster arrive.release + wait (2 CTA)",
         slope([&](int n){ run_cluster(k_cbar<1>,2,n); },64,256,UNROLL), 1.0,
         "默认语义! 会插 MEMBAR.ALL.GPU");

    sec("三、原子操作往返 (atom 返回旧值 -> 天然依赖链)");
    hdr();
    rowf("atom.shared.add.u32",  slope([&](int n){ k_atomic<0><<<1,1,SMB>>>(g_out,n,gp); },128,512,UNROLL), 1.0, "SMEM 内原子单元");
    rowf("atom.global.add.u32",  slope([&](int n){ k_atomic<1><<<1,1,SMB>>>(g_out,n,gp); },128,512,UNROLL), 1.0, "L2 原子单元");
    rowf("atom.global.cas.b32",  slope([&](int n){ k_atomic<2><<<1,1,SMB>>>(g_out,n,gp); },128,512,UNROLL), 1.0, "锁的基础");
    rowf("atom.global.exch.b32", slope([&](int n){ k_atomic<3><<<1,1,SMB>>>(g_out,n,gp); },128,512,UNROLL), 1.0);
    rowf("atom.global.min.u32",  slope([&](int n){ k_atomic<4><<<1,1,SMB>>>(g_out,n,gp); },128,512,UNROLL), 1.0);

    sec("四、无返回值原子的发射间隔 (red.*, 不等结果)");
    hdr("发射间隔");
    rowf("red.global.add.u32", slope([&](int n){ k_atomic<5><<<1,1,SMB>>>(g_out,n,gp); },128,512,UNROLL), 0.5,
         "不要返回值就用 red, 比 atom 快 70 倍");
    rowf("red.shared.add.u32", slope([&](int n){ k_atomic<6><<<1,1,SMB>>>(g_out,n,gp); },128,512,UNROLL), 0.5);

    sec("五、原子延迟随地址变化 (同一操作, 8 个相距 8MB 的地址)");
    printf("  L2 是双分区的(近 258 / 远 414 周期), 一个固定地址只会落在一边 ->\n");
    printf("  在不同 buffer 上测同一个原子操作, 结果可以差 1.8 倍。\n");
    hdr();
    uint32_t* big; CK(cudaMalloc(&big, 64ull<<20)); CK(cudaMemset(big, 0, 64ull<<20));
    double mn = 1e18, mx = 0;
    for (int k = 0; k < 8; ++k) {
        uint32_t* q = big + (size_t)k * (8ull<<20) / 4;
        double v = slope([&](int n){ k_atom_at<<<1,1>>>(g_out,n,q); },128,512,UNROLL);
        char nm[64]; snprintf(nm, sizeof nm, "atom.global.add @ 偏移 %d MB", k*8);
        rowf(nm, v, 1.0);
        mn = v < mn ? v : mn; mx = v > mx ? v : mx;
    }
    printf("  -> 最小 %.1f / 最大 %.1f 周期, 相差 %.2f 倍\n", mn, mx, mx/mn);
    CK(cudaFree(big));

    int rc = sentinel_report();
    CK(cudaFree(gp)); CK(cudaFree(g_out)); return rc;
}

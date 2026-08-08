// dsmem.cuh —— DSMEM (cluster 内跨 SM 访问对端 CTA 的 shared memory)
//  mapa.shared::cluster 把"本 CTA 的 smem 地址"翻译成"对端 CTA 同一偏移的 dsmem
//  地址", 映射是线性的 -> CTA0 可以直接把整条链写进对端, 再原地追逐。
//  MODE 0 = 读追逐, 1 = 写发射间隔, 2 = 写->读往返
#pragma once
#include "common.cuh"
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

static const int DS_BYTES  = 32 << 10;
static const int DS_STRIDE = 128;
static const int DS_SLOTS  = DS_BYTES / DS_STRIDE;
template<int MODE>
__global__ void k_dsmem(int slots, int stride, int peer_rank, int warm, int iters,
                        uint64_t* out, const uint32_t* nxt, uint32_t o1, uint32_t o2)
{
    extern __shared__ uint32_t sm[];
    cg::cluster_group cl = cg::this_cluster();
    uint32_t rank = cl.block_rank();
    uint32_t base = (uint32_t)__cvta_generic_to_shared(sm), peer;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(peer) : "r"(base), "r"(peer_rank));
    cl.sync();
    if (rank == 0 && MODE == 0)                            // 先把链写进对端
        for (int i = 0; i < slots; ++i) {
            uint32_t ad = peer + (uint32_t)i*stride, vl = peer + nxt[i]*(uint32_t)stride;
            asm volatile("st.shared::cluster.u32 [%0], %1;" :: "r"(ad), "r"(vl) : "memory");
        }
    cl.sync();
    if (rank == 0) {
        uint32_t v = 7u, a1 = peer + o1, a2 = peer + o2;
        uint32_t aa[UNROLL];
        #pragma unroll
        for (int u = 0; u < UNROLL; ++u) aa[u] = peer + u*128u;
        if constexpr (MODE == 0)
            for (int i = 0; i < warm; ++i)
                asm volatile("ld.shared::cluster.u32 %0, [%0];" : "+r"(v) :: "memory");
        if constexpr (MODE == 0) v = peer;
        uint64_t t0, t1;
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
        for (int i = 0; i < iters; ++i) {
            #pragma unroll
            for (int u = 0; u < UNROLL; ++u) {
                if constexpr (MODE == 0)
                    asm volatile("ld.shared::cluster.u32 %0, [%0];" : "+r"(v) :: "memory");
                if constexpr (MODE == 1)
                    asm volatile("st.shared::cluster.u32 [%0], %1;" :: "r"(aa[u]), "r"(v) : "memory");
                if constexpr (MODE == 2)
                    asm volatile("st.shared::cluster.u32 [%1], %0;"
                                 "ld.shared::cluster.u32 %0, [%2];"
                                 : "+r"(v) : "r"(a1), "r"(a2) : "memory");
            }
        }
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
        out[0] = t1 - t0; out[1] = v;
    }
    cl.sync();
}


static uint32_t* g_dsnxt = nullptr;
static void dsmem_init() {                       // 置换表(链表顺序), 只需建一次
    auto n = sattolo(DS_SLOTS);
    CK(cudaMalloc(&g_dsnxt, DS_SLOTS*4));
    CK(cudaMemcpy(g_dsnxt, n.data(), DS_SLOTS*4, cudaMemcpyHostToDevice));
}
// peer=0 表示读自己的 smem 但走 DSM 接口(对照用); peer=CS-1 表示读最远的那个 CTA
template<int MODE> static void dsmem_run(int cs, int peer, int iters) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = cs; cfg.blockDim = 1; cfg.dynamicSmemBytes = DS_BYTES;
    cudaLaunchAttribute at[1] = {};
    at[0].id = cudaLaunchAttributeClusterDimension;
    at[0].val.clusterDim.x = cs; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
    cfg.attrs = at; cfg.numAttrs = 1;
    CK(cudaLaunchKernelEx(&cfg, k_dsmem<MODE>, DS_SLOTS, DS_STRIDE, peer,
                          DS_SLOTS, iters, g_out, (const uint32_t*)g_dsnxt, 0u, 0u));
}

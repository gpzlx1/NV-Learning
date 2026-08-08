// ============================================================================
//  08_async_copy.cu —— 【异步数据搬运】cp.async / TMA 双向
//
//  异步拷贝的"延迟"定义为端到端: 发射 + commit + wait + 数据真的可用。
//  做法仍是指针追逐(把 [p] 处 16B 搬到 smem, 等完成, 再从 smem 读出下一个地址),
//  这样与普通 ld.global 追逐在同一 footprint 下直接可比。
//    · cp.async (Ampere 路径, global->shared)  .ca/.cg, 8/16B
//    · TMA cp.async.bulk global->shared  ("读方向", 配 mbarrier)
//    · TMA cp.async.bulk shared->global  ("写方向", 配 bulk_group)
//  公开参考值: TMA "比常规访存高约 170 cycle" [p2 5.1] (只有相对值);
//              cp.async 两篇论文都没测延迟(只有吞吐)
//  编译: make      运行: ./08_async_copy --dev 3
// ============================================================================
#include "common.cuh"
// ═══════════ 1. cp.async 指针追逐 (端到端: 发射+commit+wait+从 smem 读回) ═══════════
//  OP: 0=.ca 16B, 1=.cg 16B, 2=.ca 8B   (.cg 只支持 16B)
template<int OP>
__global__ void k_cpa(void** start, int warm, int iters, uint64_t* out)
{
    __shared__ __align__(16) uint64_t sm[4];
    uint32_t sd = (uint32_t)__cvta_generic_to_shared(sm);
    void** p = start;
    #define CPA_STEP                                                                  \
        if constexpr (OP==0) asm volatile("cp.async.ca.shared::cta.global [%0],[%1],16;"\
                                          :: "r"(sd), "l"(p) : "memory");             \
        if constexpr (OP==1) asm volatile("cp.async.cg.shared::cta.global [%0],[%1],16;"\
                                          :: "r"(sd), "l"(p) : "memory");             \
        if constexpr (OP==2) asm volatile("cp.async.ca.shared::cta.global [%0],[%1],8;" \
                                          :: "r"(sd), "l"(p) : "memory");             \
        asm volatile("cp.async.commit_group;" ::: "memory");                          \
        asm volatile("cp.async.wait_group 0;" ::: "memory");                          \
        asm volatile("ld.shared.u64 %0, [%1];" : "=l"(p) : "r"(sd) : "memory");
    for (int i = 0; i < warm; ++i) { CPA_STEP }
    uint64_t t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
    for (int i = 0; i < iters; ++i) { _Pragma("unroll")
        for (int u = 0; u < UNROLL; ++u) { CPA_STEP } }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
    out[0] = t1 - t0; out[1] = (uint64_t)p;
    #undef CPA_STEP
}

// 同步等价路径基线: ld.global -> st.shared -> ld.shared (终点同样是"指针过了一趟 smem")
// o1/o2 运行时相等但编译期不可证明 -> 禁止 store-to-load forwarding
__global__ void k_sync_via_smem(void** start, int warm, int iters, uint64_t* out,
                               uint32_t o1, uint32_t o2)
{
    __shared__ __align__(16) uint64_t sm[4];
    uint32_t b = (uint32_t)__cvta_generic_to_shared(sm), a1 = b + o1, a2 = b + o2;
    void** p = start;
    #define SYN_STEP                                                                  \
        asm volatile("{ .reg .u64 %%r;\n\t"                                           \
                     "ld.global.cg.u64 %%r, [%1];\n\t"                                \
                     "st.shared.u64 [%2], %%r;\n\t"                                   \
                     "ld.shared.u64 %0, [%3];\n\t}"                                   \
                     : "=l"(p) : "l"(p), "r"(a1), "r"(a2) : "memory");
    for (int i = 0; i < warm; ++i) { SYN_STEP }
    uint64_t t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
    for (int i = 0; i < iters; ++i) { _Pragma("unroll")
        for (int u = 0; u < UNROLL; ++u) { SYN_STEP } }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
    out[0] = t1 - t0; out[1] = (uint64_t)p;
    #undef SYN_STEP
}

// cp.async 发射间隔: 只发不等 (16 个独立源地址)
__global__ void k_cpa_issue(void** addrs, int iters, uint64_t* out)
{
    __shared__ __align__(16) uint64_t sm[64];
    uint32_t sd = (uint32_t)__cvta_generic_to_shared(sm);
    void* a[UNROLL];
    #pragma unroll
    for (int u = 0; u < UNROLL; ++u) a[u] = addrs[u];
    uint64_t t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
    for (int i = 0; i < iters; ++i) { _Pragma("unroll")
        for (int u = 0; u < UNROLL; ++u)
            asm volatile("cp.async.cg.shared::cta.global [%0],[%1],16;"
                         :: "r"(sd + u*16u), "l"(a[u]) : "memory"); }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
    asm volatile("cp.async.wait_all;" ::: "memory");
    out[0] = t1 - t0; out[1] = sm[0];
}

// ═══════════ 2. TMA bulk global->shared 追逐 ("读方向", 配 mbarrier) ═══════════
__global__ void k_tma_r(void** start, int warm, int iters, uint64_t* out)
{
    __shared__ __align__(128) uint64_t sm[16];
    __shared__ __align__(8)   uint64_t mbar;
    uint32_t sb = (uint32_t)__cvta_generic_to_shared(&mbar);
    uint32_t sd = (uint32_t)__cvta_generic_to_shared(sm);
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(sb) : "memory");
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");   // 不可省
    __syncthreads();
    void** p = start; uint32_t ph = 0;
    #define TMA_STEP                                                                  \
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], 16;"          \
                     :: "r"(sb) : "memory");                                          \
        asm volatile("cp.async.bulk.shared::cluster.global"                            \
                     ".mbarrier::complete_tx::bytes [%0],[%1],%2,[%3];"                \
                     :: "r"(sd), "l"(p), "r"(16u), "r"(sb) : "memory");                \
        asm volatile("{ .reg .pred P;\n\t"                                             \
                     "W%=: mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t"  \
                     "@!P bra W%=;\n\t}" :: "r"(sb), "r"(ph) : "memory");              \
        ph ^= 1u;                                                                     \
        asm volatile("ld.shared.u64 %0, [%1];" : "=l"(p) : "r"(sd) : "memory");
    for (int i = 0; i < warm; ++i) { TMA_STEP }
    uint64_t t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
    for (int i = 0; i < iters; ++i) { _Pragma("unroll")
        for (int u = 0; u < UNROLL; ++u) { TMA_STEP } }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
    out[0] = t1 - t0; out[1] = (uint64_t)p + ph;
    #undef TMA_STEP
}

// ═══════════ 3. TMA bulk shared->global ("写方向", 配 bulk_group) ═══════════
//  WAITMODE 0 = wait_group.read (smem 可复用即返回), 1 = wait_group (完全完成)
template<int WAITMODE>
__global__ void k_tma_w(void* g, int iters, uint64_t* out)
{
    __shared__ __align__(128) uint64_t sm[16];
    sm[0] = 0x1234; __syncthreads();
    uint32_t ss = (uint32_t)__cvta_generic_to_shared(sm);
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");   // 不可省
    uint64_t t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
    for (int i = 0; i < iters; ++i) {
        #pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0],[%1],%2;"
                         :: "l"(g), "r"(ss), "r"(128u) : "memory");
            asm volatile("cp.async.bulk.commit_group;" ::: "memory");
            if constexpr (WAITMODE == 0)
                asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
            else
                asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
        }
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
    out[0] = t1 - t0; out[1] = sm[0];
}
// TMA 写的发射间隔: 只发不等
__global__ void k_tma_w_issue(void* g, int iters, uint64_t* out)
{
    __shared__ __align__(128) uint64_t sm[16];
    sm[0] = 1; __syncthreads();
    uint32_t ss = (uint32_t)__cvta_generic_to_shared(sm);
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    uint64_t t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
    for (int i = 0; i < iters; ++i) { _Pragma("unroll")
        for (int u = 0; u < UNROLL; ++u)
            asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0],[%1],%2;"
                         :: "l"(g), "r"(ss), "r"(128u) : "memory"); }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
    asm volatile("cp.async.bulk.commit_group;" ::: "memory");
    asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
    out[0] = t1 - t0; out[1] = sm[0];
}

int main(int argc, char** argv)
{
    dev_header(arg_dev(argc, argv));
    Chain L2 = build_chain(32ull<<20, 256);
    Chain HB = build_chain(2ull<<30,  512);
    void** a_l2 = rand_addrs(L2.buf, 32ull<<20, 256);

    sec("一、异步读: cp.async / TMA 端到端 (含 commit+wait+从 smem 读回)");
    printf("  同 footprint 的普通 ld.global.cg 追逐: L2=271.9  HBM=683.3 周期\n");
    hdr("周期", "公开参考值");
    rowf("[L2 ] ld.global->st.shared->ld.shared 同步基线",
         slope([&](int n){ k_sync_via_smem<<<1,1>>>(L2.buf,(int)L2.slots,n,g_out,0,0); },64,256,UNROLL), 1.0);
    rowf("[L2 ] cp.async .ca 16B",
         slope([&](int n){ k_cpa<0><<<1,1>>>(L2.buf,(int)L2.slots,n,g_out); },64,256,UNROLL), 1.0, "论文未测");
    rowf("[L2 ] cp.async .cg 16B",
         slope([&](int n){ k_cpa<1><<<1,1>>>(L2.buf,(int)L2.slots,n,g_out); },64,256,UNROLL), 1.0);
    rowf("[L2 ] cp.async .ca  8B",
         slope([&](int n){ k_cpa<2><<<1,1>>>(L2.buf,(int)L2.slots,n,g_out); },64,256,UNROLL), 1.0);
    rowf("[L2 ] TMA bulk g->s (含 mbarrier)",
         slope([&](int n){ k_tma_r<<<1,1>>>(L2.buf,(int)L2.slots,n,g_out); },64,256,UNROLL), 1.0, "+170 [p2 5.1]");
    rowf("[HBM] ld.global->st.shared->ld.shared 同步基线",
         slope([&](int n){ k_sync_via_smem<<<1,1>>>(HB.buf,(int)HB.slots,n,g_out,0,0); },64,256,UNROLL), 1.0);
    rowf("[HBM] cp.async .cg 16B",
         slope([&](int n){ k_cpa<1><<<1,1>>>(HB.buf,(int)HB.slots,n,g_out); },64,256,UNROLL), 1.0);
    rowf("[HBM] TMA bulk g->s (含 mbarrier)",
         slope([&](int n){ k_tma_r<<<1,1>>>(HB.buf,(int)HB.slots,n,g_out); },64,256,UNROLL), 1.0, "+170 [p2 5.1]");

    sec("二、异步写: TMA bulk shared->global");
    hdr();
    rowf("TMA s->g 128B 发射间隔 (只发不等)",
         slope([&](int n){ k_tma_w_issue<<<1,1>>>(L2.buf,n,g_out); },64,256,UNROLL), 0.5,
         "比普通 8B st.global(8.11) 还便宜");
    rowf("TMA s->g 128B + commit + wait.read",
         slope([&](int n){ k_tma_w<0><<<1,1>>>(L2.buf,n,g_out); },64,256,UNROLL), 1.0, "smem 可复用即返回");
    rowf("TMA s->g 128B + commit + wait",
         slope([&](int n){ k_tma_w<1><<<1,1>>>(L2.buf,n,g_out); },64,256,UNROLL), 1.0, "完全完成");
    rowf("cp.async .cg 16B 连发",
         slope([&](int n){ k_cpa_issue<<<1,1>>>(a_l2,n,g_out); },64,256,UNROLL), 0.5, "受在飞上限限制");

    int rc = sentinel_report();
    CK(cudaFree(L2.buf)); CK(cudaFree(HB.buf)); CK(cudaFree(a_l2)); CK(cudaFree(g_out));
    return rc;
}

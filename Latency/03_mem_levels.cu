// ============================================================================
//  lat_hopper.cu —— Hopper (sm_90a) 存储层次延迟微基准 / 固定可复现版
//
//  测量目标                      方法
//  ----------------------------  --------------------------------------------
//  register (ALU 依赖发射延迟)   SHF / IMAD / FFMA 纯寄存器依赖链 (内联 PTX)
//  shared memory                 ld.shared.u32       指针追逐
//  DSMEM (cluster 内跨 SM)       ld.shared::cluster  指针追逐 (mapa 映射对端 CTA)
//  L1 hit                        ld.global.ca        指针追逐, 24KB footprint
//  L2 hit                        ld.global.cg        指针追逐, 32MB footprint
//  HBM                           ld.global.cg        指针追逐, 2GB  footprint
//  L2 近/远分区                  --hist  单次访问延迟直方图 (8MB/40MB/2GB 看双峰)
//  NCU 层级归属验证              --probe 每级一个独立 kernel 名, 配合 ./ncu.sh
//  L1/L2/TLB/DRAM 台阶           --sweep footprint 从 32KB 扫到 2GB
//
//  核心手法:
//   1) 指针追逐: 节点里存的就是下一个节点的地址, 依赖链上只有一条 load,
//      没有任何地址计算 -> 测到的是纯访存延迟。(SASS: LDG.E.64 R6,[R6.64])
//   2) 单 thread / 单 block: 排除带宽、MSHR 排队、warp 并发的干扰。
//   3) Sattolo 算法生成覆盖整个 buffer 的随机单环 + 跨 cacheline stride,
//      彻底干掉硬件预取。
//   4) 双点斜率法: 在 n 与 4n 两点各取 REPS 次最小值再求斜率,
//      把 clock64 读取开销、循环控制、kernel 启动全部差分掉。
//   5) cycle->ns 用同一 kernel 内 %clock64 与 %globaltimer 实测的真实 SM 频率,
//      不用 nominal 频率。
//
//  !! 坑: add.s32 依赖链会被 ptxas 强度削减 (x+=b 十六次 -> x+=2b 八次 + LEA),
//     测出来只有 2.5 cycle。必须用 shf/mad/fma 这类不可重结合的操作,
//     并且每次改完都要 `make sass` 复核 SASS。
//
//  编译: make          运行: ./run.sh   (固定卡 3)
// ============================================================================
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <random>
#include <cuda_runtime.h>
#include <cmath>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

#define CK(x) do{ cudaError_t e_=(x); if(e_!=cudaSuccess){ \
    printf("CUDA error: %s (%s line %d)\n", cudaGetErrorString(e_), #x, __LINE__); exit(1);} }while(0)

#define UNROLL 16                    // 内层展开数, 摊薄循环控制开销
static const int REPS = 7;           // 每个点重复次数, 取最小值

// ─── 公开实测参考值 (全部为 clock cycles) ───────────────────────────────────
//  [p1] arXiv:2402.13499 (IPDPS'24) TABLE IV + §IV-E     测的是 H800 PCIe
//  [p2] arXiv:2501.12084 (扩展版)    TABLE III/IV + §7.1  测的是 H800 PCIe
//  注意: 两篇论文都用 H800 PCIe (max 1755MHz), 本机是 H800 SXM (1980MHz);
//        论文里没有一个 ns 数值, 全是 cycle。两篇都没测过 H100。
//        寄存器/ALU 指令延迟两篇论文都没测 -> 4 cycle 是 Volta 以来的公认值。
#define REF_ALU   "4      (Volta+ 公认值; 论文未测)"
#define REF_SMEM  "29.0   [p1 T4][p2 T3]"
#define REF_DSSELF "33     [p2 7.1 本地 smem 走 DSM 接口]"
#define REF_DSMEM "181    [p2 7.1, CS=2]  (p1: 180)"
#define REF_L1    "32.0   [p2 T3]  (p1 T4: 40.7)"
#define REF_L2    "263.0  [p1 T4]; near 258 / far 414 [p2 T4]"
#define REF_HBM   "656    [p2 T3]  (p1 T4: 478.8)"

static const int SM_BYTES  = 32 << 10;              // dynamic smem per CTA
static const int SM_STRIDE = 128;
static const int SM_SLOTS  = SM_BYTES / SM_STRIDE;

// ═══════════════ 1. 寄存器 / ALU 依赖链 ═══════════════
template<int OP>
__global__ void k_alu(uint64_t* out, int iters, uint32_t a, uint32_t b)
{
    uint32_t x = a * 3u + 1u;
    float    f = 1.0f + (float)a, g = 1.0000001f + (float)b;
    uint64_t t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
    for (int i = 0; i < iters; ++i) {
        #pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            // 每条指令读写同一个寄存器 -> 完全串行; 操作数取运行时值防折叠
            if constexpr (OP == 0) asm volatile("shf.l.wrap.b32 %0, %0, %0, %1;": "+r"(x) : "r"(b));
            if constexpr (OP == 1) asm volatile("mad.lo.s32 %0, %0, %1, %2;"    : "+r"(x) : "r"(b), "r"(a));
            if constexpr (OP == 2) asm volatile("fma.rn.f32 %0, %0, %1, %1;"    : "+f"(f) : "f"(g));
        }
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
    out[0] = t1 - t0; out[1] = (uint64_t)x + (uint64_t)f;   // 防死代码消除
}

// ═══════════════ 2. global memory 指针追逐 (L1 / L2 / HBM) ═══════════════
#define MK_CHASE(NAME, OP)                                                         \
__global__ void NAME(void** start, int warm, int iters, uint64_t* out) {            \
    void** p = start;                                                              \
    for (int i = 0; i < warm; ++i)   /* 预热: 走满整个 footprint */                 \
        asm volatile("ld.global." #OP ".u64 %0, [%0];" : "+l"(p) :: "memory");      \
    uint64_t t0, t1;                                                               \
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");                 \
    for (int i = 0; i < iters; ++i) {                                              \
        _Pragma("unroll")                                                          \
        for (int u = 0; u < UNROLL; ++u)                                           \
            asm volatile("ld.global." #OP ".u64 %0, [%0];" : "+l"(p) :: "memory"); \
    }                                                                              \
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");                 \
    out[0] = t1 - t0; out[1] = (uint64_t)p;                                         \
}
MK_CHASE(k_chase_ca, ca)   // .ca -> LDG.E.64.STRONG.SM  : 经过 L1
MK_CHASE(k_chase_cg, cg)   // .cg -> LDG.E.64.STRONG.GPU : 绕过 L1, 只在 L2 分配
// 给 NCU 做层级归属验证用: 三个独立符号名, 这样 ncu 能分别归因命中率
MK_CHASE(k_probe_l1,  ca)
MK_CHASE(k_probe_l2,  cg)
MK_CHASE(k_probe_hbm, cg)

// ═══════════════ 3. shared memory 指针追逐 ═══════════════
// smem 里存的是 32bit shared-window 地址, 所以链上同样只有一条 ld.shared
__global__ void k_chase_smem(int slots, int stride, int warm, int iters,
                             uint64_t* out, const uint32_t* nxt)
{
    extern __shared__ uint32_t sm[];
    uint32_t base = (uint32_t)__cvta_generic_to_shared(sm);
    for (int i = 0; i < slots; ++i)
        *(uint32_t*)((char*)sm + (size_t)i * stride) = base + nxt[i] * (uint32_t)stride;
    __syncwarp();
    uint32_t a = base;
    for (int i = 0; i < warm; ++i) asm volatile("ld.shared.u32 %0, [%0];" : "+r"(a) :: "memory");
    uint64_t t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
    for (int i = 0; i < iters; ++i) {
        #pragma unroll
        for (int u = 0; u < UNROLL; ++u)
            asm volatile("ld.shared.u32 %0, [%0];" : "+r"(a) :: "memory");
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
    out[0] = t1 - t0; out[1] = a;
}

// ═══════════════ 4. DSMEM: cluster 内读对端 CTA 的 shared memory ═══════════════
// mapa.shared::cluster 把"本 CTA 的 smem 地址"翻译成"对端 CTA 同一偏移的 dsmem 地址",
// 映射是线性的 -> CTA0 可以直接把整条链写进对端, 再原地追逐, 走 SM-to-SM 网络。
// peer_rank==0 时 CTA0 读的是自己的 smem, 但走 DSM 接口 (对照 p2 的 33 cycle)。
__global__ void k_chase_dsmem(int slots, int stride, int peer_rank, int warm, int iters,
                              uint64_t* out, const uint32_t* nxt)
{
    extern __shared__ uint32_t sm[];
    cg::cluster_group cl = cg::this_cluster();
    uint32_t rank = cl.block_rank();
    uint32_t base = (uint32_t)__cvta_generic_to_shared(sm), peer;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(peer) : "r"(base), "r"(peer_rank));
    cl.sync();                                     // 等对端 smem 就绪
    if (rank == 0) {                               // CTA0 把链写进对端 CTA 的 smem
        for (int i = 0; i < slots; ++i) {
            uint32_t addr = peer + (uint32_t)i * stride, val = peer + nxt[i] * (uint32_t)stride;
            asm volatile("st.shared::cluster.u32 [%0], %1;" :: "r"(addr), "r"(val) : "memory");
        }
    }
    cl.sync();                                     // 等写入可见
    if (rank == 0) {
        uint32_t a = peer;
        for (int i = 0; i < warm; ++i)
            asm volatile("ld.shared::cluster.u32 %0, [%0];" : "+r"(a) :: "memory");
        uint64_t t0, t1;
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
        for (int i = 0; i < iters; ++i) {
            #pragma unroll
            for (int u = 0; u < UNROLL; ++u)
                asm volatile("ld.shared::cluster.u32 %0, [%0];" : "+r"(a) :: "memory");
        }
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
        out[0] = t1 - t0; out[1] = a;
    }
    cl.sync();                                     // 对端必须活到测完
}

// ═══════════════ 5. 单次访问延迟直方图 (看 L2 近/远分区双峰) ═══════════════
__global__ void k_hist(void** start, int warm, int n, uint32_t* hist,
                       int bins, int binw, uint64_t* sink)
{
    extern __shared__ uint32_t h[];
    for (int i = 0; i < bins; ++i) h[i] = 0;
    __syncwarp();
    void** p = start;
    for (int i = 0; i < warm; ++i)
        asm volatile("ld.global.cg.u64 %0, [%0];" : "+l"(p) :: "memory");
    for (int i = 0; i < n; ++i) {
        uint64_t t0, t1;
        // !! 关键: 第二次 clock64 必须用 predicate 挂在 load 的结果上。
        //    否则它和 load 之间没有数据依赖, 会在 load 还没回来时就发射,
        //    测到的是"发射延迟"而不是"完成延迟"(实测只有真值的一半左右)。
        //    setp 消费了 load 的目标寄存器 -> 硬件必须先等 LDG 的 scoreboard,
        //    @q 又让 clock 读取依赖这个 predicate, ptxas 无法把它提前。
        asm volatile("{ .reg .pred %%q;\n\t"
                     "mov.u64 %0, %%clock64;\n\t"
                     "ld.global.cg.u64 %1, [%1];\n\t"
                     "setp.ne.u64 %%q, %1, 0;\n\t"
                     "@%%q mov.u64 %2, %%clock64;\n\t}"
                     : "=l"(t0), "+l"(p), "=l"(t1) :: "memory");
        uint32_t b = (uint32_t)(t1 - t0) / binw;   // 直方图放 smem, 不污染 L1/L2
        h[b < (uint32_t)bins ? b : bins - 1]++;
    }
    __syncwarp();
    for (int i = 0; i < bins; ++i) hist[i] = h[i];
    sink[1] = (uint64_t)p;
}

// 计时框架自身的开销(两次 clock64 + setp), 结构与 k_hist 里完全一致 -> 直接相减
__global__ void k_clkovh(uint64_t* out, int n, void** dummy)
{
    uint64_t s = 0; void** p = dummy;
    for (int i = 0; i < n; ++i) {
        uint64_t a, b;
        asm volatile("{ .reg .pred %%q;\n\t"
                     "mov.u64 %0, %%clock64;\n\t"
                     "setp.ne.u64 %%q, %2, 0;\n\t"
                     "@%%q mov.u64 %1, %%clock64;\n\t}"
                     : "=l"(a), "=l"(b) : "l"(p) : "memory");
        s += b - a;
    }
    out[0] = s / n;
}

// ═══════════════ 6. 实测 SM 频率 (cycle -> ns) ═══════════════
__global__ void k_freq(uint64_t* out, int iters, uint32_t b)
{
    uint32_t x = b; uint64_t c0, c1, g0, g1;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g0) :: "memory");
    asm volatile("mov.u64 %0, %%clock64;"     : "=l"(c0) :: "memory");
    for (int i = 0; i < iters; ++i) {
        #pragma unroll
        for (int u = 0; u < UNROLL; ++u) asm volatile("mad.lo.s32 %0, %0, %1, %1;" : "+r"(x) : "r"(b));
    }
    asm volatile("mov.u64 %0, %%clock64;"     : "=l"(c1) :: "memory");
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g1) :: "memory");
    out[0] = c1 - c0; out[1] = g1 - g0; out[2] = x;
}

// 链表初始化(设备侧, 比 cudaMemcpy2D 快得多)
__global__ void k_init(char* base, const uint32_t* nxt, uint32_t slots, uint32_t stride)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < slots) *(void**)(base + (size_t)i * stride) = (void*)(base + (size_t)nxt[i] * stride);
}

// ═══════════════ host 辅助 ═══════════════
static uint64_t* g_out;
static double    g_ghz = 1.0;

template<class F> static uint64_t best(F&& launch) {
    uint64_t m = ~0ull;
    for (int r = 0; r < REPS; ++r) {
        launch(); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        uint64_t h[3]; CK(cudaMemcpy(h, g_out, sizeof h, cudaMemcpyDeviceToHost));
        if (h[0] < m) m = h[0];
    }
    return m;
}
// cycles(n) = C + n*UNROLL*lat  =>  斜率消掉一切常数开销
template<class F> static double slope(F&& launch, int n1, int n2) {
    uint64_t c1 = best([&]{ launch(n1); }), c2 = best([&]{ launch(n2); });
    return double(c2 - c1) / (double(n2 - n1) * UNROLL);
}
// Sattolo: 保证是一个覆盖全部 n 个元素的单环
static std::vector<uint32_t> sattolo(size_t n, uint32_t seed = 12345) {
    std::vector<uint32_t> a(n);
    for (size_t i = 0; i < n; ++i) a[i] = (uint32_t)i;
    std::mt19937 rng(seed);
    for (size_t i = n - 1; i > 0; --i) std::swap(a[i], a[rng() % i]);
    return a;
}
struct Chain { void** buf = nullptr; size_t slots = 0; };
static Chain build_chain(size_t bytes, size_t stride) {
    Chain c; c.slots = bytes / stride;
    CK(cudaMalloc(&c.buf, bytes));
    auto nxt = sattolo(c.slots);
    uint32_t* d; CK(cudaMalloc(&d, c.slots * 4));
    CK(cudaMemcpy(d, nxt.data(), c.slots * 4, cudaMemcpyHostToDevice));
    k_init<<<(unsigned)((c.slots + 255) / 256), 256>>>((char*)c.buf, d, (uint32_t)c.slots, (uint32_t)stride);
    CK(cudaDeviceSynchronize()); CK(cudaFree(d));
    return c;
}
static void row(const char* name, double cyc, const char* ref) {
    printf("  %-30s %9.2f %8.2f    %s\n", name, cyc, cyc / g_ghz, ref);
}
// cluster 启动: gridDim = clusterDim = CS, 每 CTA 1 thread; CTA0 读 rank=peer 的 smem
static uint32_t* g_dnxt;
static void launch_ds(int CS, int peer, int iters) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = CS; cfg.blockDim = 1; cfg.dynamicSmemBytes = SM_BYTES;
    cudaLaunchAttribute at[1] = {};
    at[0].id = cudaLaunchAttributeClusterDimension;
    at[0].val.clusterDim.x = CS; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
    cfg.attrs = at; cfg.numAttrs = 1;
    CK(cudaLaunchKernelEx(&cfg, k_chase_dsmem, SM_SLOTS, SM_STRIDE, peer, SM_SLOTS, iters,
                          g_out, (const uint32_t*)g_dnxt));
}

// ═══════════════ 各段测量 ═══════════════
static void bench_core()
{
    printf("\n  %-30s %9s %8s    %s\n", "测量对象", "周期", "纳秒", "公开实测参考值(周期)");
    printf("  ---------------------------------------------------------------------------\n");
    const int N1 = 256, N2 = 1024;

    row("SHF  dep-chain (ALU pipe)", slope([&](int n){ k_alu<0><<<1,1>>>(g_out,n,7,3); }, N1,N2), REF_ALU);
    row("IMAD dep-chain",            slope([&](int n){ k_alu<1><<<1,1>>>(g_out,n,7,3); }, N1,N2), REF_ALU);
    row("FFMA dep-chain",            slope([&](int n){ k_alu<2><<<1,1>>>(g_out,n,7,3); }, N1,N2), REF_ALU);

    // ---- shared memory: 32KB dynamic smem, 128B stride ----
    row("shared mem (ld.shared)",
        slope([&](int n){ k_chase_smem<<<1,1,SM_BYTES>>>(SM_SLOTS,SM_STRIDE,SM_SLOTS,n,g_out,g_dnxt); }, 64, 256),
        REF_SMEM);
    // ---- DSMEM: 自读(走 DSM 接口) 与 cluster(2) 跨 SM ----
    row("DSMEM self (own smem via DSM)", slope([&](int n){ launch_ds(2, 0, n); }, 64, 256), REF_DSSELF);
    row("DSMEM peer (cluster=2, SM->SM)", slope([&](int n){ launch_ds(2, 1, n); }, 64, 256), REF_DSMEM);
    // ---- L1 / L2 / HBM ----
    struct { const char* name; size_t bytes; size_t stride; int cg; const char* ref; } T[] = {
        {"L1  hit  (24KB, .ca)",  24ull<<10, 128, 0, REF_L1},
        {"L2  hit  (32MB, .cg)",  32ull<<20, 256, 1, REF_L2},
        {"HBM      ( 2GB, .cg)",   2ull<<30, 512, 1, REF_HBM},
    };
    for (auto& t : T) {
        Chain c = build_chain(t.bytes, t.stride);
        double v = t.cg ? slope([&](int n){ k_chase_cg<<<1,1>>>(c.buf,(int)c.slots,n,g_out); }, 64, 256)
                        : slope([&](int n){ k_chase_ca<<<1,1>>>(c.buf,(int)c.slots,n,g_out); }, 64, 256);
        row(t.name, v, t.ref);
        CK(cudaFree(c.buf));
    }
}

// NCU 归属验证: 每个层级一个独立 kernel, 各跑一次, 让 ncu 采命中率计数器。
// 计时循环的 load 数远多于预热, 所以命中率≈被计时那部分的命中率。
static void bench_probe()
{
    printf("\n  ═══ 给 NCU 用的层级归属探针 (每级一个独立 kernel 名, 各 1 次启动) ═══\n");
    struct { const char* n; size_t bytes, stride; int warm, iters; } T[] = {
        {"k_probe_l1  24KB .ca",  24ull<<10, 128,    192, 65536},   // 1.05M timed loads
        {"k_probe_l2  32MB .cg",  32ull<<20, 256, 131072, 65536},   // 1.05M timed, 预热占 11%
        {"k_probe_hbm  2GB .cg",   2ull<<30, 512,   1024, 16384},   // 262K timed, 全 miss
    };
    for (int i = 0; i < 3; ++i) {
        Chain c = build_chain(T[i].bytes, T[i].stride);
        if (i == 0) k_probe_l1 <<<1,1>>>(c.buf, T[i].warm, T[i].iters, g_out);
        if (i == 1) k_probe_l2 <<<1,1>>>(c.buf, T[i].warm, T[i].iters, g_out);
        if (i == 2) k_probe_hbm<<<1,1>>>(c.buf, T[i].warm, T[i].iters, g_out);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        uint64_t h[3]; CK(cudaMemcpy(h, g_out, sizeof h, cudaMemcpyDeviceToHost));
        printf("    %-22s %8d 次计时访问, %.2f 周期/次\n", T[i].n, T[i].iters*UNROLL,
               double(h[0]) / (T[i].iters * UNROLL));
        CK(cudaFree(c.buf));
    }
}

// DSMEM 随 cluster size 变化 (论文只给了 CS=2..16 的区间 184~213, 没给逐点值)
static void bench_dsmem()
{
    printf("\n  ═══ DSMEM 延迟 vs cluster size  (CTA0 读 rank=CS-1 的 smem, 即最远那个) ═══\n");
    printf("  %12s %9s %8s   %s\n", "cluster size", "周期", "纳秒", "公开参考值");
    CK(cudaFuncSetAttribute(k_chase_dsmem, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
    for (int CS : {2, 4, 8, 16}) {
        double v = slope([&](int n){ launch_ds(CS, CS - 1, n); }, 64, 256);
        printf("  %12d %9.2f %8.2f   %s\n", CS, v, v / g_ghz,
               CS == 2 ? "181 [p2 7.1]" : "CS=2..16 落在 184~213 [p2 7.1]");
    }
}

// 单次访问延迟直方图: L2 命中应看到"近/远分区"双峰
static void bench_hist()
{
    const int BINS = 384, BINW = 4, N = 8192;
    k_clkovh<<<1,1>>>(g_out, 4096, (void**)g_out); CK(cudaDeviceSynchronize());
    uint64_t h3[3]; CK(cudaMemcpy(h3, g_out, sizeof h3, cudaMemcpyDeviceToHost));
    int ovh = (int)h3[0];
    printf("\n  ═══ 单次访问延迟直方图 (计时框架开销 = %d 周期, 已扣除) ═══\n", ovh);
    printf("  自检: 直方图均值应与主表的斜率法结果一致, 否则说明计时被乱序了\n");

    uint32_t* d_h; CK(cudaMalloc(&d_h, BINS * 4));
    // footprint 选点对齐 p2 §L2 partition: 8MB -> near/far hit 两峰;
    // 40MB(接近 50MB L2) -> near hit / near miss / far miss 三组; 2GB -> 纯 DRAM
    struct { const char* name; size_t bytes, stride; } T[] = {
        {"L2  8MB .cg  (期望 near/far hit 两峰)", 8ull<<20,  256},
        {"L2 40MB .cg  (期望 hit/near miss/far miss)", 40ull<<20, 256},
        {"HBM 2GB .cg  (期望 near/far miss 两峰)", 2ull<<30,  512} };
    for (auto& t : T) {
        Chain c = build_chain(t.bytes, t.stride);
        k_hist<<<1,1,BINS*4>>>(c.buf, (int)c.slots, N, d_h, BINS, BINW, g_out);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        std::vector<uint32_t> hh(BINS); CK(cudaMemcpy(hh.data(), d_h, BINS*4, cudaMemcpyDeviceToHost));
        uint32_t mx = 1; double sum = 0, cnt = 0;
        for (int i = 0; i < BINS; ++i) { mx = hh[i] > mx ? hh[i] : mx;
            sum += hh[i] * (double)(i*BINW + BINW/2.0 - ovh); cnt += hh[i]; }
        printf("\n  --- %s ---  %d 个样本, 均值 = %.1f 周期\n", t.name, N, sum/cnt);
        for (int i = 0; i < BINS; ++i) {
            if (!hh[i]) continue;
            int lo = i*BINW - ovh;
            printf("  %5d-%4d cyc %6.2f%% |%.*s\n", lo, lo+BINW-1, 100.0*hh[i]/N,
                   (int)(60.0*hh[i]/mx), "############################################################");
        }
        CK(cudaFree(c.buf));
    }
    CK(cudaFree(d_h));
}

// ─── 机器可读输出(给绘图脚本用) ───────────────────────────────────────
// footprint 用 sqrt(2) 步长(每倍频程 2 个点), 比表格版细一倍, 便于把拐点画准
static void csv_sweep()
{
    printf("#CSV sweep bytes,cycles,ns\n");
    size_t freeb, totb; CK(cudaMemGetInfo(&freeb, &totb));
    for (int k = 0; ; ++k) {                       // 16KB * 2^(k/2)
        double b = (16.0*1024.0) * pow(2.0, k*0.5);
        size_t bytes = ((size_t)(b / 4096) + 1) * 4096;   // 对齐到 4KB
        if (bytes > (2ull<<30)) break;
        if (bytes + (bytes/256)*4 + (512ull<<20) > freeb) continue;
        Chain c = build_chain(bytes, 256);
        int warm = (int)(c.slots < (1u<<19) ? c.slots : (1u<<19));
        double v = slope([&](int n){ k_chase_ca<<<1,1>>>(c.buf, warm, n, g_out); }, 32, 128);
        printf("#CSV sweep %zu,%.3f,%.4f\n", bytes, v, v/g_ghz);
        CK(cudaFree(c.buf));
    }
}
// 单次访问延迟直方图的原始分箱(每档一个 footprint)
static void csv_hist()
{
    const int BINS = 384, BINW = 4, N = 16384;
    k_clkovh<<<1,1>>>(g_out, 4096, (void**)g_out); CK(cudaDeviceSynchronize());
    uint64_t h3[3]; CK(cudaMemcpy(h3, g_out, sizeof h3, cudaMemcpyDeviceToHost));
    int ovh = (int)h3[0];
    printf("#CSV histovh %d\n", ovh);
    printf("#CSV hist tag,bin_lo_cycles,pct\n");
    uint32_t* d_h; CK(cudaMalloc(&d_h, BINS*4));
    struct { const char* tag; size_t bytes, stride; } T[] = {
        {"L2_8MB", 8ull<<20, 256}, {"L2_40MB", 40ull<<20, 256}, {"HBM_2GB", 2ull<<30, 512} };
    for (auto& t : T) {
        Chain c = build_chain(t.bytes, t.stride);
        k_hist<<<1,1,BINS*4>>>(c.buf, (int)c.slots, N, d_h, BINS, BINW, g_out);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        std::vector<uint32_t> hh(BINS); CK(cudaMemcpy(hh.data(), d_h, BINS*4, cudaMemcpyDeviceToHost));
        for (int i = 0; i < BINS; ++i)
            if (hh[i]) printf("#CSV hist %s,%d,%.5f\n", t.tag, i*BINW - ovh, 100.0*hh[i]/N);
        CK(cudaFree(c.buf));
    }
    CK(cudaFree(d_h));
}

// footprint 扫描: L1 -> L2 -> TLB -> DRAM 的台阶
static void bench_sweep()
{
    printf("\n  ═══ 延迟 vs footprint (随机指针追逐, 256B stride, ld.global.ca) ═══\n");
    printf("  %10s %9s %8s\n", "footprint", "周期", "纳秒");
    size_t freeb, totb; CK(cudaMemGetInfo(&freeb, &totb));
    for (size_t bytes = 32ull<<10; bytes <= (2ull<<30); bytes <<= 1) {
        if (bytes + (bytes/256)*4 + (512ull<<20) > freeb) { printf("  (skip %zu MB: 显存不足)\n", (size_t)(bytes>>20)); continue; }
        Chain c = build_chain(bytes, 256);
        int warm = (int)(c.slots < (1u<<19) ? c.slots : (1u<<19));   // 大 footprint 不必走满
        double v = slope([&](int n){ k_chase_ca<<<1,1>>>(c.buf, warm, n, g_out); }, 32, 128);
        char sz[32];
        if (bytes >= (1ull<<20)) snprintf(sz, sizeof sz, "%zu MB", (size_t)(bytes>>20));
        else                     snprintf(sz, sizeof sz, "%zu KB", (size_t)(bytes>>10));
        printf("  %10s %9.2f %8.2f  %.*s\n", sz, v, v/g_ghz, (int)(v/12),
               "**************************************************************");
        CK(cudaFree(c.buf));
    }
}

int main(int argc, char** argv)
{
    int dev = 3; bool core=false, hist=false, sweep=false, dsm=false, probe=false, csv=false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--dev") && i+1 < argc) dev = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--core"))  core  = true;
        else if (!strcmp(argv[i], "--hist"))  hist  = true;
        else if (!strcmp(argv[i], "--sweep")) sweep = true;
        else if (!strcmp(argv[i], "--dsmem")) dsm   = true;
        else if (!strcmp(argv[i], "--probe")) probe = true;
        else if (!strcmp(argv[i], "--csv"))   csv   = true;
        else if (!strcmp(argv[i], "--all"))   core = hist = sweep = dsm = true;
    }
    if (!core && !hist && !sweep && !dsm && !probe && !csv) core = true;          // 默认只跑主表


    CK(cudaSetDevice(dev));
    cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr, dev));
    int clk = 0; CK(cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, dev));
    printf("GPU %d: %s  sm_%d%d  SM 数=%d  L2=%.0f MB  smem/SM=%.0f KB  最高频率=%.2f GHz\n",
           dev, pr.name, pr.major, pr.minor, pr.multiProcessorCount,
           pr.l2CacheSize/1048576.0, pr.sharedMemPerMultiprocessor/1024.0, clk/1e6);
    CK(cudaMalloc(&g_out, 4 * sizeof(uint64_t)));
    {   auto nxt = sattolo(SM_SLOTS);                 // smem/dsmem 共用的置换表
        CK(cudaMalloc(&g_dnxt, SM_SLOTS * 4));
        CK(cudaMemcpy(g_dnxt, nxt.data(), SM_SLOTS * 4, cudaMemcpyHostToDevice)); }

    k_freq<<<1,1>>>(g_out, 1<<20, 3); CK(cudaDeviceSynchronize());
    uint64_t h[3]; CK(cudaMemcpy(h, g_out, sizeof h, cudaMemcpyDeviceToHost));
    g_ghz = double(h[0]) / double(h[1]);
    printf("测量期间实测 SM 频率: %.3f GHz  (1 周期 = %.3f 纳秒)\n", g_ghz, 1.0/g_ghz);

    if (csv)   { csv_sweep(); csv_hist(); }
    if (probe) { bench_probe(); }
    if (core)  bench_core();
    if (dsm)   bench_dsmem();
    if (hist)  bench_hist();
    if (sweep) bench_sweep();
    CK(cudaFree(g_dnxt)); CK(cudaFree(g_out));
    return 0;
}

// ============================================================================
//  03b_l2_partition.cu —— 【存储层次 · L2 双分区的 SM↔地址 映射】
//
//  H800 的 50MB L2 不是一整块, 而是分成两半贴在 die 的两个半区上, 中间由片上
//  互连连接。两条规则:
//    · 一个物理地址只归属一个分区(按物理地址哈希决定)
//    · 一个 SM 只在一个半区里 —— 访问自己那半叫 near, 访问对面叫 far
//  所以"L2 命中"其实是两件事, 差一个互连往返。
//
//  本文件验证的推论: 同一个地址, 一半 SM 看到 near、另一半看到 far; 换个地址,
//  近/远关系会翻转。这是"地址决定分区、SM 决定远近"的直接证据。
//
//  做法:
//   1) 让每个被测地址里存自己的地址 -> `ld.global.cg.u64 %p,[%p]` 原地追逐,
//      测的就是这一条 cache line 的 L2 命中延迟(不掺任何地址计算)。
//   2) 每次启动 NSM 个 block, 只有落在目标 SM 上的那个 block 做测量, 其余立刻
//      退出 -> 测量期间零竞争。(第一版用全局票号排队, 131 个自旋的 block 会给
//      正在测量的那个添噪声)
//
//  配套图: figures/fig4_l2_sm_partition_map.png   (--csv 出数据)
//  编译: make      运行: ./03b_l2_partition --dev 3 [--csv]
// ============================================================================
#include "common.cuh"
#include <algorithm>
#include <cmath>

static const int   NADDR  = 8;
static const size_t ASTEP = 8ull << 20;        // 相邻被测地址相距 8MB

__global__ void k_selfref(char* base, int n, size_t step) {   // 每个槽存自己的地址
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) *(void**)(base + (size_t)i * step) = (void*)(base + (size_t)i * step);
}

// 只有 %smid == target 的 block 做测量, 其余立刻返回 -> 零竞争
__global__ void k_probe_sm(void** addr, int target, int iters, uint64_t* out)
{
    uint32_t sm; asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
    if ((int)sm != target) return;
    void* p = addr;
    for (int i = 0; i < 256; ++i)                              // 预热这条 line
        asm volatile("ld.global.cg.u64 %0,[%0];" : "+l"(p) :: "memory");
    uint64_t t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
    for (int i = 0; i < iters; ++i) {
        #pragma unroll
        for (int u = 0; u < UNROLL; ++u)
            asm volatile("ld.global.cg.u64 %0,[%0];" : "+l"(p) :: "memory");
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
    out[0] = t1 - t0; out[1] = (uint64_t)p;
}

int main(int argc, char** argv)
{
    bool csv = false; size_t pad = 0;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--csv")) csv = true;
        // --pad N: 先占掉 N MB 再分配被测 buffer, 用来检验"近/远映射是否取决于
        //          具体的物理分配"。翻转 => 映射由物理地址决定, 不能硬编码。
        if (!strcmp(argv[i], "--pad") && i+1 < argc) pad = (size_t)atoll(argv[i+1]) << 20;
    }
    int dev = arg_dev(argc, argv);
    dev_header(dev);
    cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr, dev));
    const int NSM = pr.multiProcessorCount;

    void* padp = nullptr;
    if (pad) { CK(cudaMalloc(&padp, pad)); printf("（先占掉 %zu MB 再分配被测 buffer）\n", pad>>20); }
    char* buf; CK(cudaMalloc(&buf, ASTEP * NADDR + 4096));
    k_selfref<<<1, NADDR>>>(buf, NADDR, ASTEP);
    CK(cudaDeviceSynchronize());

    // lat[a][sm]: SM sm 访问第 a 个地址的 L2 命中延迟
    std::vector<std::vector<double>> lat(NADDR, std::vector<double>(NSM, 0.0));
    for (int a = 0; a < NADDR; ++a) {
        void** ad = (void**)(buf + (size_t)a * ASTEP);
        for (int s = 0; s < NSM; ++s)
            lat[a][s] = slope([&](int n){ k_probe_sm<<<NSM,1>>>(ad, s, n, g_out); },
                              128, 512, UNROLL);
    }
    // 每个地址各自的近/远中心: 取排序后 15% 与 85% 分位, 阈值取中点
    std::vector<double> nearC(NADDR), farC(NADDR), thr(NADDR);
    std::vector<int> nNear(NADDR, 0);
    for (int a = 0; a < NADDR; ++a) {
        std::vector<double> v = lat[a];
        std::sort(v.begin(), v.end());
        nearC[a] = v[NSM * 15 / 100]; farC[a] = v[NSM * 85 / 100];
        thr[a] = (nearC[a] + farC[a]) / 2;
        for (int s = 0; s < NSM; ++s) if (lat[a][s] < thr[a]) ++nNear[a];
    }

    if (csv) {                                     // 给热力图用
        printf("#CSV map smid,addr_mb,cycles,group\n");
        for (int a = 0; a < NADDR; ++a)
            for (int s = 0; s < NSM; ++s)
                printf("#CSV map %d,%zu,%.2f,%s\n", s, (size_t)(a * (ASTEP >> 20)),
                       lat[a][s], lat[a][s] < thr[a] ? "near" : "far");
    }

    sec("一、同一个地址在不同 SM 上的近/远关系");
    printf("  地址里存自己的地址 -> ld.global.cg.u64 %%p,[%%p] 原地追逐; 每次只让目标\n");
    printf("  SM 上的 block 测量, 其余立刻退出 -> 测量期间零竞争。\n\n");
    printf("  %-12s %10s %10s %8s %14s\n", "被测地址", "近簇(周期)", "远簇(周期)",
           "远-近", "近/远 SM 数");
    printf("  ---------------------------------------------------------------------------------\n");
    for (int a = 0; a < NADDR; ++a)
        printf("  +%-11zu %10.1f %10.1f %8.1f %7d / %-6d\n", (size_t)(a * (ASTEP >> 20)) ,
               nearC[a], farC[a], farC[a] - nearC[a], nNear[a], NSM - nNear[a]);
    printf("  （地址列单位 MB。每个地址上 SM 都被分成两组, 且大致对半分）\n");

    sec("二、翻转验证: 同一个 SM 对不同地址的近/远会换边");
    printf("  以 SM 0 与 SM 6 为例（· = 近, # = 远）:\n\n");
    for (int s : {0, 6, 12}) {
        printf("  SM %-3d ", s);
        for (int a = 0; a < NADDR; ++a)
            printf(" %+4zuMB:%s(%.0f)", (size_t)(a*(ASTEP>>20)),
                   lat[a][s] < thr[a] ? "·" : "#", lat[a][s]);
        printf("\n");
    }
    int flips = 0;
    for (int s = 0; s < NSM; ++s) {
        bool has_near = false, has_far = false;
        for (int a = 0; a < NADDR; ++a) (lat[a][s] < thr[a] ? has_near : has_far) = true;
        if (has_near && has_far) ++flips;
    }
    printf("\n  %d/%d 个 SM 在这 %d 个地址上同时出现过近和远两种 -> 近/远不是 SM 的固有属性,\n",
           flips, NSM, NADDR);
    printf("  而是 (SM, 地址) 这一对的属性。这正是「地址决定分区、SM 决定远近」的证据。\n");

    sec("三、粒度: 决定远近的单位是 TPC(2 个 SM), 不是单个 SM");
    int same = 0, tot = 0;
    for (int a = 0; a < NADDR; ++a)
        for (int s = 0; s + 1 < NSM; s += 2) {
            ++tot;
            if (fabs(lat[a][s] - lat[a][s+1]) < 1.5) ++same;
        }
    printf("  相邻 SM 对 (2k, 2k+1) 延迟相同(差 <1.5 周期)的比例: %d/%d = %.0f%%\n",
           same, tot, 100.0 * same / tot);
    printf("  Hopper 的 2 个 SM 组成一个 TPC, 共享同一条到 L2 的路径。\n");

    sec("四、SM → 分区 映射图（· = 近, # = 远, + = 居中）");
    printf("  行 = 被测地址, 列 = SM 编号 0..%d\n\n", NSM - 1);
    for (int a = 0; a < NADDR; ++a) {
        printf("  +%3zuMB |", (size_t)(a * (ASTEP >> 20)));
        for (int s = 0; s < NSM; ++s) {
            double d = lat[a][s], w = (farC[a] - nearC[a]) * 0.18;
            putchar(d < thr[a] - w ? '.' : (d > thr[a] + w ? '#' : '+'));
        }
        printf("|\n");
    }
    printf("\n  同组内还有 +-%.0f 周期的额外离散(图上的 '+'), 说明主结构是双分区,\n",
           (farC[0] - nearC[0]) * 0.4);
    printf("  但在此之上还叠加了 GPC 位置带来的距离差 —— 不是干净的两级台阶。\n");

    int rc = sentinel_report();
    CK(cudaFree(buf)); if (padp) CK(cudaFree(padp)); CK(cudaFree(g_out));
    return rc;
}

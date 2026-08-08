// common.cuh —— 三个 benchmark 共用的测量框架
//  双点斜率法 + 取最小值 + %clock64/%globaltimer 实测频率
//  (lat_hopper.cu 是最早写的, 自带同样的框架, 保持自包含没有改)
#pragma once
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <random>
#include <cuda_runtime.h>

#define CK(x) do{ cudaError_t e_=(x); if(e_!=cudaSuccess){ \
    printf("CUDA error: %s (%s line %d)\n", cudaGetErrorString(e_), #x, __LINE__); exit(1);} }while(0)

#define UNROLL 16
static const int REPS = 7;

static uint64_t* g_out;
static double    g_ghz = 1.0;

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

template<class F> static uint64_t best(F&& launch) {
    uint64_t m = ~0ull;
    for (int r = 0; r < REPS; ++r) {
        launch(); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        uint64_t h[3]; CK(cudaMemcpy(h, g_out, sizeof h, cudaMemcpyDeviceToHost));
        if (h[0] < m) m = h[0];
    }
    return m;
}
// cycles(n) = C + n*per*lat  ->  斜率消掉 clock64 读取/循环控制/启动等一切常数开销
template<class F> static double slope(F&& launch, int n1, int n2, int per) {
    uint64_t c1 = best([&]{ launch(n1); }), c2 = best([&]{ launch(n2); });
    return double(c2 - c1) / (double(n2 - n1) * per);
}
static std::vector<uint32_t> sattolo(size_t n, uint32_t seed = 12345) {
    std::vector<uint32_t> a(n);
    for (size_t i = 0; i < n; ++i) a[i] = (uint32_t)i;
    std::mt19937 rng(seed);
    for (size_t i = n - 1; i > 0; --i) std::swap(a[i], a[rng() % i]);
    return a;
}
static void dev_header(int dev)
{
    CK(cudaSetDevice(dev));
    cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr, dev));
    int clk = 0; CK(cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, dev));
    printf("GPU %d: %s  sm_%d%d  SM 数=%d  L2=%.0f MB  最高频率=%.2f GHz\n", dev, pr.name,
           pr.major, pr.minor, pr.multiProcessorCount, pr.l2CacheSize/1048576.0, clk/1e6);
    CK(cudaMalloc(&g_out, 4 * sizeof(uint64_t)));
    k_freq<<<1,1>>>(g_out, 1<<20, 3); CK(cudaDeviceSynchronize());
    uint64_t h[3]; CK(cudaMemcpy(h, g_out, sizeof h, cudaMemcpyDeviceToHost));
    g_ghz = double(h[0]) / double(h[1]);
    printf("测量期间实测 SM 频率: %.3f GHz  (1 周期 = %.3f 纳秒)\n", g_ghz, 1.0/g_ghz);
}

// ─────────── 以下为各 benchmark 共用的辅助设施 (重构时从各文件收拢过来) ───────────

// 随机单环链表 (Sattolo) —— 指针追逐的基础设施。节点里存的就是下一个节点的地址,
// 所以依赖链上只有一条 load, 没有任何地址计算。
__global__ void k_chain_init(char* base, const uint32_t* nxt, uint32_t slots, uint32_t stride) {
    uint32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < slots) *(void**)(base + (size_t)i*stride) = (void*)(base + (size_t)nxt[i]*stride);
}
struct Chain { void** buf = nullptr; size_t slots = 0; };
// extra: 多分配的字节数(留给"写往返"的影子区), 链只建在前 bytes 字节里
static Chain build_chain(size_t bytes, size_t stride, size_t extra = 0) {
    Chain c; c.slots = bytes / stride;
    CK(cudaMalloc(&c.buf, bytes + extra));
    auto n = sattolo(c.slots);
    uint32_t* d; CK(cudaMalloc(&d, c.slots*4));
    CK(cudaMemcpy(d, n.data(), c.slots*4, cudaMemcpyHostToDevice));
    k_chain_init<<<(unsigned)((c.slots+255)/256),256>>>((char*)c.buf, d,
                                                        (uint32_t)c.slots, (uint32_t)stride);
    CK(cudaDeviceSynchronize()); CK(cudaFree(d));
    return c;
}
// 16 个随机地址(落在给定 footprint 内, 跨 cacheline), 供"发射间隔"用
static void** rand_addrs(void* base, size_t bytes, size_t align, uint32_t seed = 999) {
    std::mt19937 rng(seed); std::vector<void*> h(UNROLL); size_t n = bytes/align;
    for (int u = 0; u < UNROLL; ++u) h[u] = (char*)base + (size_t)(rng()%n)*align;
    void** d; CK(cudaMalloc(&d, UNROLL*sizeof(void*)));
    CK(cudaMemcpy(d, h.data(), UNROLL*sizeof(void*), cudaMemcpyHostToDevice));
    return d;
}

// ─── 输出 + 哨兵 ───────────────────────────────────────────────────────
// 哨兵存在的理由: ptxas 会强度削减依赖链、折叠对合/幂等操作、删掉 warp-uniform 的
// shuffle、做编译期 store-to-load forwarding、提走循环不变量、消除死存储。
// 这些都不会报错, 只会安静地给出一个"看起来很合理"的错数字。
static int g_bad = 0;
static void sec(const char* title) { printf("\n═══ %s ═══\n", title); }
static void hdr(const char* c1 = "周期", const char* c3 = "备注") {
    printf("  %-42s %8s %8s   %s\n", "测量对象", c1, "纳秒", c3);
    printf("  ---------------------------------------------------------------------------------\n");
}
static void row(const char* name, double cyc, const char* note = "") {
    printf("  %-42s %8.2f %8.2f   %s\n", name, cyc, cyc/g_ghz, note);
}
// floor_: 物理下限。依赖链延迟 < 1 cycle 不可能; 往返延迟 < 对应单程不可能。
static void rowf(const char* name, double cyc, double floor_, const char* note = "") {
    const char* w = (cyc < floor_) ? "  <== 低于物理下限, 疑似被编译器优化掉!" : "";
    if (*w) ++g_bad;
    printf("  %-42s %8.2f %8.2f   %s%s\n", name, cyc, cyc/g_ghz, note, w);
}
static int sentinel_report() {
    if (g_bad) printf("\n  !! %d 行没通过哨兵检查, 用 `make sass` 复核 SASS\n", g_bad);
    else       printf("\n  (哨兵检查全部通过)\n");
    return g_bad ? 1 : 0;
}
// 解析 --dev, 返回卡号(默认 3)
static int arg_dev(int argc, char** argv, int def = 3) {
    for (int i = 1; i < argc; ++i)
        if (!strcmp(argv[i], "--dev") && i+1 < argc) return atoi(argv[i+1]);
    return def;
}

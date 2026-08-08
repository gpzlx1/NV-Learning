// ============================================================================
//  lat_p2p.cu —— P2P / NVLink 读写延迟 (H800, sm_90a)
//
//  kernel 跑在 SRC 卡上, 内存在 DST 卡上, 经 NVLink 访问。默认 SRC=6 DST=7。
//  (本机 nvidia-smi topo -m 显示任意两卡都是 NV18, 即 18 条 NVLink 经 NVSwitch)
//
//  测什么:
//   A) 读: 指针追逐, 三档 footprint (24KB / 32MB / 2GB) x cache operator
//      -> footprint 那一档能直接回答"远端的行会不会被本地 L2 缓存"
//   B) 写: 发射间隔 / 写->读往返
//   C) 原子: atom.global.add 打在远端显存上 (多卡同步原语的真实代价)
//   D) 方向对称性: 6->7 与 7->6
//   全部都配同 footprint 的本地基线, 以及 host pinned (PCIe) 作为对照。
//
//  本地对照值 (卡3 实测): L1 32.0  L2 271.9  HBM 683.3  host pinned 2514 cycles
//
//  编译: make lat_p2p     运行: ./lat_p2p --src 6 --dst 7
// ============================================================================
#include "common.cuh"

// ── 读: 指针追逐 (链上只有一条 load) ──
#define MK_CHASE(NAME, INSN)                                                     \
__global__ void NAME(void** start, int warm, int iters, uint64_t* out) {          \
    void** p = start;                                                            \
    for (int i = 0; i < warm; ++i) asm volatile(INSN : "+l"(p) :: "memory");      \
    uint64_t t0, t1;                                                             \
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");               \
    for (int i = 0; i < iters; ++i) { _Pragma("unroll")                          \
        for (int u = 0; u < UNROLL; ++u) asm volatile(INSN : "+l"(p) :: "memory");}\
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");               \
    out[0] = t1 - t0; out[1] = (uint64_t)p; }

MK_CHASE(ch_ca,  "ld.global.ca.u64 %0, [%0];")
MK_CHASE(ch_cg,  "ld.global.cg.u64 %0, [%0];")
MK_CHASE(ch_cv,  "ld.global.cv.u64 %0, [%0];")
MK_CHASE(ch_nc,  "ld.global.nc.u64 %0, [%0];")
MK_CHASE(ch_gpu, "ld.relaxed.gpu.global.u64 %0, [%0];")
MK_CHASE(ch_sys, "ld.relaxed.sys.global.u64 %0, [%0];")

// ── 写: 发射间隔 (16 个独立地址, 不等完成) ──
#define MK_WI(NAME, INSN)                                                        \
__global__ void NAME(void** addrs, int iters, uint64_t* out, uint64_t seed) {     \
    void* a[UNROLL];                                                             \
    _Pragma("unroll") for (int u = 0; u < UNROLL; ++u) a[u] = addrs[u];           \
    uint64_t v = seed | 1u, t0, t1;                                               \
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");               \
    for (int i = 0; i < iters; ++i) { _Pragma("unroll")                          \
        for (int u = 0; u < UNROLL; ++u)                                          \
            asm volatile(INSN :: "l"(a[u]), "l"(v) : "memory"); }                 \
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");               \
    out[0] = t1 - t0; out[1] = v; }

MK_WI(wi_wb,  "st.global.wb.u64 [%0], %1;")
MK_WI(wi_cg,  "st.global.cg.u64 [%0], %1;")
MK_WI(wi_wt,  "st.global.wt.u64 [%0], %1;")
MK_WI(wi_sys, "st.relaxed.sys.global.u64 [%0], %1;")

// ── 写->读往返 / 原子: p1/p2 运行时相等但编译期不可证明 -> 禁止 store-to-load forwarding ──
#define MK_RT(NAME, INSN)                                                        \
__global__ void NAME(void* p1, void* p2, int iters, uint64_t* out, uint64_t seed) {\
    uint64_t v = seed | 1u, t0, t1;                                               \
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");               \
    for (int i = 0; i < iters; ++i) { _Pragma("unroll")                          \
        for (int u = 0; u < UNROLL; ++u)                                          \
            asm volatile(INSN : "+l"(v) : "l"(p1), "l"(p2) : "memory"); }         \
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");               \
    out[0] = t1 - t0; out[1] = v; }

MK_RT(rt_cg,  "st.global.cg.u64 [%1], %0; ld.global.cg.u64 %0, [%2];")
MK_RT(rt_sys, "st.relaxed.sys.global.u64 [%1], %0; ld.relaxed.sys.global.u64 %0, [%2];")
MK_RT(rt_rel, "st.release.sys.global.u64 [%1], %0; ld.acquire.sys.global.u64 %0, [%2];")

// 原子往返 (atom 返回旧值 -> 天然依赖链)
__global__ void k_atom(uint32_t* p, int iters, uint64_t* out, uint32_t seed, int op) {
    uint32_t v = seed | 1u; uint64_t t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
    for (int i = 0; i < iters; ++i) {
        #pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            if (op == 0) asm volatile("atom.global.add.u32 %0, [%1], %0;" : "+r"(v) : "l"(p) : "memory");
            else if (op == 1) asm volatile("atom.global.cas.b32 %0, [%1], %0, %0;" : "+r"(v) : "l"(p) : "memory");
            else asm volatile("atom.relaxed.sys.global.add.u32 %0, [%1], %0;" : "+r"(v) : "l"(p) : "memory");
        }
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
    out[0] = t1 - t0; out[1] = v;
}

__global__ void k_init(char* base, const uint32_t* nxt, uint32_t slots, uint32_t stride) {
    uint32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < slots) *(void**)(base + (size_t)i*stride) = (void*)(base + (size_t)nxt[i]*stride);
}

// ══════════════════════════ host ══════════════════════════
// 带哨兵的输出(下限固定 1 周期; 本文件所有量都是依赖链或往返)
static void rowf1(const char* n, double c, const char* note = "") { rowf(n, c, 1.0, note); }
static uint64_t* g_outs[16];
static int g_cur = -1;
static void use_dev(int d) { CK(cudaSetDevice(d)); g_out = g_outs[d]; g_cur = d; }

// 在 owner 卡上分配并初始化链 (在 owner 卡上跑 init kernel, 比经 NVLink 写快得多)
static Chain build_chain_on(int owner, size_t bytes, size_t stride, size_t extra = 0) {
    int back = g_cur;
    CK(cudaSetDevice(owner));
    Chain c; c.slots = bytes / stride;
    CK(cudaMalloc(&c.buf, bytes + extra));
    auto n = sattolo(c.slots);
    uint32_t* d; CK(cudaMalloc(&d, c.slots*4));
    CK(cudaMemcpy(d, n.data(), c.slots*4, cudaMemcpyHostToDevice));
    k_init<<<(unsigned)((c.slots+255)/256),256>>>((char*)c.buf, d, (uint32_t)c.slots, (uint32_t)stride);
    CK(cudaDeviceSynchronize()); CK(cudaFree(d));
    if (back >= 0) use_dev(back);
    return c;
}
static void** rand_addrs_on(int owner, void* base, size_t bytes, size_t align) {
    int back = g_cur;
    std::mt19937 rng(777); std::vector<void*> h(UNROLL); size_t n = bytes/align;
    for (int u = 0; u < UNROLL; ++u) h[u] = (char*)base + (size_t)(rng()%n)*align;
    CK(cudaSetDevice(owner));
    void** d; CK(cudaMalloc(&d, UNROLL*sizeof(void*)));
    CK(cudaMemcpy(d, h.data(), UNROLL*sizeof(void*), cudaMemcpyHostToDevice));
    if (back >= 0) use_dev(back);
    return d;
}

int main(int argc, char** argv)
{
    int src = 6, dst = 7;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--src") && i+1 < argc) src = atoi(argv[++i]);
        if (!strcmp(argv[i], "--dst") && i+1 < argc) dst = atoi(argv[++i]);
    }
    int can_sd = 0, can_ds = 0;
    CK(cudaDeviceCanAccessPeer(&can_sd, src, dst));
    CK(cudaDeviceCanAccessPeer(&can_ds, dst, src));
    printf("P2P: %d->%d %s   %d->%d %s\n", src, dst, can_sd ? "可以" : "不可以",
           dst, src, can_ds ? "可以" : "不可以");
    if (!can_sd) { printf("两卡之间不支持 P2P, 退出\n"); return 1; }

    for (int d : {src, dst}) {
        CK(cudaSetDevice(d));
        CK(cudaMalloc(&g_outs[d], 4*sizeof(uint64_t)));
    }
    CK(cudaSetDevice(src)); CK(cudaDeviceEnablePeerAccess(dst, 0));
    CK(cudaSetDevice(dst)); CK(cudaDeviceEnablePeerAccess(src, 0));

    use_dev(src);
    cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr, src));
    printf("计算卡 GPU %d: %s   内存卡 GPU %d\n", src, pr.name, dst);
    k_freq<<<1,1>>>(g_out, 1<<20, 3); CK(cudaDeviceSynchronize());
    uint64_t h[3]; CK(cudaMemcpy(h, g_out, sizeof h, cudaMemcpyDeviceToHost));
    g_ghz = double(h[0]) / double(h[1]);
    printf("计算卡实测 SM 频率: %.3f GHz  (1 周期 = %.3f 纳秒)\n", g_ghz, 1.0/g_ghz);

    // 本地(在 src 上) 与 远端(在 dst 上) 各三档 footprint
    struct { const char* tag; size_t bytes, stride; } F[] = {
        {"24KB", 24ull<<10, 128}, {"32MB", 32ull<<20, 256}, {"2GB", 2ull<<30, 512} };
    Chain loc[3], rem[3];
    for (int i = 0; i < 3; ++i) {
        loc[i] = build_chain_on(src, F[i].bytes, F[i].stride);
        rem[i] = build_chain_on(dst, F[i].bytes, F[i].stride);
    }

    // 确认每个 buffer 真的落在预期的卡上 —— "远端 24KB 只要 32 cycle" 这个结论
    // 太反直觉, 必须先排除 buffer 其实在本地的可能
    printf("\n  buffer 归属确认 (cudaPointerGetAttributes):\n");
    for (int i = 0; i < 3; ++i) {
        cudaPointerAttributes al{}, ar{};
        CK(cudaPointerGetAttributes(&al, loc[i].buf));
        CK(cudaPointerGetAttributes(&ar, rem[i].buf));
        printf("    %-5s  本地链在 GPU %d   远端链在 GPU %d\n", F[i].tag, al.device, ar.device);
    }

    // ─────────── A) 读 ───────────
    sec("一、读延迟: 本地 vs 经 NVLink 远端 (指针追逐)");
    hdr();
    char nm[96];
    for (int i = 0; i < 3; ++i) {
        snprintf(nm, sizeof nm, "本地 %s .ca (在 %d 上读 %d 的显存)", F[i].tag, src, src);
        rowf1(nm, slope([&](int n){ ch_ca<<<1,1>>>(loc[i].buf,(int)loc[i].slots,n,g_out); },64,256,UNROLL), "");
    }
    for (int i = 0; i < 3; ++i) {
        snprintf(nm, sizeof nm, "远端 %s .ca (在 %d 上读 %d 的显存)", F[i].tag, src, dst);
        int it2 = (i == 2) ? 64 : 256;
        rowf1(nm, slope([&](int n){ ch_ca<<<1,1>>>(rem[i].buf,(int)(i==2?4096:rem[i].slots),n,g_out); },16,it2,UNROLL),
             i == 0 ? "footprint 极小, 若远端行会被本地缓存这里会很快" : "");
    }

    sec("二、远端读 x cache operator (32MB footprint)");
    hdr();
    struct { const char* n; void(*k)(void**,int,int,uint64_t*); } CO[] = {
        {".ca", ch_ca}, {".cg", ch_cg}, {".cv 不缓存", ch_cv}, {".nc 非一致(__ldg)", ch_nc},
        {"relaxed.gpu 作用域", ch_gpu}, {"relaxed.sys 作用域", ch_sys} };
    for (auto& o : CO) {
        snprintf(nm, sizeof nm, "远端 32MB %s", o.n);
        rowf1(nm, slope([&](int n){ o.k<<<1,1>>>(rem[1].buf,(int)rem[1].slots,n,g_out); },64,256,UNROLL), "");
    }

    // ─────────── B) 写 ───────────
    // !! 写/原子必须打在独立的 scratch 上: 直接打在链表 buffer 上会改坏第一个
    //    节点的指针, 后面的指针追逐就 misaligned 了 (第一版就是这么崩的)
    void *scr_loc, *scr_rem;
    CK(cudaSetDevice(src)); CK(cudaMalloc(&scr_loc, 32ull<<20)); CK(cudaMemset(scr_loc,0,32ull<<20));
    CK(cudaSetDevice(dst)); CK(cudaMalloc(&scr_rem, 32ull<<20)); CK(cudaMemset(scr_rem,0,32ull<<20));
    use_dev(src);
    void** a_loc = rand_addrs_on(src, scr_loc, 32ull<<20, 256);
    void** a_rem = rand_addrs_on(src, scr_rem, 32ull<<20, 256);
    sec("三、写的发射间隔 (16 个独立地址, 不等完成)");
    hdr("发射间隔");
    rowf1("-> 本地 32MB  st.cg", slope([&](int n){ wi_cg<<<1,1>>>(a_loc,n,g_out,7); },128,512,UNROLL), "");
    rowf1("-> 远端 32MB  st.cg", slope([&](int n){ wi_cg<<<1,1>>>(a_rem,n,g_out,7); },128,512,UNROLL), "经 NVLink");
    rowf1("-> 远端 32MB  st.wb", slope([&](int n){ wi_wb<<<1,1>>>(a_rem,n,g_out,7); },128,512,UNROLL), "");
    rowf1("-> 远端 32MB  st.wt 写穿", slope([&](int n){ wi_wt<<<1,1>>>(a_rem,n,g_out,7); },128,512,UNROLL), "");
    rowf1("-> 远端 32MB  st.relaxed.sys", slope([&](int n){ wi_sys<<<1,1>>>(a_rem,n,g_out,7); },128,512,UNROLL), "");

    sec("四、写->读往返");
    hdr();
    rowf1("本地: st.cg -> ld.cg", slope([&](int n){ rt_cg<<<1,1>>>(scr_loc,scr_loc,n,g_out,7); },128,512,UNROLL), "");
    rowf1("远端: st.cg -> ld.cg", slope([&](int n){ rt_cg<<<1,1>>>(scr_rem,scr_rem,n,g_out,7); },128,512,UNROLL), "经 NVLink");
    rowf1("远端: st/ld.relaxed.sys", slope([&](int n){ rt_sys<<<1,1>>>(scr_rem,scr_rem,n,g_out,7); },128,512,UNROLL), "");
    rowf1("远端: st.release.sys -> ld.acquire.sys", slope([&](int n){ rt_rel<<<1,1>>>(scr_rem,scr_rem,n,g_out,7); },128,512,UNROLL), "跨卡传标志位");

    // ─────────── C) 原子 ───────────
    sec("五、原子往返 (多卡同步原语的真实代价)");
    hdr();
    rowf1("本地 atom.global.add.u32", slope([&](int n){ k_atom<<<1,1>>>((uint32_t*)scr_loc,n,g_out,7,0); },128,512,UNROLL), "");
    rowf1("远端 atom.global.add.u32", slope([&](int n){ k_atom<<<1,1>>>((uint32_t*)scr_rem,n,g_out,7,0); },128,512,UNROLL), "经 NVLink");
    rowf1("远端 atom.global.cas.b32", slope([&](int n){ k_atom<<<1,1>>>((uint32_t*)scr_rem,n,g_out,7,1); },128,512,UNROLL), "跨卡锁的基础");
    rowf1("远端 atom.relaxed.sys.add", slope([&](int n){ k_atom<<<1,1>>>((uint32_t*)scr_rem,n,g_out,7,2); },128,512,UNROLL), "");

    // ─────────── D) 方向对称性 ───────────
    sec("六、方向对称性 (32MB, .ca 追逐)");
    hdr();
    snprintf(nm, sizeof nm, "%d -> %d", src, dst);
    rowf1(nm, slope([&](int n){ ch_ca<<<1,1>>>(rem[1].buf,(int)rem[1].slots,n,g_out); },64,256,UNROLL), "");
    use_dev(dst);
    k_freq<<<1,1>>>(g_out, 1<<20, 3); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(h, g_out, sizeof h, cudaMemcpyDeviceToHost));
    double ghz_dst = double(h[0])/double(h[1]);
    snprintf(nm, sizeof nm, "%d -> %d  (DST 频率 %.3f GHz)", dst, src, ghz_dst);
    rowf1(nm, slope([&](int n){ ch_ca<<<1,1>>>(loc[1].buf,(int)loc[1].slots,n,g_out); },64,256,UNROLL), "");
    use_dev(src);

    sentinel_report();
    for (int i = 0; i < 3; ++i) {
        CK(cudaSetDevice(src)); CK(cudaFree(loc[i].buf));
        CK(cudaSetDevice(dst)); CK(cudaFree(rem[i].buf));
    }
    CK(cudaSetDevice(src)); CK(cudaFree(scr_loc));
    CK(cudaSetDevice(dst)); CK(cudaFree(scr_rem));
    return 0;
}

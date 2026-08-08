// ============================================================================
//  02_mem_write.cu —— 【存储层次 · 写延迟】
//
//  "写延迟"必须先说清测的是哪一个 —— store 本身是 posted(fire-and-forget)的,
//  三种定义的数值能差 40 倍以上:
//    (1) 发射间隔   不等完成, 每条 store 占多少发射槽。这是写在流水线里的真实占用。
//    (2) 写->读往返 生产者写完到消费者读到。这才是通常说的"写延迟"。
//    (3) 可见性代价 再加 fence / release-acquire, 保证跨 SM 或跨设备可见。
//  写的目标端: shared / DSMEM(对端) / local / L1 / L2 / HBM / host pinned
//
//  如何避免被编译期 store-to-load forwarding 干掉:
//    传两个"编译期无法证明相等、运行时却相同"的指针(p1/p2 指向同一处)。这样既
//    禁止转发, 又保留原生 STS/LDS/STG/LDG 指令(volatile/relaxed 会把
//    shared::cluster 降级成通用 ST/LD 路径)。
//  编译: make      运行: ./02_mem_write --dev 3
// ============================================================================
#include "common.cuh"
#include "dsmem.cuh"
// ══════════════ 写: (1) 发射间隔 —— 16 个独立地址, 不等完成 ══════════════
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

MK_WI(wi_wb,  "st.global.wb.u64 [%0], %1;")               // 默认(写回 L2)
MK_WI(wi_cg,  "st.global.cg.u64 [%0], %1;")               // 绕过 L1
MK_WI(wi_wt,  "st.global.wt.u64 [%0], %1;")               // write-through
MK_WI(wi_cs,  "st.global.cs.u64 [%0], %1;")               // streaming
MK_WI(wi_sys, "st.relaxed.sys.global.u64 [%0], %1;")      // 系统作用域(host)

// shared / local 的写发射间隔
__global__ void wi_smem(int iters, uint64_t* out, uint32_t seed) {
    extern __shared__ uint32_t sm[];
    uint32_t a[UNROLL];
    #pragma unroll
    for (int u = 0; u < UNROLL; ++u) a[u] = (uint32_t)__cvta_generic_to_shared(sm) + u*128u;
    uint32_t v = seed | 1u; uint64_t t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
    for (int i = 0; i < iters; ++i) { _Pragma("unroll")
        for (int u = 0; u < UNROLL; ++u)
            asm volatile("st.shared.u32 [%0], %1;" :: "r"(a[u]), "r"(v) : "memory"); }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
    out[0] = t1 - t0; out[1] = v;
}
__global__ void wi_local(int iters, uint64_t* out, uint32_t seed, const uint32_t* init) {
    uint32_t buf[512];
    #pragma unroll 1
    for (int i = 0; i < 512; ++i) buf[i] = init[i];
    // 反复写同一批地址会被死存储消除(只有最后一次有效), 必须用显式 PTX + volatile
    uint32_t la[UNROLL];
    #pragma unroll
    for (int u = 0; u < UNROLL; ++u) la[u] = (uint32_t)__cvta_generic_to_local(buf + u*31);
    uint32_t v = seed | 1u; uint64_t t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
    for (int i = 0; i < iters; ++i) { _Pragma("unroll")
        for (int u = 0; u < UNROLL; ++u)
            asm volatile("st.local.u32 [%0], %1;" :: "r"(la[u]), "r"(v) : "memory"); }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
    out[0] = t1 - t0;
    uint32_t s = 0;
    #pragma unroll 1
    for (int i = 0; i < 512; i += 37) s += buf[i];
    out[1] = s;
}

// ══════════ 写: (2) 写->读往返  p1/p2 指向同一处但编译期不可证明 ══════════
#define MK_RT(NAME, INSN)                                                        \
__global__ void NAME(void* p1, void* p2, int iters, uint64_t* out, uint64_t seed) {\
    uint64_t v = seed | 1u, t0, t1;                                               \
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");               \
    for (int i = 0; i < iters; ++i) { _Pragma("unroll")                          \
        for (int u = 0; u < UNROLL; ++u)                                          \
            asm volatile(INSN : "+l"(v) : "l"(p1), "l"(p2) : "memory"); }         \
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");               \
    out[0] = t1 - t0; out[1] = v; }

MK_RT(rt_l1,  "st.global.wb.u64 [%1], %0; ld.global.ca.u64 %0, [%2];")
MK_RT(rt_l2,  "st.global.cg.u64 [%1], %0; ld.global.cg.u64 %0, [%2];")
MK_RT(rt_mem, "st.global.wt.u64 [%1], %0; ld.global.cv.u64 %0, [%2];")   // 写穿+强制重取
MK_RT(rt_sys, "st.relaxed.sys.global.u64 [%1], %0; ld.relaxed.sys.global.u64 %0, [%2];")
MK_RT(rt_rel, "st.release.gpu.global.u64 [%1], %0; ld.acquire.gpu.global.u64 %0, [%2];")
MK_RT(rt_fen, "st.global.cg.u64 [%1], %0; membar.gl; ld.global.cg.u64 %0, [%2];")

// ── 真正的 HBM 写->读往返 ──────────────────────────────────────────────
// 固定地址的往返一定落在 L2 里(.cv 在 sm_90 上并不绕过 L2, 见 README), 所以
// 必须每次换一个全新的随机地址: A 区(1GB)指针追逐给出随机地址, B = A + OFF。
// !! 关键: B 的地址只依赖 A 的读结果, 下一次 A 读并不等 B 往返 -> 两条链
//    默认是并行的, 测出来是 max() 不是和。所以用一个"运行时为 0、编译期未知"
//    的掩码把 B 的结果异或回 p, 强制串成一条链。再减掉同样 ALU 开销的基线。
// off1/off2 运行时相等但编译期不可证明 -> 禁止 store-to-load forwarding。
template<int WITH_B>
__global__ void rt_hbm(void** start, int warm, int iters, uint64_t* out,
                       uint64_t off1, uint64_t off2, uint64_t zmask, uint64_t seed) {
    void** p = start; uint64_t v = seed | 1u;
    for (int i = 0; i < warm; ++i)
        asm volatile("ld.global.cg.u64 %0, [%0];" : "+l"(p) :: "memory");
    uint64_t t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
    for (int i = 0; i < iters; ++i) {
        #pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            if constexpr (WITH_B)
                asm volatile("{ .reg .u64 %%b1, %%b2, %%t;\n\t"
                             "ld.global.cg.u64 %0, [%0];\n\t"    // A: 追逐(随机地址)
                             "add.s64 %%b1, %0, %2;\n\t"
                             "add.s64 %%b2, %0, %3;\n\t"
                             "st.global.cg.u64 [%%b1], %1;\n\t"  // B: 写不在 L2 的行
                             "ld.global.cg.u64 %1, [%%b2];\n\t"  // B: 读回来
                             "and.b64 %%t, %1, %4;\n\t"          // zmask=0 -> t=0
                             "xor.b64 %0, %0, %%t;\n\t"          // p 不变但依赖 v
                             "}" : "+l"(p), "+l"(v)
                                 : "l"(off1), "l"(off2), "l"(zmask) : "memory");
            else                                                   // 基线: 同样的 ALU, 无 B
                asm volatile("{ .reg .u64 %%b1, %%b2, %%t;\n\t"
                             "ld.global.cg.u64 %0, [%0];\n\t"
                             "add.s64 %%b1, %0, %2;\n\t"
                             "add.s64 %%b2, %0, %3;\n\t"
                             "and.b64 %%t, %1, %4;\n\t"
                             "xor.b64 %0, %0, %%t;\n\t"
                             "}" : "+l"(p), "+l"(v)
                                 : "l"(off1), "l"(off2), "l"(zmask) : "memory");
        }
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
    out[0] = t1 - t0; out[1] = (uint64_t)p + v;
}

__global__ void rt_smem(uint32_t o1, uint32_t o2, int iters, uint64_t* out, uint32_t seed) {
    extern __shared__ uint32_t sm[];
    uint32_t b = (uint32_t)__cvta_generic_to_shared(sm), a1 = b + o1, a2 = b + o2;
    uint32_t v = seed | 1u; uint64_t t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
    for (int i = 0; i < iters; ++i) { _Pragma("unroll")
        for (int u = 0; u < UNROLL; ++u)
            asm volatile("st.shared.u32 [%1], %0; ld.shared.u32 %0, [%2];"
                         : "+r"(v) : "r"(a1), "r"(a2) : "memory"); }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
    out[0] = t1 - t0; out[1] = v;
}

int main(int argc, char** argv)
{
    dev_header(arg_dev(argc, argv));
    dsmem_init();
    uint32_t* d_lnxt; { auto n = sattolo(512); std::vector<uint32_t> b(512);
      for (int i = 0; i < 512; ++i) b[i] = n[i]*4u;
      CK(cudaMalloc(&d_lnxt, 512*4));
      CK(cudaMemcpy(d_lnxt, b.data(), 512*4, cudaMemcpyHostToDevice)); }

    Chain c1 = build_chain(24ull<<10, 128);
    Chain c2 = build_chain(32ull<<20, 256);
    Chain c4 = build_chain(1ull<<30, 512, 1ull<<30);   // A 区 1GB + B 区 1GB(写往返用)
    void* hp; CK(cudaHostAlloc(&hp, 32ull<<20, cudaHostAllocMapped));
    CK(cudaMemset(hp, 0, 32ull<<20));
    void** hchain; CK(cudaHostGetDevicePointer((void**)&hchain, hp, 0));

    void** a_l1  = rand_addrs(c1.buf, 24ull<<10, 128);
    void** a_l2  = rand_addrs(c2.buf, 32ull<<20, 256);
    // !! 写类测试的随机地址必须落在"影子区"(c4 的后 1GB), 不能落在链表区:
    //    写坏节点指针后, 下面的 rt_hbm 指针追逐会非法访问。(重构时又踩了一次)
    void** a_hbm = rand_addrs((char*)c4.buf + (1ull<<30), 1ull<<30, 512);
    void** a_hp  = rand_addrs(hchain, 32ull<<20, 4096);

    sec("一、写的发射间隔 —— fire-and-forget, 16 个独立地址不等完成");
    hdr("发射间隔");
    rowf("-> shared memory (STS)",  slope([&](int n){ wi_smem<<<1,1,DS_BYTES>>>(n,g_out,7); },128,512,UNROLL), 0.5);
    rowf("-> local memory (STL)",   slope([&](int n){ wi_local<<<1,1>>>(n,g_out,7,d_lnxt); },128,512,UNROLL), 0.5);
    rowf("-> DSMEM 对端 (跨 SM)",   slope([&](int n){ dsmem_run<1>(2,1,n); },128,512,UNROLL), 0.5);
    rowf("-> L1  (24KB, st.wb)",    slope([&](int n){ wi_wb<<<1,1>>>(a_l1,n,g_out,7); },128,512,UNROLL), 0.5);
    rowf("-> L2  (32MB, st.cg)",    slope([&](int n){ wi_cg<<<1,1>>>(a_l2,n,g_out,7); },128,512,UNROLL), 0.5);
    rowf("-> HBM (1GB,  st.cg)",    slope([&](int n){ wi_cg<<<1,1>>>(a_hbm,n,g_out,7); },128,512,UNROLL), 0.5, "远超 L2");
    rowf("-> HBM (1GB,  st.wt 写穿)",slope([&](int n){ wi_wt<<<1,1>>>(a_hbm,n,g_out,7); },128,512,UNROLL), 0.5);
    rowf("-> HBM (1GB,  st.cs 流式)",slope([&](int n){ wi_cs<<<1,1>>>(a_hbm,n,g_out,7); },128,512,UNROLL), 0.5);
    rowf("-> host pinned (经 PCIe)", slope([&](int n){ wi_sys<<<1,1>>>(a_hp,n,g_out,7); },32,128,UNROLL), 0.5,
         "与写到哪里无关, 这是本文件最反直觉的一条");

    sec("二、写->读往返 —— 生产者写完到消费者读到");
    hdr();
    rowf("shared: STS -> LDS",   slope([&](int n){ rt_smem<<<1,1,DS_BYTES>>>(0,0,n,g_out,7); },128,512,UNROLL), 1.0, "纯读 23.0");
    rowf("DSMEM 对端: ST -> LD", slope([&](int n){ dsmem_run<2>(2,1,n); },128,512,UNROLL), 1.0, "同地址; 纯读(追逐) 180.8");
    rowf("L1:  st.wb -> ld.ca",  slope([&](int n){ rt_l1<<<1,1>>>(c1.buf,c1.buf,n,g_out,7); },128,512,UNROLL), 1.0, "纯读 32.0");
    rowf("L2:  st.cg -> ld.cg",  slope([&](int n){ rt_l2<<<1,1>>>(c2.buf,c2.buf,n,g_out,7); },128,512,UNROLL), 1.0, "纯读 271.9");
    rowf("host pinned: relaxed.sys", slope([&](int n){ rt_sys<<<1,1>>>(hchain,hchain,n,g_out,7); },32,128,UNROLL), 1.0, "经 PCIe 往返");
    // HBM: 固定地址的往返一定落在 L2, 必须每次换全新随机行; 且要把两条链强制串起来
    const uint64_t BOFF = 1ull<<30;
    double hc = slope([&](int n){ rt_hbm<1><<<1,1>>>(c4.buf,(int)c4.slots,n,g_out,BOFF,BOFF,0,7); },64,256,UNROLL);
    double hb = slope([&](int n){ rt_hbm<0><<<1,1>>>(c4.buf,(int)c4.slots,n,g_out,BOFF,BOFF,0,7); },64,256,UNROLL);
    rowf("HBM: A追逐+B往返 串成一条链",  hc, 1.0, "两条链默认并行, 必须强制串行");
    rowf("HBM: 基线(同 ALU, 去掉 B 读写)", hb, 1.0);
    rowf("HBM: 相减 => 真 st.cg->ld.cg 往返", hc - hb, 1.0, "写不在 L2 的行要先 write-allocate");

    sec("三、可见性代价 —— 加 fence / release-acquire");
    hdr();
    rowf("st.release.gpu -> ld.acquire.gpu",
         slope([&](int n){ rt_rel<<<1,1>>>(c2.buf,c2.buf,n,g_out,7); },128,512,UNROLL), 1.0, "spin-wait 传标志位");
    rowf("st.cg + membar.gl -> ld.cg",
         slope([&](int n){ rt_fen<<<1,1>>>(c2.buf,c2.buf,n,g_out,7); },128,512,UNROLL), 1.0,
         "对比普通 L2 往返: 涨 3 倍多");

    int rc = sentinel_report();
    CK(cudaFree(c1.buf)); CK(cudaFree(c2.buf)); CK(cudaFree(c4.buf)); CK(cudaFreeHost(hp));
    CK(cudaFree(a_l1)); CK(cudaFree(a_l2)); CK(cudaFree(a_hbm)); CK(cudaFree(a_hp));
    CK(cudaFree(d_lnxt)); CK(cudaFree(g_dsnxt)); CK(cudaFree(g_out));
    return rc;
}

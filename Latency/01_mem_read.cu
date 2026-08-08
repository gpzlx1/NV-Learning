// ============================================================================
//  01_mem_read.cu —— 【存储层次 · 读延迟】
//
//  第一类测量: 每个存储空间的读延迟。全部用指针追逐 —— 节点里存的就是下一个节点
//  的地址, 依赖链上只有一条 load, 没有任何地址计算, 所以测到的是纯访存延迟。
//
//    寄存器 / 常量 / shared / DSMEM(自读,对端) / local(栈溢出) / L1 / L2 / HBM
//    / host pinned(经 PCIe)
//  再加两组正交变量:
//    · cache operator: .ca .cg .cs .lu .cv .nc / relaxed.gpu / relaxed.sys
//    · 访问宽度: 64-bit vs 128-bit
//
//  跨设备(NVLink)见 04_mem_p2p.cu; 容量台阶与 L2 分区见 03_mem_levels.cu
//  编译: make      运行: ./01_mem_read --dev 3
// ============================================================================
#include "common.cuh"
#include "dsmem.cuh"
// ══════════════ 读: 指针追逐 (节点里存的就是下一个节点的地址) ══════════════
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

MK_CHASE(ch_ca,  "ld.global.ca.u64 %0, [%0];")           // 经 L1
MK_CHASE(ch_cg,  "ld.global.cg.u64 %0, [%0];")           // 绕过 L1, 只在 L2
MK_CHASE(ch_cs,  "ld.global.cs.u64 %0, [%0];")           // streaming, 优先淘汰
MK_CHASE(ch_lu,  "ld.global.lu.u64 %0, [%0];")           // last-use
MK_CHASE(ch_cv,  "ld.global.cv.u64 %0, [%0];")           // don't cache, 强制重取
MK_CHASE(ch_nc,  "ld.global.nc.u64 %0, [%0];")           // non-coherent (__ldg)
MK_CHASE(ch_sys, "ld.relaxed.sys.global.u64 %0, [%0];")  // 系统作用域 (host pinned)
MK_CHASE(ch_gpu, "ld.relaxed.gpu.global.u64 %0, [%0];")  // GPU 作用域

// 128-bit 追逐: 取回的第一个 64bit 就是下一个地址
__global__ void ch_v2(void** start, int warm, int iters, uint64_t* out) {
    void** p = start; uint64_t j;
    for (int i = 0; i < warm; ++i)
        asm volatile("ld.global.cg.v2.u64 {%0,%1}, [%0];" : "+l"(p), "=l"(j) :: "memory");
    uint64_t t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
    for (int i = 0; i < iters; ++i) { _Pragma("unroll")
        for (int u = 0; u < UNROLL; ++u)
            asm volatile("ld.global.cg.v2.u64 {%0,%1}, [%0];" : "+l"(p), "=l"(j) :: "memory"); }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
    out[0] = t1 - t0; out[1] = (uint64_t)p + j;
}

// ── constant memory: 链上存字节偏移, SASS 应为 LDC c[..][Rx] 单指令 ──
__constant__ uint32_t C_ARR[2048];                        // 8KB
__global__ void ch_const(int warm, int iters, uint64_t* out) {
    uint32_t o = 0;
    for (int i = 0; i < warm; ++i) o = *(const uint32_t*)((const char*)C_ARR + o);
    uint64_t t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
    for (int i = 0; i < iters; ++i) {
        #pragma unroll
        for (int u = 0; u < UNROLL; ++u) o = *(const uint32_t*)((const char*)C_ARR + o);
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
    out[0] = t1 - t0; out[1] = o;
}

// ── local memory (栈溢出, SASS 应为 LDL/STL): 2KB 数组 + 数据相关下标 ──
__global__ void ch_local(const uint32_t* init, int warm, int iters, uint64_t* out) {
    uint32_t buf[512];                                    // 2KB, 放不进寄存器
    #pragma unroll 1
    for (int i = 0; i < 512; ++i) buf[i] = init[i];       // 内容是字节偏移
    uint32_t o = 0;
    for (int i = 0; i < warm; ++i) o = *(uint32_t*)((char*)buf + o);
    uint64_t t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
    for (int i = 0; i < iters; ++i) {
        #pragma unroll
        for (int u = 0; u < UNROLL; ++u) o = *(uint32_t*)((char*)buf + o);
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
    out[0] = t1 - t0; out[1] = o;
}

// ── shared memory 追逐 (链上存 32bit shared-window 地址) ──
__global__ void ch_smem(int slots, int stride, int warm, int iters,
                        uint64_t* out, const uint32_t* nxt) {
    extern __shared__ uint32_t sm[];
    uint32_t base = (uint32_t)__cvta_generic_to_shared(sm);
    for (int i = 0; i < slots; ++i)
        *(uint32_t*)((char*)sm + (size_t)i*stride) = base + nxt[i]*(uint32_t)stride;
    __syncwarp();
    uint32_t a = base;
    for (int i = 0; i < warm; ++i) asm volatile("ld.shared.u32 %0, [%0];" : "+r"(a) :: "memory");
    uint64_t t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
    for (int i = 0; i < iters; ++i) { _Pragma("unroll")
        for (int u = 0; u < UNROLL; ++u)
            asm volatile("ld.shared.u32 %0, [%0];" : "+r"(a) :: "memory"); }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
    out[0] = t1 - t0; out[1] = a;
}

int main(int argc, char** argv)
{
    dev_header(arg_dev(argc, argv));
    dsmem_init();
    { auto n = sattolo(2048); std::vector<uint32_t> b(2048);          // 常量: 8KB
      for (int i = 0; i < 2048; ++i) b[i] = n[i]*4u;
      CK(cudaMemcpyToSymbol(C_ARR, b.data(), 2048*4)); }
    uint32_t* d_lnxt; { auto n = sattolo(512); std::vector<uint32_t> b(512);  // local: 2KB
      for (int i = 0; i < 512; ++i) b[i] = n[i]*4u;
      CK(cudaMalloc(&d_lnxt, 512*4));
      CK(cudaMemcpy(d_lnxt, b.data(), 512*4, cudaMemcpyHostToDevice)); }

    Chain c1 = build_chain(24ull<<10, 128);                 // 落在 L1
    Chain c2 = build_chain(32ull<<20, 256);                 // 落在 L2 (< 50MB)
    Chain c3 = build_chain(2ull<<30,  512);                 // 远超 L2 -> HBM
    void* hp; CK(cudaHostAlloc(&hp, 32ull<<20, cudaHostAllocMapped));
    { size_t slots = (32ull<<20)/4096; auto n = sattolo(slots);
      char* h = (char*)hp; void* dp; CK(cudaHostGetDevicePointer(&dp, hp, 0));
      for (size_t i = 0; i < slots; ++i) *(void**)(h + i*4096) = (char*)dp + n[i]*4096; }
    void** hchain; CK(cudaHostGetDevicePointer((void**)&hchain, hp, 0));

    sec("一、各存储空间的读延迟 (指针追逐, 依赖链上只有一条 load)");
    hdr();
    rowf("常量内存 (8KB, LDC + 寄存器偏移)",
         slope([&](int n){ ch_const<<<1,1>>>(2048,n,g_out); },128,512,UNROLL), 1.0,
         "下标数据相关 -> 走非 uniform 路径");
    rowf("shared memory (LDS)",
         slope([&](int n){ ch_smem<<<1,1,DS_BYTES>>>(DS_SLOTS,DS_STRIDE,DS_SLOTS,n,g_out,g_dsnxt); },64,256,UNROLL), 1.0);
    rowf("local memory (LDL, 2KB 栈溢出)",
         slope([&](int n){ ch_local<<<1,1>>>(d_lnxt,512,n,g_out); },64,256,UNROLL), 1.0, "背后是 L1/L2");
    rowf("DSMEM 自读 (走 DSM 接口)",  slope([&](int n){ dsmem_run<0>(2,0,n); },64,256,UNROLL), 1.0);
    rowf("DSMEM 对端 (cluster=2, 跨 SM)", slope([&](int n){ dsmem_run<0>(2,1,n); },64,256,UNROLL), 1.0);
    rowf("L1 命中 (24KB, .ca)",
         slope([&](int n){ ch_ca<<<1,1>>>(c1.buf,(int)c1.slots,n,g_out); },64,256,UNROLL), 1.0);
    rowf("L2 命中 (32MB, .cg)",
         slope([&](int n){ ch_cg<<<1,1>>>(c2.buf,(int)c2.slots,n,g_out); },64,256,UNROLL), 1.0);
    rowf("HBM (2GB, .cg)",
         slope([&](int n){ ch_cg<<<1,1>>>(c3.buf,(int)c3.slots,n,g_out); },64,256,UNROLL), 1.0);
    rowf("host pinned (32MB, zero-copy 经 PCIe)",
         slope([&](int n){ ch_sys<<<1,1>>>(hchain,0,n,g_out); },8,32,UNROLL), 1.0);

    sec("二、cache operator 对读延迟的影响 (同一 32MB footprint)");
    hdr();
    struct { const char* n; void(*k)(void**,int,int,uint64_t*); const char* c; } CO[] = {
        {".ca  经 L1", ch_ca, ""}, {".cg  绕过 L1", ch_cg, ""}, {".cs  streaming", ch_cs, ""},
        {".lu  last-use", ch_lu, "提示用完即弃 -> 行被提前淘汰"},
        {".cv  不缓存强制重取", ch_cv, "在 sm_90 上并不绕过 L2"},
        {".nc  非一致 (__ldg)", ch_nc, ""},
        {"relaxed.gpu 作用域", ch_gpu, ""}, {"relaxed.sys 作用域", ch_sys, ""} };
    for (auto& o : CO)
        rowf(o.n, slope([&](int n){ o.k<<<1,1>>>(c2.buf,(int)c2.slots,n,g_out); },64,256,UNROLL), 1.0, o.c);

    sec("三、访问宽度对读延迟的影响 (32MB, .cg)");
    hdr();
    rowf("64-bit  (LDG.E.64)",
         slope([&](int n){ ch_cg<<<1,1>>>(c2.buf,(int)c2.slots,n,g_out); },64,256,UNROLL), 1.0);
    rowf("128-bit (LDG.E.128)",
         slope([&](int n){ ch_v2<<<1,1>>>(c2.buf,(int)c2.slots,n,g_out); },64,256,UNROLL), 1.0,
         "加宽不增加延迟 -> 向量化读是免费的");

    int rc = sentinel_report();
    CK(cudaFree(c1.buf)); CK(cudaFree(c2.buf)); CK(cudaFree(c3.buf));
    CK(cudaFreeHost(hp)); CK(cudaFree(d_lnxt)); CK(cudaFree(g_dsnxt)); CK(cudaFree(g_out));
    return rc;
}

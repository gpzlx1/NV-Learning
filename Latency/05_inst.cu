// ============================================================================
//  05_inst.cu —— 【指令延迟 · 标量指令】
//
//  测两件不同的事, 别混淆:
//   (1) 依赖链延迟 (ILP=1): 后一条指令依赖前一条的结果 -> 流水线深度。
//       这是"一条指令要多久才能把结果交给下一条"。
//   (2) 发射间隔 / 吞吐 (ILP=8): 8 条互不依赖的指令轮流发 -> 单位吞吐。
//       这是"每条指令占多少 cycle 的发射槽"。
//   latency / issue = 想把这条流水线填满需要多少路 ILP。
//
//  1 warp (32 线程) 单 block: 测的是 warp 指令的延迟/吞吐, 与真实 kernel 同单位。
//  编译: make lat_inst      运行: ./lat_inst --dev 3
// ============================================================================
#include "common.cuh"

static const int NOPS = 19;
static const char* OPNAME[NOPS] = {
    "SHF    funnel shift (ALU pipe)",
    "LOP3   bitwise 3-in (ALU pipe)",
    "IMAD   mad.lo.s32 (IMAD pipe)",
    "POPC   popc.b32",
    "FADD   add.f32",
    "FMUL   mul.f32",
    "FFMA   fma.rn.f32 (FMA pipe)",
    "DADD   add.f64 (FP64 pipe)",
    "DFMA   fma.rn.f64 (FP64 pipe)",
    "HADD2  add.f16x2",
    "MUFU   rcp.approx.f32 (SFU)",
    "MUFU   sqrt.approx.f32 (SFU)",
    "MUFU   sin.approx.f32 (SFU)",
    "SHFL   shfl.sync.idx.b32 (warp 洗牌)",
    "IMAD64 mad.lo.s64 (64位乘加)",
    "PRMT   prmt.b32 (字节重排)",
    "DPX    __viaddmax_s32 (VIADDMNMX)",
    "DPX    __vimax3_s32   (VIMNMX3)",
    "REDUX  redux.sync.add.s32 (warp 归约)",
};
static const int OPDIV[NOPS] = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};

template<int OP, int ILP>
__global__ void k_inst(uint64_t* out, int iters, uint32_t ai)
{
    uint32_t x[ILP]; float f[ILP]; double d[ILP]; uint64_t w[ILP];
    #pragma unroll
    // !! 必须掺进 threadIdx.x: 否则整个 warp 的值是 uniform 的, ptxas 能证明
    //    "对 uniform 值做 shuffle 是恒等操作", 会把整条 SHFL 链删干净。
    for (int j = 0; j < ILP; ++j) { x[j] = ai + j*7u + 1u + threadIdx.x;
                                    f[j] = 1.5f + j + threadIdx.x; d[j] = 1.5 + j;
                                    w[j] = ai + j + threadIdx.x; }
    uint32_t b = ai, b2 = ai * 3u + 1u;
    // add.f16x2 用 1.0h/0.5h 的位模式, 避免把小整数当成 fp16 denormal
    if constexpr (OP == 9) { b = 0x38003800u;
        #pragma unroll
        for (int j = 0; j < ILP; ++j) x[j] = 0x3C003C00u + threadIdx.x; }
    float  bf = 1.0000001f + (float)ai;
    double bd = 1.0000001  + (double)ai;
    uint64_t t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
    for (int i = 0; i < iters; ++i) {
        #pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            #pragma unroll
            for (int j = 0; j < ILP; ++j) {
                // 全部用运行时操作数, 且都是不可重结合的形式, 防 ptxas 折叠。
                // FP 的加/乘默认不允许重结合, 所以 add.f32/mul.f32 也是安全的。
                if constexpr (OP== 0) asm volatile("shf.l.wrap.b32 %0, %0, %0, %1;"       : "+r"(x[j]) : "r"(b));
                if constexpr (OP== 1) asm volatile("lop3.b32 %0, %0, %1, %2, 0xE8;"       : "+r"(x[j]) : "r"(b), "r"(b2));
                if constexpr (OP== 2) asm volatile("mad.lo.s32 %0, %0, %1, %1;"           : "+r"(x[j]) : "r"(b));
                if constexpr (OP== 3) asm volatile("popc.b32 %0, %0;"                     : "+r"(x[j]));
                if constexpr (OP== 4) asm volatile("add.f32 %0, %0, %1;"                  : "+f"(f[j]) : "f"(bf));
                if constexpr (OP== 5) asm volatile("mul.f32 %0, %0, %1;"                  : "+f"(f[j]) : "f"(bf));
                if constexpr (OP== 6) asm volatile("fma.rn.f32 %0, %0, %1, %1;"           : "+f"(f[j]) : "f"(bf));
                if constexpr (OP== 7) asm volatile("add.f64 %0, %0, %1;"                  : "+d"(d[j]) : "d"(bd));
                if constexpr (OP== 8) asm volatile("fma.rn.f64 %0, %0, %1, %1;"           : "+d"(d[j]) : "d"(bd));
                if constexpr (OP== 9) asm volatile("add.f16x2 %0, %0, %1;"                : "+r"(x[j]) : "r"(b));
                if constexpr (OP==10) asm volatile("rcp.approx.f32 %0, %0;"               : "+f"(f[j]));
                if constexpr (OP==11) asm volatile("sqrt.approx.f32 %0, %0;"              : "+f"(f[j]));
                if constexpr (OP==12) asm volatile("sin.approx.f32 %0, %0;"               : "+f"(f[j]));
                // bfly 是对合(两次=恒等), 固定源 lane 的 idx 是幂等(第二次结果不变),
                // 两者都会被折叠掉 -> 源 lane 必须取自数据本身, 才是真依赖链
                if constexpr (OP==13) asm volatile("shfl.sync.idx.b32 %0, %0, %0, 0x1f, 0xffffffff;" : "+r"(x[j]));
                if constexpr (OP==14) asm volatile("mad.lo.s64 %0, %0, %1, %2;" : "+l"(w[j]) : "l"((uint64_t)b), "l"((uint64_t)b2));  // mad.wide 的 a*b 是循环不变量会被提出去, 必须用 mad.lo.s64
                if constexpr (OP==15) asm volatile("prmt.b32 %0, %0, %1, %2;" : "+r"(x[j]) : "r"(b), "r"(b2));
                // DPX 在 PTX 里没有助记符, 只有 C 内建函数, nvcc 直接下译到 SASS
                if constexpr (OP==16) x[j] = __viaddmax_s32((int)x[j], (int)b, (int)b2);
                if constexpr (OP==17) x[j] = __vimax3_s32((int)x[j], (int)b, (int)b2);
                if constexpr (OP==18) asm volatile("redux.sync.add.s32 %0, %0, 0xffffffff;" : "+r"(x[j]));
            }
        }
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
    out[0] = t1 - t0;
    uint64_t s = 0;
    #pragma unroll
    for (int j = 0; j < ILP; ++j) s += x[j] + (uint64_t)f[j] + (uint64_t)d[j] + w[j];
    out[1] = s;                                            // 防死代码消除
}

// 无操作数的杂项: clock64 读取 / 线程块同步 / 内存栅栏
template<int OP>
__global__ void k_misc(uint64_t* out, int iters)
{
    uint64_t t0, t1, z = 0;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
    for (int i = 0; i < iters; ++i) {
        #pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            if constexpr (OP == 0) { uint64_t c; asm volatile("mov.u64 %0, %%clock64;" : "=l"(c) :: "memory"); z += c; }
            if constexpr (OP == 1) { uint64_t c; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(c) :: "memory"); z += c; }
            if constexpr (OP == 2) asm volatile("barrier.sync 0;" ::: "memory");
            if constexpr (OP == 3) asm volatile("membar.cta;" ::: "memory");
            if constexpr (OP == 4) asm volatile("membar.gl;"  ::: "memory");
            // elect 的输出(leader lane)与数据无关, 挂不上依赖链 -> 只能量发射开销
            if constexpr (OP == 5) { uint32_t t;
                asm volatile("{ .reg .pred p;\n\t elect.sync %0|p, 0xffffffff;\n\t}"
                             : "=r"(t) :: "memory"); z += t; }
        }
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
    out[0] = t1 - t0; out[1] = z;
}

template<int ILP, int OP = 0> static void run_ops(double* r)
{
    if constexpr (OP < NOPS) {
        r[OP] = slope([&](int n){ k_inst<OP,ILP><<<1,32>>>(g_out, n, 7); }, 128, 512,
                      UNROLL * ILP * OPDIV[OP]);
        run_ops<ILP, OP+1>(r);
    }
}

int main(int argc, char** argv)
{
    int dev = 3;
    for (int i = 1; i < argc; ++i)
        if (!strcmp(argv[i], "--dev") && i+1 < argc) dev = atoi(argv[++i]);
    dev_header(dev);

    double lat[NOPS], thr[NOPS];
    run_ops<1>(lat);                                       // ILP=1 -> 依赖链延迟
    run_ops<8>(thr);                                       // ILP=8 -> 发射间隔

    printf("\n═══ 标量指令: 依赖链延迟 vs 发射间隔 (1 warp = 32 线程, 单 block) ═══\n");
    printf("  %-32s %9s %9s %9s %8s\n", "指令", "延迟(周期)", "发射(周期)",
           "需要ILP", "延迟(纳秒)");
    printf("  ------------------------------------------------------------------------------\n");
    int bad = 0;
    for (int i = 0; i < NOPS; ++i) {
        // 哨兵: 依赖链延迟 < 1 cycle 物理上不可能 -> 一定被 ptxas 消除/折叠了
        const char* warn = (lat[i] < 1.0 || thr[i] < 0.02) ? "  <== 低于物理下限, 疑似被优化掉!" : "";
        if (*warn) ++bad;
        printf("  %-32s %9.2f %9.2f %9.1f %8.2f%s\n", OPNAME[i], lat[i], thr[i],
               thr[i] > 0.01 ? lat[i]/thr[i] : 0.0, lat[i]/g_ghz, warn);
    }
    if (bad) printf("\n  !! %d 行没通过哨兵检查: 依赖链延迟不可能 <1 周期, 用 `make sass` 复核\n", bad);

    printf("\n═══ 特殊寄存器与栅栏 (每条的周期数; 多方同步原语见 07_sync) ═══\n");
    printf("  ------------------------------------------------------------------------------\n");
    struct { const char* n; int op; } M[] = {
        {"mov %clock64        (读 SM 周期计数器)", 0}, {"mov %globaltimer   (读全局纳秒计时器)", 1},
        {"barrier.sync       (__syncthreads, 1 warp)", 2},
        {"membar.cta         (__threadfence_block)", 3},
        {"membar.gl          (__threadfence)", 4},
        {"elect.sync         (选 warp leader, 仅发射开销)", 5} };
    for (auto& m : M) {
        double v = 0;
        if (m.op==0) v = slope([&](int n){ k_misc<0><<<1,32>>>(g_out,n); }, 128,512, UNROLL);
        if (m.op==1) v = slope([&](int n){ k_misc<1><<<1,32>>>(g_out,n); }, 128,512, UNROLL);
        if (m.op==2) v = slope([&](int n){ k_misc<2><<<1,32>>>(g_out,n); }, 128,512, UNROLL);
        if (m.op==3) v = slope([&](int n){ k_misc<3><<<1,32>>>(g_out,n); }, 128,512, UNROLL);
        if (m.op==4) v = slope([&](int n){ k_misc<4><<<1,32>>>(g_out,n); }, 128,512, UNROLL);
        if (m.op==5) v = slope([&](int n){ k_misc<5><<<1,32>>>(g_out,n); }, 128,512, UNROLL);
        printf("  %-44s %8.2f 周期  %8.2f 纳秒\n", m.n, v, v/g_ghz);
    }
    CK(cudaFree(g_out));
    return 0;
}

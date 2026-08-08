# Hopper 延迟微基准套件
#  注意: CUDA 13.1 上 `-arch=sm_90a` 走整程序编译时会退化成 sm_90, wgmma 编不过。
#  必须用显式 gencode 才能真正拿到 sm_90a。
NVCC   ?= nvcc
ARCH   ?= compute_90a
CODE   ?= sm_90a
FLAGS  := -O3 -gencode arch=$(ARCH),code=$(CODE) -std=c++17 -lineinfo

# 按层次分类, 编号即执行顺序
STORAGE := 01_mem_read 02_mem_write 03_mem_levels 04_mem_p2p
INST    := 05_inst 06_tensor
OTHER   := 07_sync 08_async_copy
BINS    := $(STORAGE) $(INST) $(OTHER)

all: $(BINS)

%: %.cu common.cuh dsmem.cuh
	$(NVCC) $(FLAGS) $< -o $@

# ── SASS 复核: 不是可选步骤 ──────────────────────────────────────────────
# ptxas 会强度削减依赖链、折叠对合/幂等操作、删掉 warp-uniform 的 shuffle、
# 做编译期 store-to-load forwarding、提走循环不变量、消除死存储 —— 全都不报错,
# 只会安静地给出一个"看起来很合理"的错数字。
sass: $(BINS)
	@for b in $(BINS); do cuobjdump -sass $$b > sass-$$b.txt; done
	@echo "── 读: 指针追逐 (地址必须来自上一次 load 的目标寄存器) ──"
	@grep -hoE 'LDG[^;]*;' sass-01_mem_read.txt | sort | uniq -c | sort -rn | head -3
	@grep -hoE 'LDS [^;]*;|LDL[^;]*;|LDC[^;]*;' sass-01_mem_read.txt | sort | uniq -c | sort -rn | head -3
	@echo "── 写: store 和 load 必须都在(否则被编译期转发了) ──"
	@printf '  STS x%s  LDS x%s  STG x%s  LDG x%s  STL x%s\n' \
	  $$(grep -cE '\bSTS\b' sass-02_mem_write.txt) $$(grep -cE '\bLDS\b' sass-02_mem_write.txt) \
	  $$(grep -cE '\bSTG\b' sass-02_mem_write.txt) $$(grep -cE '\bLDG\b' sass-02_mem_write.txt) \
	  $$(grep -cE '\bSTL\b' sass-02_mem_write.txt)
	@echo "── 指令: 依赖链必须是同一寄存器自依赖 ──"
	@grep -hoE '(SHF|IMAD|FFMA|DFMA|HADD2|SHFL|MUFU|VIADDMNMX|VIMNMX3|REDUX)[^;]*;' sass-05_inst.txt \
	  | sort | uniq -c | sort -rn | head -8
	@echo "── 张量核心 / 异步 / 同步 ──"
	@grep -hoE '(HGMMA[0-9x.A-Z]*|HMMA[0-9.A-Z]*|WARPGROUP[.A-Z]*)' sass-06_tensor.txt | sort | uniq -c | head -4
	@grep -hoE '(LDGSTS[A-Z0-9.]*|UBLKCP[A-Z.]*|UTMACMDFLUSH)' sass-08_async_copy.txt | sort | uniq -c | head -4
	@grep -hoE '(SYNCS[A-Z0-9._]*|UCGABAR_[A-Z]+|ATOMS[.A-Z]*|ATOMG[.A-Z]*|RED[.A-Z]*|MEMBAR[.A-Z]*)' sass-07_sync.txt | sort | uniq -c | sort -rn | head -6
	@echo "(完整 SASS 见 sass-*.txt)"

clean:
	rm -f $(BINS) sass-*.txt

.PHONY: all sass clean

NVCC ?= nvcc
ARCH ?= compute_110
CODE ?= sm_110
FLAGS := -O3 -gencode arch=$(ARCH),code=$(CODE) -std=c++17 -lineinfo
BINS := 01_global_memory 02_cache_working_set 03_shared_memory 03b_shared_patterns \
        04_dsmem 05_atomics 06_copy_engines 07_tcgen05_throughput 08_async_copy 09_tma

all: $(BINS)

01_global_memory: 01_global_memory.cu
	$(NVCC) $(FLAGS) $< -o $@

02_cache_working_set: 02_cache_working_set.cu
	$(NVCC) $(FLAGS) $< -o $@

03_shared_memory: 03_shared_memory.cu
	$(NVCC) $(FLAGS) $< -o $@

03b_shared_patterns: 03b_shared_patterns.cu
	$(NVCC) $(FLAGS) $< -o $@

04_dsmem: 04_dsmem.cu
	$(NVCC) $(FLAGS) $< -o $@

05_atomics: 05_atomics.cu
	$(NVCC) $(FLAGS) $< -o $@

06_copy_engines: 06_copy_engines.cu
	$(NVCC) $(FLAGS) $< -o $@

CUTLASS_INC ?= /opt/pytorch/ao/third_party/cutlass/include
07_tcgen05_throughput: 07_tcgen05_throughput.cu ../Latency/06b_tcgen05.cu
	$(NVCC) -O3 -gencode arch=compute_110f,code=sm_110f -std=c++17 -lineinfo -I$(CUTLASS_INC) $< -o $@

08_async_copy: 08_async_copy.cu
	$(NVCC) $(FLAGS) $< -o $@

09_tma: 09_tma.cu
	$(NVCC) $(FLAGS) $< -o $@

sass: $(BINS)
	@mkdir -p results
	@for b in $(BINS); do cuobjdump -sass $$b > results/$$b.sass.txt; done

clean:
	rm -f $(BINS)

.PHONY: all sass clean

NVCC ?= nvcc
FLAGS := -O3 -gencode arch=compute_110,code=sm_110 -std=c++17 -lineinfo
all: 01_stream_priority
01_stream_priority: 01_stream_priority.cu
	$(NVCC) $(FLAGS) $< -o $@
clean:
	rm -f 01_stream_priority
.PHONY: all clean

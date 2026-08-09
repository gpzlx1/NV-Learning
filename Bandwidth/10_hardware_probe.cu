// One launch per process: deterministic probes for Nsight Compute counters.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){fprintf(stderr,"CUDA: %s\n",cudaGetErrorString(e)); exit(1);} } while(0)
constexpr int THREADS=256, BLOCKS=160, REPS=8192;
__device__ __forceinline__ unsigned lds(unsigned a){unsigned v;asm volatile("ld.shared.u32 %0,[%1];":"=r"(v):"r"(a):"memory");return v;}
__device__ __forceinline__ void sts(unsigned a,unsigned v){asm volatile("st.shared.u32 [%0],%1;"::"r"(a),"r"(v):"memory");}

__global__ void probe_global(const unsigned *src, size_t words, unsigned stride,
                             unsigned long long *sink) {
  unsigned lane=threadIdx.x&31, warp=(blockIdx.x*blockDim.x+threadIdx.x)>>5;
  unsigned warps=(gridDim.x*blockDim.x)>>5, v=0;
  // Each warp starts on a 128-byte boundary. stride changes useful bytes per
  // 32-byte sector without changing the instruction count.
  for(size_t base=size_t(warp)*32*stride; base+31*stride<words;
      base+=size_t(warps)*32*stride)
    v ^= src[base+lane*stride];
  if((threadIdx.x&31)==0) atomicAdd(sink,(unsigned long long)v);
}

template<bool WRITE> __global__ void probe_shared(unsigned stride,
                                                   unsigned long long *sink) {
  __shared__ unsigned sm[8192];
  for(int i=threadIdx.x;i<8192;i+=blockDim.x) sm[i]=i+1;
  __syncthreads();
  unsigned base=(unsigned)__cvta_generic_to_shared(sm);
  unsigned lane=(threadIdx.x*stride)&1023;
  unsigned p[8];
#pragma unroll
  for(int j=0;j<8;++j) p[j]=base+(lane+j*1024)*4;
  unsigned a[8]={1,2,3,4,5,6,7,8};
#pragma unroll 1
  for(int i=0;i<REPS;++i) {
    if constexpr(WRITE) {
#pragma unroll
      for(int j=0;j<8;++j) sts(p[j],a[j]+i);
    } else {
#pragma unroll
      for(int j=0;j<8;++j) a[j]^=lds(p[j]);
    }
  }
  unsigned v=0;
#pragma unroll
  for(int j=0;j<8;++j) v+=a[j];
  if((threadIdx.x&31)==0) atomicAdd(sink,(unsigned long long)v);
}

int main(int argc,char **argv) {
  const char *mode=argc>1?argv[1]:"global";
  unsigned stride=argc>2?unsigned(atoi(argv[2])):1;
  int dev=argc>3?atoi(argv[3]):0;
  if(!stride || (strcmp(mode,"global") && strcmp(mode,"shared-read") && strcmp(mode,"shared-write"))){
    fprintf(stderr,"usage: %s global|shared-read|shared-write stride [device]\n",argv[0]); return 2;
  }
  CK(cudaSetDevice(dev)); unsigned long long *sink; CK(cudaMalloc(&sink,sizeof(*sink))); CK(cudaMemset(sink,0,sizeof(*sink)));
  if(!strcmp(mode,"global")) {
    constexpr size_t BYTES=512ull<<20; unsigned *src; CK(cudaMalloc(&src,BYTES)); CK(cudaMemset(src,1,BYTES));
    probe_global<<<BLOCKS,THREADS>>>(src,BYTES/4,stride,sink); CK(cudaGetLastError()); CK(cudaDeviceSynchronize()); cudaFree(src);
  } else if(!strcmp(mode,"shared-read")) {
    probe_shared<false><<<BLOCKS,THREADS>>>(stride,sink); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  } else {
    probe_shared<true><<<BLOCKS,THREADS>>>(stride,sink); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  }
  unsigned long long out=0; CK(cudaMemcpy(&out,sink,sizeof(out),cudaMemcpyDeviceToHost));
  printf("mode=%s stride=%u sink=%llu\n",mode,stride,out); cudaFree(sink); return 0;
}

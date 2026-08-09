// Atomic throughput versus contention (operations/s, not guessed bytes/s).
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA: %s line %d\n",cudaGetErrorString(e),__LINE__);exit(1);}}while(0)
constexpr int T=256,ITERS=8192,RUNS=7;
__global__ void k_global(unsigned* p,unsigned slots){unsigned id=blockIdx.x*blockDim.x+threadIdx.x;unsigned idx=id%slots;
  #pragma unroll 1
  for(int i=0;i<ITERS;++i)atomicAdd(p+idx,1u);}
__global__ void k_shared(unsigned slots,unsigned long long*out){extern __shared__ unsigned sm[];for(unsigned i=threadIdx.x;i<slots;i+=blockDim.x)sm[i]=0;__syncthreads();unsigned idx=threadIdx.x%slots;
  unsigned addr=(unsigned)__cvta_generic_to_shared(sm+idx),one=1;
  #pragma unroll 1
  for(int i=0;i<ITERS;++i)asm volatile("red.shared.add.u32 [%0],%1;"::"r"(addr),"r"(one):"memory");__syncthreads();if(threadIdx.x==0){unsigned s=0;for(unsigned i=0;i<slots;++i)s+=sm[i];atomicAdd(out,(unsigned long long)s);}}
template<class F>double best(F f){cudaEvent_t a,b;cudaEventCreate(&a);cudaEventCreate(&b);f();CK(cudaDeviceSynchronize());std::vector<float>v;for(int r=0;r<RUNS;++r){cudaEventRecord(a);f();cudaEventRecord(b);cudaEventSynchronize(b);float x;cudaEventElapsedTime(&x,a,b);v.push_back(x);}std::sort(v.begin(),v.end());cudaEventDestroy(a);cudaEventDestroy(b);return v[0];}
int main(int argc,char**argv){int dev=0;for(int i=1;i+1<argc;++i)if(!strcmp(argv[i],"--dev"))dev=atoi(argv[++i]);CK(cudaSetDevice(dev));cudaDeviceProp pr{};cudaGetDeviceProperties(&pr,dev);int blocks=pr.multiProcessorCount;unsigned*p;unsigned long long*out;cudaMalloc(&p,65536*4);cudaMalloc(&out,8);cudaMemset(out,0,8);double ops=double(blocks)*T*ITERS;
  printf("GPU %d: %s sm_%d%d, %d SM\n",dev,pr.name,pr.major,pr.minor,pr.multiProcessorCount);printf("\n=== Global atomic add throughput ===\nindependent_addresses,contention_threads_per_address,Gops,time_ms\n");for(unsigned slots:{1u,32u,1024u,65536u}){cudaMemset(p,0,slots*4);double ms=best([&]{k_global<<<blocks,T>>>(p,slots);});printf("%u,%.2f,%.4f,%.4f\n",slots,double(blocks*T)/slots,ops/(ms*1e6),ms);}
  printf("\n=== Shared atomic add throughput (addresses per block) ===\nindependent_addresses_per_block,threads_per_address,Gops,time_ms\n");for(unsigned slots:{1u,2u,4u,8u,16u,32u,256u}){double ms=best([&]{k_shared<<<blocks,T,slots*4>>>(slots,out);});printf("%u,%.2f,%.4f,%.4f\n",slots,double(T)/slots,ops/(ms*1e6),ms);}unsigned long long h;cudaMemcpy(&h,out,8,cudaMemcpyDeviceToHost);printf("observable_sink=%llu\n",h);cudaFree(out);cudaFree(p);}

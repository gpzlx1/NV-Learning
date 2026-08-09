// TMA bulk global->shared throughput versus transfer size and batch depth.
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA: %s line %d\n",cudaGetErrorString(e),__LINE__);exit(1);}}while(0)
constexpr int RUNS=7;
__device__ __forceinline__ unsigned smaddr(const void*p){return(unsigned)__cvta_generic_to_shared(p);}
__device__ __forceinline__ void waitbar(unsigned bar,unsigned phase){asm volatile("{.reg .pred p; W%=: mbarrier.try_wait.parity.shared::cta.b64 p,[%0],%1; @!p bra.uni W%=;}"::"r"(bar),"r"(phase):"memory");}
template<int BYTES,int STAGES>__global__ void k_tma(const char*src,size_t tiles,unsigned long long*out){__shared__ __align__(128) char sm[8192];__shared__ __align__(8) unsigned long long bar64;if(threadIdx.x==0)asm volatile("mbarrier.init.shared::cta.b64 [%0],1;"::"r"(smaddr(&bar64)):"memory");asm volatile("fence.proxy.async.shared::cta;":::"memory");__syncthreads();if(threadIdx.x==0){unsigned bar=smaddr(&bar64),phase=0;for(size_t first=blockIdx.x*STAGES;first+STAGES<=tiles;first+=gridDim.x*STAGES){asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;"::"r"(bar),"r"(STAGES*BYTES):"memory");
      #pragma unroll
      for(int u=0;u<STAGES;++u){const char*g=src+(first+u)*BYTES;asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0],[%1],%2,[%3];"::"r"(smaddr(sm+u*BYTES)),"l"(g),"r"(BYTES),"r"(bar):"memory");}waitbar(bar,phase);phase^=1;}atomicAdd(out,(unsigned long long)*(unsigned*)sm);}}
template<class F>double best(F f){cudaEvent_t a,b;cudaEventCreate(&a);cudaEventCreate(&b);f();CK(cudaDeviceSynchronize());std::vector<float>v;for(int r=0;r<RUNS;++r){cudaEventRecord(a);f();cudaEventRecord(b);cudaEventSynchronize(b);float x;cudaEventElapsedTime(&x,a,b);v.push_back(x);}std::sort(v.begin(),v.end());cudaEventDestroy(a);cudaEventDestroy(b);return v[0];}
template<int B,int S>void run(const char*src,size_t bytes,int blocks,unsigned long long*out){size_t tiles=bytes/B;double ms=best([&]{k_tma<B,S><<<blocks,32>>>(src,tiles,out);});size_t done=(tiles/(blocks*S))*(blocks*S)*B;printf("%d,%d,%.2f,%.4f\n",B,S,double(done)/(ms*1e6),ms);}
int main(int argc,char**argv){int dev=0;for(int i=1;i+1<argc;++i)if(!strcmp(argv[i],"--dev"))dev=atoi(argv[++i]);CK(cudaSetDevice(dev));cudaDeviceProp p{};cudaGetDeviceProperties(&p,dev);size_t bytes=512ull<<20;char*src;unsigned long long*out;cudaMalloc(&src,bytes);cudaMalloc(&out,8);cudaMemset(src,1,bytes);cudaMemset(out,0,8);int blocks=p.multiProcessorCount*4;printf("GPU %d: %s sm_%d%d, %d SM\n",dev,p.name,p.major,p.minor,p.multiProcessorCount);printf("transfer_bytes,batch_depth,requested_GBps,time_ms\n");run<128,1>(src,bytes,blocks,out);run<128,4>(src,bytes,blocks,out);run<128,8>(src,bytes,blocks,out);run<512,1>(src,bytes,blocks,out);run<512,4>(src,bytes,blocks,out);run<512,8>(src,bytes,blocks,out);run<1024,1>(src,bytes,blocks,out);run<1024,4>(src,bytes,blocks,out);run<1024,8>(src,bytes,blocks,out);unsigned long long h;cudaMemcpy(&h,out,8,cudaMemcpyDeviceToHost);printf("observable_sink=%llu\n",h);cudaFree(out);cudaFree(src);}

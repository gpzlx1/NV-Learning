// Shared-memory throughput and bank-conflict scaling.
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA: %s line %d\n",cudaGetErrorString(e),__LINE__);exit(1);}}while(0)
constexpr int T=256,RUNS=7,ITERS=65535;
__device__ __forceinline__ unsigned lds(unsigned a){unsigned v;asm volatile("ld.shared.u32 %0,[%1];":"=r"(v):"r"(a):"memory");return v;}
__device__ __forceinline__ void sts(unsigned a,unsigned v){asm volatile("st.shared.u32 [%0],%1;"::"r"(a),"r"(v):"memory");}

template<bool WRITE> __global__ void k_shared(unsigned stride,unsigned long long*out){__shared__ unsigned sm[8192];
  for(int i=threadIdx.x;i<8192;i+=blockDim.x)sm[i]=i+1;__syncthreads();unsigned base=(unsigned)__cvta_generic_to_shared(sm);unsigned lane=(threadIdx.x*stride)&1023;
  unsigned a0=1,a1=2,a2=3,a3=4,a4=5,a5=6,a6=7,a7=8;
  unsigned p0=base+(lane+0*1024)*4,p1=base+(lane+1*1024)*4,p2=base+(lane+2*1024)*4,p3=base+(lane+3*1024)*4;
  unsigned p4=base+(lane+4*1024)*4,p5=base+(lane+5*1024)*4,p6=base+(lane+6*1024)*4,p7=base+(lane+7*1024)*4;
  #pragma unroll 1
  for(int i=0;i<ITERS;++i){
    if constexpr(WRITE){sts(p0,a0+i);sts(p1,a1+i);sts(p2,a2+i);sts(p3,a3+i);sts(p4,a4+i);sts(p5,a5+i);sts(p6,a6+i);sts(p7,a7+i);}
    else{a0^=lds(p0);a1^=lds(p1);a2^=lds(p2);a3^=lds(p3);a4^=lds(p4);a5^=lds(p5);a6^=lds(p6);a7^=lds(p7);}
  }
  unsigned s=a0+a1+a2+a3+a4+a5+a6+a7;__shared__ unsigned red[T];red[threadIdx.x]=s;__syncthreads();
  for(int d=T/2;d;d>>=1){if(threadIdx.x<d)red[threadIdx.x]+=red[threadIdx.x+d];__syncthreads();}if(threadIdx.x==0)atomicAdd(out,(unsigned long long)red[0]);}

template<class F>double best(F f){cudaEvent_t a,b;CK(cudaEventCreate(&a));CK(cudaEventCreate(&b));f();CK(cudaDeviceSynchronize());std::vector<float>v;for(int r=0;r<RUNS;++r){cudaEventRecord(a);f();cudaEventRecord(b);cudaEventSynchronize(b);float x;cudaEventElapsedTime(&x,a,b);v.push_back(x);}std::sort(v.begin(),v.end());cudaEventDestroy(a);cudaEventDestroy(b);return v[0];}
int main(int argc,char**argv){int dev=0;for(int i=1;i+1<argc;++i)if(!strcmp(argv[i],"--dev"))dev=atoi(argv[++i]);CK(cudaSetDevice(dev));cudaDeviceProp p{};CK(cudaGetDeviceProperties(&p,dev));int blocks=p.multiProcessorCount;unsigned long long*out;CK(cudaMalloc(&out,8));CK(cudaMemset(out,0,8));double bytes=double(blocks)*T*ITERS*8*4;
  printf("GPU %d: %s sm_%d%d, %d SM\n",dev,p.name,p.major,p.minor,p.multiProcessorCount);printf("\n=== Shared-memory bank conflict ===\nstride_words,bank_conflict_degree,read_GBps,write_GBps,read_ms,write_ms\n");
  for(unsigned s:{1u,2u,4u,8u,16u,32u}){double r=best([&]{k_shared<false><<<blocks,T>>>(s,out);});double w=best([&]{k_shared<true><<<blocks,T>>>(s,out);});printf("%u,%u,%.2f,%.2f,%.4f,%.4f\n",s,s,bytes/(r*1e6),bytes/(w*1e6),r,w);}
  unsigned long long h;cudaMemcpy(&h,out,8,cudaMemcpyDeviceToHost);printf("observable_sink=%llu\n",h);cudaFree(out);}

// Independent validation of shared broadcast, distinct same-bank reads and width.
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA: %s line %d\n",cudaGetErrorString(e),__LINE__);exit(1);}}while(0)
constexpr int T=256,ITERS=65535,RUNS=7;
template<int W>__device__ __forceinline__ unsigned ld_s(unsigned a){unsigned x=0,y=0,z=0,q=0;if constexpr(W==1)asm volatile("ld.shared.u32 %0,[%1];":"=r"(x):"r"(a):"memory");if constexpr(W==2)asm volatile("ld.shared.v2.u32 {%0,%1},[%2];":"=r"(x),"=r"(y):"r"(a):"memory");if constexpr(W==4){asm volatile("ld.shared.v2.u32 {%0,%1},[%2];":"=r"(x),"=r"(y):"r"(a):"memory");asm volatile("ld.shared.v2.u32 {%0,%1},[%2];":"=r"(z),"=r"(q):"r"(a+8):"memory");}return x^y^z^q;}
template<int W,int PAT>__global__ void k(unsigned long long*out){__shared__ unsigned sm[8192+128];for(int i=threadIdx.x;i<8192+128;i+=blockDim.x)sm[i]=i+1;__syncthreads();unsigned base=(unsigned)__cvta_generic_to_shared(sm);unsigned lane=threadIdx.x&31;unsigned word=PAT==0?0:(PAT==1?lane*W:lane*32);unsigned a[8];
  #pragma unroll
  for(int u=0;u<8;++u)a[u]=base+(word+u*1024)*4;unsigned x0=1,x1=2,x2=3,x3=4,x4=5,x5=6,x6=7,x7=8;
  #pragma unroll 1
  for(int i=0;i<ITERS;++i){x0^=ld_s<W>(a[0]);x1^=ld_s<W>(a[1]);x2^=ld_s<W>(a[2]);x3^=ld_s<W>(a[3]);x4^=ld_s<W>(a[4]);x5^=ld_s<W>(a[5]);x6^=ld_s<W>(a[6]);x7^=ld_s<W>(a[7]);}unsigned s=x0+x1+x2+x3+x4+x5+x6+x7;__shared__ unsigned red[T];red[threadIdx.x]=s;__syncthreads();for(int d=T/2;d;d>>=1){if(threadIdx.x<d)red[threadIdx.x]+=red[threadIdx.x+d];__syncthreads();}if(threadIdx.x==0)atomicAdd(out,(unsigned long long)red[0]);}
template<class F>double best(F f){cudaEvent_t a,b;cudaEventCreate(&a);cudaEventCreate(&b);f();CK(cudaDeviceSynchronize());std::vector<float>v;for(int r=0;r<RUNS;++r){cudaEventRecord(a);f();cudaEventRecord(b);cudaEventSynchronize(b);float x;cudaEventElapsedTime(&x,a,b);v.push_back(x);}std::sort(v.begin(),v.end());cudaEventDestroy(a);cudaEventDestroy(b);return v[0];}
template<int W,int P>void run(int blocks,unsigned long long*out,const char*name){double ms=best([&]{k<W,P><<<blocks,T>>>(out);});double bytes=double(blocks)*T*ITERS*8*W*4;printf("%s,%d,%.2f,%.4f\n",name,W*32,bytes/(ms*1e6),ms);}
int main(int argc,char**argv){int dev=0;for(int i=1;i+1<argc;++i)if(!strcmp(argv[i],"--dev"))dev=atoi(argv[++i]);CK(cudaSetDevice(dev));cudaDeviceProp p{};cudaGetDeviceProperties(&p,dev);unsigned long long*out;cudaMalloc(&out,8);cudaMemset(out,0,8);int b=p.multiProcessorCount;printf("GPU %d: %s sm_%d%d, %d SM\n",dev,p.name,p.major,p.minor,p.multiProcessorCount);printf("pattern,access_bits,requested_GBps,time_ms\n");run<1,0>(b,out,"broadcast_same_address");run<1,1>(b,out,"conflict_free_distinct");run<1,2>(b,out,"same_bank_distinct");run<2,1>(b,out,"conflict_free_distinct");run<2,2>(b,out,"same_bank_distinct");run<4,1>(b,out,"conflict_free_distinct");run<4,2>(b,out,"same_bank_distinct");unsigned long long h;cudaMemcpy(&h,out,8,cudaMemcpyDeviceToHost);printf("observable_sink=%llu\n",h);cudaFree(out);}

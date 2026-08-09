// Cache-level bandwidth via controlled reuse and cache operators.
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){fprintf(stderr,"CUDA: %s (%d)\n",cudaGetErrorString(e),__LINE__);exit(1);}}while(0)
constexpr int T=256, RUNS=7;

template<int OP> __device__ __forceinline__ uint4 ld128(const uint4* p) {
  uint4 v;
  if constexpr (OP==0) asm volatile("ld.global.ca.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x),"=r"(v.y),"=r"(v.z),"=r"(v.w):"l"(p):"memory");
  if constexpr (OP==1) asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x),"=r"(v.y),"=r"(v.z),"=r"(v.w):"l"(p):"memory");
  if constexpr (OP==2) asm volatile("ld.global.cs.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x),"=r"(v.y),"=r"(v.z),"=r"(v.w):"l"(p):"memory");
  return v;
}

template<int OP, bool PRIVATE>
__global__ void k_reuse(const uint4* src, size_t elems, int passes, unsigned long long* sink) {
  size_t base = PRIVATE ? size_t(blockIdx.x)*elems : 0;
  unsigned a0=0,a1=1,a2=2,a3=3,a4=4,a5=5,a6=6,a7=7;
  for(int p=0;p<passes;++p)
    for(size_t i=threadIdx.x;i<elems;i+=blockDim.x*8ull){
      uint4 v0=ld128<OP>(src+base+i+blockDim.x*0ull),v1=ld128<OP>(src+base+i+blockDim.x*1ull);
      uint4 v2=ld128<OP>(src+base+i+blockDim.x*2ull),v3=ld128<OP>(src+base+i+blockDim.x*3ull);
      uint4 v4=ld128<OP>(src+base+i+blockDim.x*4ull),v5=ld128<OP>(src+base+i+blockDim.x*5ull);
      uint4 v6=ld128<OP>(src+base+i+blockDim.x*6ull),v7=ld128<OP>(src+base+i+blockDim.x*7ull);
      a0^=v0.x^v0.y^v0.z^v0.w;a1^=v1.x^v1.y^v1.z^v1.w;
      a2^=v2.x^v2.y^v2.z^v2.w;a3^=v3.x^v3.y^v3.z^v3.w;
      a4^=v4.x^v4.y^v4.z^v4.w;a5^=v5.x^v5.y^v5.z^v5.w;
      a6^=v6.x^v6.y^v6.z^v6.w;a7^=v7.x^v7.y^v7.z^v7.w;
    }
  unsigned s=a0+a1+a2+a3+a4+a5+a6+a7;
  __shared__ unsigned sm[T]; sm[threadIdx.x]=s; __syncthreads();
  for(int d=T/2;d;d>>=1){if(threadIdx.x<d)sm[threadIdx.x]+=sm[threadIdx.x+d];__syncthreads();}
  if(threadIdx.x==0) atomicAdd(sink,(unsigned long long)sm[0]);
}

template<class F> double best_ms(F f){cudaEvent_t a,b;CK(cudaEventCreate(&a));CK(cudaEventCreate(&b));f();CK(cudaDeviceSynchronize());std::vector<float>v;
  for(int i=0;i<RUNS;++i){CK(cudaEventRecord(a));f();CK(cudaEventRecord(b));CK(cudaEventSynchronize(b));float x;CK(cudaEventElapsedTime(&x,a,b));v.push_back(x);}std::sort(v.begin(),v.end());cudaEventDestroy(a);cudaEventDestroy(b);return v[0];}
static double rate(double bytes,double ms){return bytes/(ms*1e6);}

int main(int argc,char**argv){int dev=0;for(int i=1;i+1<argc;++i)if(!strcmp(argv[i],"--dev"))dev=atoi(argv[++i]);CK(cudaSetDevice(dev));cudaDeviceProp pr{};CK(cudaGetDeviceProperties(&pr,dev));
  printf("GPU %d: %s sm_%d%d, %d SM, L2 %.0f MB\n",dev,pr.name,pr.major,pr.minor,pr.multiProcessorCount,pr.l2CacheSize/1048576.0);
  const int blocks=pr.multiProcessorCount; const size_t alloc=512ull<<20; uint4*buf;unsigned long long*sink;CK(cudaMalloc(&buf,alloc));CK(cudaMalloc(&sink,8));CK(cudaMemset(buf,1,alloc));CK(cudaMemset(sink,0,8));
  const double target=16.0*(1ull<<30);
  struct C{const char*n;size_t footprint;bool priv;int op;};
  C cs[]={{"L1_private_64KB_per_SM",64ull<<10,true,0},{"L2_shared_16MB",16ull<<20,false,1},{"L2_shared_32MB",32ull<<20,false,1}};
  printf("\n=== Working-set reuse bandwidth ===\nname,footprint_bytes,passes,requested_GBps,time_ms\n");
  for(auto c:cs){size_t per=c.footprint/16;double bytes_per_pass=double(c.footprint)*(c.priv?blocks:blocks);int passes=std::max(1,int(target/bytes_per_pass));double ms=0;
    if(c.priv)ms=best_ms([&]{k_reuse<0,true><<<blocks,T>>>(buf,per,passes,sink);});
    else ms=best_ms([&]{k_reuse<1,false><<<blocks,T>>>(buf,per,passes,sink);});
    double requested=bytes_per_pass*passes;printf("%s,%zu,%d,%.2f,%.4f\n",c.n,c.footprint,passes,rate(requested,ms),ms);
  }
  printf("\n=== Cache operator on shared 16MB working set ===\noperator,requested_GBps,time_ms\n");size_t elems=(16ull<<20)/16;int passes=int(target/(double(16ull<<20)*blocks));
  double a=best_ms([&]{k_reuse<0,false><<<blocks,T>>>(buf,elems,passes,sink);});
  double g=best_ms([&]{k_reuse<1,false><<<blocks,T>>>(buf,elems,passes,sink);});
  double s=best_ms([&]{k_reuse<2,false><<<blocks,T>>>(buf,elems,passes,sink);});double bytes=double(16ull<<20)*blocks*passes;
  printf("ca,%.2f,%.4f\ncg,%.2f,%.4f\ncs,%.2f,%.4f\n",rate(bytes,a),a,rate(bytes,g),g,rate(bytes,s),s);
  unsigned long long h;CK(cudaMemcpy(&h,sink,8,cudaMemcpyDeviceToHost));printf("observable_sink=%llu\n",h);cudaFree(sink);cudaFree(buf);}

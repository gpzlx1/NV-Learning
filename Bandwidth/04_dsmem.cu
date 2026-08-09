// Distributed shared-memory (cluster network) bandwidth.
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
namespace cg=cooperative_groups;
#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA: %s line %d\n",cudaGetErrorString(e),__LINE__);exit(1);}}while(0)
constexpr int T=256,ITERS=16384,RUNS=7,SMWORDS=8192;
__device__ __forceinline__ unsigned ldsm(unsigned a){unsigned v;asm volatile("ld.shared::cluster.u32 %0,[%1];":"=r"(v):"r"(a):"memory");return v;}
__device__ __forceinline__ void stsm(unsigned a,unsigned v){asm volatile("st.shared::cluster.u32 [%0],%1;"::"r"(a),"r"(v):"memory");}

template<bool RING> __global__ void k_dsm(unsigned cs,unsigned long long*out){extern __shared__ unsigned occupancy_pad[];__shared__ unsigned sm[SMWORDS];for(int i=threadIdx.x;i<SMWORDS;i+=blockDim.x)sm[i]=i+blockIdx.x;cg::cluster_group cl=cg::this_cluster();unsigned rank=cl.block_rank();if(threadIdx.x==0)occupancy_pad[0]=rank;unsigned base=(unsigned)__cvta_generic_to_shared(sm),peer;unsigned target=(rank+1)%cs;asm volatile("mapa.shared::cluster.u32 %0,%1,%2;":"=r"(peer):"r"(base),"r"(target));cl.sync();
  unsigned a0=1,a1=2,a2=3,a3=4,a4=5,a5=6,a6=7,a7=8;unsigned lane=threadIdx.x;
  if(RING||rank==0){
    #pragma unroll 1
    for(int i=0;i<ITERS;++i){a0^=ldsm(peer+(lane+0*1024)*4);a1^=ldsm(peer+(lane+1*1024)*4);a2^=ldsm(peer+(lane+2*1024)*4);a3^=ldsm(peer+(lane+3*1024)*4);a4^=ldsm(peer+(lane+4*1024)*4);a5^=ldsm(peer+(lane+5*1024)*4);a6^=ldsm(peer+(lane+6*1024)*4);a7^=ldsm(peer+(lane+7*1024)*4);}}
  unsigned s=a0+a1+a2+a3+a4+a5+a6+a7;__shared__ unsigned red[T];red[threadIdx.x]=s;__syncthreads();for(int d=T/2;d;d>>=1){if(threadIdx.x<d)red[threadIdx.x]+=red[threadIdx.x+d];__syncthreads();}if(threadIdx.x==0)atomicAdd(out,(unsigned long long)red[0]);cl.sync();}

__global__ void k_dsm_write(unsigned cs,unsigned long long*out){extern __shared__ unsigned occupancy_pad[];__shared__ unsigned sm[SMWORDS];for(int i=threadIdx.x;i<SMWORDS;i+=blockDim.x)sm[i]=i+blockIdx.x;cg::cluster_group cl=cg::this_cluster();unsigned rank=cl.block_rank();if(threadIdx.x==0)occupancy_pad[0]=rank;unsigned base=(unsigned)__cvta_generic_to_shared(sm),peer;unsigned target=(rank+1)%cs;asm volatile("mapa.shared::cluster.u32 %0,%1,%2;":"=r"(peer):"r"(base),"r"(target));cl.sync();unsigned lane=threadIdx.x,v=lane+rank+1;
  #pragma unroll 1
  for(int i=0;i<ITERS;++i){stsm(peer+(lane+0*1024)*4,v+i);stsm(peer+(lane+1*1024)*4,v+i);stsm(peer+(lane+2*1024)*4,v+i);stsm(peer+(lane+3*1024)*4,v+i);stsm(peer+(lane+4*1024)*4,v+i);stsm(peer+(lane+5*1024)*4,v+i);stsm(peer+(lane+6*1024)*4,v+i);stsm(peer+(lane+7*1024)*4,v+i);}cl.sync();if(threadIdx.x==0)atomicAdd(out,(unsigned long long)sm[0]);}

template<bool RING>void launch(int grid,int cs,unsigned long long*out){cudaLaunchConfig_t c{};c.gridDim=grid;c.blockDim=T;c.dynamicSmemBytes=96<<10;cudaLaunchAttribute a{};a.id=cudaLaunchAttributeClusterDimension;a.val.clusterDim.x=cs;a.val.clusterDim.y=a.val.clusterDim.z=1;c.attrs=&a;c.numAttrs=1;CK(cudaLaunchKernelEx(&c,k_dsm<RING>,(unsigned)cs,out));}
void launch_write(int grid,int cs,unsigned long long*out){cudaLaunchConfig_t c{};c.gridDim=grid;c.blockDim=T;c.dynamicSmemBytes=96<<10;cudaLaunchAttribute a{};a.id=cudaLaunchAttributeClusterDimension;a.val.clusterDim.x=cs;a.val.clusterDim.y=a.val.clusterDim.z=1;c.attrs=&a;c.numAttrs=1;CK(cudaLaunchKernelEx(&c,k_dsm_write,(unsigned)cs,out));}
template<class F>double best(F f){cudaEvent_t a,b;cudaEventCreate(&a);cudaEventCreate(&b);for(int i=0;i<4;++i)f();CK(cudaDeviceSynchronize());std::vector<float>v;for(int i=0;i<RUNS;++i){cudaEventRecord(a);f();cudaEventRecord(b);cudaEventSynchronize(b);float x;cudaEventElapsedTime(&x,a,b);v.push_back(x);}std::sort(v.begin(),v.end());cudaEventDestroy(a);cudaEventDestroy(b);return v[0];}
int main(int argc,char**argv){int dev=0;for(int i=1;i+1<argc;++i)if(!strcmp(argv[i],"--dev"))dev=atoi(argv[++i]);CK(cudaSetDevice(dev));cudaDeviceProp p{};cudaGetDeviceProperties(&p,dev);CK(cudaFuncSetAttribute(k_dsm<false>,cudaFuncAttributeNonPortableClusterSizeAllowed,1));CK(cudaFuncSetAttribute(k_dsm<true>,cudaFuncAttributeNonPortableClusterSizeAllowed,1));CK(cudaFuncSetAttribute(k_dsm_write,cudaFuncAttributeNonPortableClusterSizeAllowed,1));CK(cudaFuncSetAttribute(k_dsm<false>,cudaFuncAttributeMaxDynamicSharedMemorySize,96<<10));CK(cudaFuncSetAttribute(k_dsm<true>,cudaFuncAttributeMaxDynamicSharedMemorySize,96<<10));CK(cudaFuncSetAttribute(k_dsm_write,cudaFuncAttributeMaxDynamicSharedMemorySize,96<<10));unsigned long long*out;cudaMalloc(&out,8);cudaMemset(out,0,8);for(int i=0;i<4;++i)launch<true>(20,2,out);CK(cudaDeviceSynchronize());printf("GPU %d: %s sm_%d%d, %d SM (shared allocation forces one CTA/SM)\n",dev,p.name,p.major,p.minor,p.multiProcessorCount);printf("cluster_size,clusters,one_way_read_GBps,ring_read_GBps,ring_write_GBps,one_way_ms,ring_read_ms,ring_write_ms\n");
  for(int cs:{2,4,8}){int clusters=p.multiProcessorCount/cs,grid=clusters*cs;double o=best([&]{launch<false>(grid,cs,out);});double r=best([&]{launch<true>(grid,cs,out);});double w=best([&]{launch_write(grid,cs,out);});double unit=double(clusters)*T*ITERS*8*4;printf("%d,%d,%.2f,%.2f,%.2f,%.4f,%.4f,%.4f\n",cs,clusters,unit/(o*1e6),(unit*cs)/(r*1e6),(unit*cs)/(w*1e6),o,r,w);}
  unsigned long long h;cudaMemcpy(&h,out,8,cudaMemcpyDeviceToHost);printf("observable_sink=%llu\n",h);cudaFree(out);}

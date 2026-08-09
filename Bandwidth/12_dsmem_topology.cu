// Cluster-rank distance/direction probe for the DSMEM interconnect.
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>
namespace cg=cooperative_groups;
#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA: %s line %d\n",cudaGetErrorString(e),__LINE__);exit(1);}}while(0)
constexpr int T=256,ITERS=8192,WORDS=8192;
__device__ __forceinline__ unsigned ldsm(unsigned a){unsigned v;asm volatile("ld.shared::cluster.u32 %0,[%1];":"=r"(v):"r"(a):"memory");return v;}
__global__ void topology(unsigned cs,unsigned delta,unsigned long long*out){extern __shared__ unsigned pad[];__shared__ unsigned sm[WORDS];for(int i=threadIdx.x;i<WORDS;i+=T)sm[i]=i+blockIdx.x;cg::cluster_group cl=cg::this_cluster();unsigned rank=cl.block_rank();if(threadIdx.x==0)pad[0]=rank;unsigned base=(unsigned)__cvta_generic_to_shared(sm),peer;asm volatile("mapa.shared::cluster.u32 %0,%1,%2;":"=r"(peer):"r"(base),"r"((rank+delta)%cs));cl.sync();unsigned a[8]={1,2,3,4,5,6,7,8},lane=threadIdx.x;
#pragma unroll 1
  for(int i=0;i<ITERS;++i){
#pragma unroll
    for(int j=0;j<8;++j)a[j]^=ldsm(peer+(lane+j*1024)*4);}
  unsigned v=0;
#pragma unroll
  for(int j=0;j<8;++j)v+=a[j];if((threadIdx.x&31)==0)atomicAdd(out,(unsigned long long)v);cl.sync();}
void launch(int grid,int cs,int delta,unsigned long long*out){cudaLaunchConfig_t c{};c.gridDim=grid;c.blockDim=T;c.dynamicSmemBytes=96<<10;cudaLaunchAttribute a{};a.id=cudaLaunchAttributeClusterDimension;a.val.clusterDim.x=cs;a.val.clusterDim.y=a.val.clusterDim.z=1;c.attrs=&a;c.numAttrs=1;CK(cudaLaunchKernelEx(&c,topology,(unsigned)cs,(unsigned)delta,out));}
double best(int grid,int cs,int delta,unsigned long long*out){cudaEvent_t a,b;cudaEventCreate(&a);cudaEventCreate(&b);launch(grid,cs,delta,out);CK(cudaDeviceSynchronize());std::vector<float>v;for(int i=0;i<7;++i){cudaEventRecord(a);launch(grid,cs,delta,out);cudaEventRecord(b);cudaEventSynchronize(b);float x;cudaEventElapsedTime(&x,a,b);v.push_back(x);}std::sort(v.begin(),v.end());cudaEventDestroy(a);cudaEventDestroy(b);return v[0];}
int main(int argc,char**argv){int dev=argc>1?atoi(argv[1]):0;CK(cudaSetDevice(dev));cudaDeviceProp p{};CK(cudaGetDeviceProperties(&p,dev));CK(cudaFuncSetAttribute(topology,cudaFuncAttributeNonPortableClusterSizeAllowed,1));CK(cudaFuncSetAttribute(topology,cudaFuncAttributeMaxDynamicSharedMemorySize,96<<10));unsigned long long*out;CK(cudaMalloc(&out,8));CK(cudaMemset(out,0,8));printf("cluster_size,rank_delta,direction,active_ctas,GBps,time_ms\n");for(int cs:{2,4,8}){int grid=(p.multiProcessorCount/cs)*cs;for(int d=1;d<cs;++d){double ms=best(grid,cs,d,out);double bytes=double(grid)*T*ITERS*8*4;printf("%d,%d,%s,%d,%.2f,%.4f\n",cs,d,d*2==cs?"opposite":(d*2<cs?"forward":"reverse"),grid,bytes/(ms*1e6),ms);}}unsigned long long h;CK(cudaMemcpy(&h,out,8,cudaMemcpyDeviceToHost));printf("observable_sink=%llu\n",h);cudaFree(out);}

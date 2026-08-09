// Trade per-thread memory-level parallelism against blocks/SM.
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA: %s line %d\n",cudaGetErrorString(e),__LINE__);exit(1);}}while(0)
constexpr int T=256;
template<int ILP>__global__ void mlp(const uint4*p,size_t n,unsigned long long*out){uint4 a[ILP];
#pragma unroll
 for(int j=0;j<ILP;++j)a[j]={0,0,0,0};size_t workers=size_t(gridDim.x)*T,step=workers*ILP;
 for(size_t base=size_t(blockIdx.x)*T+threadIdx.x;base<n;base+=step){
#pragma unroll
  for(int j=0;j<ILP;++j)if(base+j*workers<n){uint4 v=p[base+j*workers];a[j].x^=v.x;a[j].y^=v.y;a[j].z^=v.z;a[j].w^=v.w;}}
 unsigned v=0;
#pragma unroll
 for(int j=0;j<ILP;++j)v^=a[j].x^a[j].y^a[j].z^a[j].w;if((threadIdx.x&31)==0)atomicAdd(out,(unsigned long long)v);}
template<int I>double run(const uint4*p,size_t n,int blocks,unsigned long long*out){cudaEvent_t a,b;cudaEventCreate(&a);cudaEventCreate(&b);mlp<I><<<blocks,T>>>(p,n,out);CK(cudaDeviceSynchronize());std::vector<float>v;for(int r=0;r<7;++r){cudaEventRecord(a);mlp<I><<<blocks,T>>>(p,n,out);cudaEventRecord(b);cudaEventSynchronize(b);float x;cudaEventElapsedTime(&x,a,b);v.push_back(x);}std::sort(v.begin(),v.end());cudaEventDestroy(a);cudaEventDestroy(b);return v[0];}
int main(int argc,char**argv){int dev=argc>1?atoi(argv[1]):0,want_b=argc>2?atoi(argv[2]):0,want_i=argc>3?atoi(argv[3]):0;CK(cudaSetDevice(dev));cudaDeviceProp p{};CK(cudaGetDeviceProperties(&p,dev));size_t bytes=512ull<<20,n=bytes/16;uint4*src;unsigned long long*out;CK(cudaMalloc(&src,bytes));CK(cudaMalloc(&out,8));cudaMemset(src,1,bytes);cudaMemset(out,0,8);printf("blocks_per_sm,ILP,GBps,time_ms\n");for(int b:{1,2,4,8}){if(want_b&&b!=want_b)continue;int blocks=b*p.multiProcessorCount;for(int ilp:{1,2,4,8}){if(want_i&&ilp!=want_i)continue;double ms=ilp==1?run<1>(src,n,blocks,out):ilp==2?run<2>(src,n,blocks,out):ilp==4?run<4>(src,n,blocks,out):run<8>(src,n,blocks,out);printf("%d,%d,%.2f,%.4f\n",b,ilp,bytes/(ms*1e6),ms);}}unsigned long long h;cudaMemcpy(&h,out,8,cudaMemcpyDeviceToHost);printf("observable_sink=%llu\n",h);cudaFree(out);cudaFree(src);}

// cp.async global->shared sustained bandwidth versus in-flight groups.
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA: %s line %d\n",cudaGetErrorString(e),__LINE__);exit(1);}}while(0)
constexpr int T=256,RUNS=7,TILE=T*16;
template<int STAGES>__global__ void k_cpa(const char*src,size_t tiles,unsigned long long*out){extern __shared__ __align__(16) char sm[];unsigned sd=(unsigned)__cvta_generic_to_shared(sm);size_t iter=0;for(size_t tile=blockIdx.x;tile<tiles;tile+=gridDim.x,++iter){unsigned stage=iter%STAGES;const char*g=src+tile*TILE+threadIdx.x*16;asm volatile("cp.async.cg.shared::cta.global [%0],[%1],16;"::"r"(sd+stage*TILE+threadIdx.x*16),"l"(g):"memory");asm volatile("cp.async.commit_group;":::"memory");if(stage==STAGES-1)asm volatile("cp.async.wait_all;":::"memory");}asm volatile("cp.async.wait_all;":::"memory");__syncthreads();if(threadIdx.x==0){unsigned v=*(unsigned*)sm;atomicAdd(out,(unsigned long long)v);}}
template<class F>double best(F f){cudaEvent_t a,b;cudaEventCreate(&a);cudaEventCreate(&b);f();CK(cudaDeviceSynchronize());std::vector<float>v;for(int r=0;r<RUNS;++r){cudaEventRecord(a);f();cudaEventRecord(b);cudaEventSynchronize(b);float x;cudaEventElapsedTime(&x,a,b);v.push_back(x);}std::sort(v.begin(),v.end());cudaEventDestroy(a);cudaEventDestroy(b);return v[0];}
template<int S>void run(const char*src,size_t tiles,int blocks,unsigned long long*out){double ms=best([&]{k_cpa<S><<<blocks,T,S*TILE>>>(src,tiles,out);});double bytes=double(tiles)*TILE;printf("%d,%.2f,%.4f\n",S,bytes/(ms*1e6),ms);}
int main(int argc,char**argv){int dev=0;for(int i=1;i+1<argc;++i)if(!strcmp(argv[i],"--dev"))dev=atoi(argv[++i]);CK(cudaSetDevice(dev));cudaDeviceProp p{};cudaGetDeviceProperties(&p,dev);size_t bytes=512ull<<20;char*src;unsigned long long*out;cudaMalloc(&src,bytes);cudaMalloc(&out,8);cudaMemset(src,1,bytes);cudaMemset(out,0,8);int blocks=p.multiProcessorCount*4;printf("GPU %d: %s sm_%d%d, %d SM\n",dev,p.name,p.major,p.minor,p.multiProcessorCount);printf("inflight_groups,requested_GBps,time_ms\n");run<1>(src,bytes/TILE,blocks,out);run<2>(src,bytes/TILE,blocks,out);run<4>(src,bytes/TILE,blocks,out);run<8>(src,bytes/TILE,blocks,out);unsigned long long h;cudaMemcpy(&h,out,8,cudaMemcpyDeviceToHost);printf("observable_sink=%llu\n",h);cudaFree(out);cudaFree(src);}

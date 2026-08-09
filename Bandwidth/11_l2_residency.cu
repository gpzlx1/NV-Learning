// L2 eviction pressure and access-policy-window capability probe.
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA: %s line %d\n",cudaGetErrorString(e),__LINE__);exit(1);}}while(0)

__global__ void stream_read(const uint4 *p,size_t n,unsigned long long *out){uint4 a{0,0,0,0};for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=size_t(gridDim.x)*blockDim.x){uint4 v=p[i];a.x^=v.x;a.y^=v.y;a.z^=v.z;a.w^=v.w;}if((threadIdx.x&31)==0)atomicAdd(out,(unsigned long long)(a.x+a.y+a.z+a.w));}
double reload(uint4*hot,size_t hn,uint4*pollute,size_t pn,unsigned long long*out,int blocks,cudaStream_t s){cudaEvent_t a,b;CK(cudaEventCreate(&a));CK(cudaEventCreate(&b));std::vector<float>v;for(int i=0;i<9;++i){stream_read<<<blocks,256,0,s>>>(hot,hn,out);stream_read<<<blocks,256,0,s>>>(pollute,pn,out);cudaEventRecord(a,s);stream_read<<<blocks,256,0,s>>>(hot,hn,out);cudaEventRecord(b,s);cudaEventSynchronize(b);float x;cudaEventElapsedTime(&x,a,b);if(i)v.push_back(x);}std::sort(v.begin(),v.end());cudaEventDestroy(a);cudaEventDestroy(b);return v[0];}

int main(int argc,char**argv){int dev=argc>1?atoi(argv[1]):0;CK(cudaSetDevice(dev));cudaDeviceProp p{};CK(cudaGetDeviceProperties(&p,dev));
  printf("GPU=%s L2_bytes=%d persisting_max=%d access_window_max=%d\n",p.name,p.l2CacheSize,p.persistingL2CacheMaxSize,p.accessPolicyMaxWindowSize);
  size_t max_hot=32ull<<20,pollute_bytes=128ull<<20;uint4 *hot,*pollute;unsigned long long*out;CK(cudaMalloc(&hot,max_hot));CK(cudaMalloc(&pollute,pollute_bytes));CK(cudaMalloc(&out,8));CK(cudaMemset(hot,1,max_hot));CK(cudaMemset(pollute,2,pollute_bytes));CK(cudaMemset(out,0,8));int blocks=p.multiProcessorCount*8;
  cudaStream_t s;CK(cudaStreamCreate(&s));
  printf("hot_MiB,pollute_MiB,normal_reload_GBps,persist_reload_GBps,speedup\n");
  for(size_t hot_bytes:{4ull<<20,16ull<<20,32ull<<20}){cudaStreamAttrValue off{};off.accessPolicyWindow.base_ptr=nullptr;off.accessPolicyWindow.num_bytes=0;off.accessPolicyWindow.hitRatio=0;off.accessPolicyWindow.hitProp=cudaAccessPropertyNormal;off.accessPolicyWindow.missProp=cudaAccessPropertyNormal;CK(cudaStreamSetAttribute(s,cudaStreamAttributeAccessPolicyWindow,&off));CK(cudaCtxResetPersistingL2Cache());double normal=reload(hot,hot_bytes/16,pollute,pollute_bytes/16,out,blocks,s),persist=0;
    if(p.persistingL2CacheMaxSize>0&&p.accessPolicyMaxWindowSize>0){size_t reserve=std::min<size_t>(p.persistingL2CacheMaxSize,hot_bytes);CK(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize,reserve));cudaStreamAttrValue a{};a.accessPolicyWindow.base_ptr=hot;a.accessPolicyWindow.num_bytes=std::min<size_t>(hot_bytes,p.accessPolicyMaxWindowSize);a.accessPolicyWindow.hitRatio=std::min(1.0,double(reserve)/hot_bytes);a.accessPolicyWindow.hitProp=cudaAccessPropertyPersisting;a.accessPolicyWindow.missProp=cudaAccessPropertyStreaming;CK(cudaStreamSetAttribute(s,cudaStreamAttributeAccessPolicyWindow,&a));persist=reload(hot,hot_bytes/16,pollute,pollute_bytes/16,out,blocks,s);}
    double ng=hot_bytes/(normal*1e6),pg=persist?hot_bytes/(persist*1e6):0;printf("%zu,%zu,%.2f,%.2f,%.3f\n",hot_bytes>>20,pollute_bytes>>20,ng,pg,pg/ng);}
  unsigned long long h;CK(cudaMemcpy(&h,out,8,cudaMemcpyDeviceToHost));printf("observable_sink=%llu\n",h);cudaStreamDestroy(s);cudaFree(out);cudaFree(pollute);cudaFree(hot);}

// CUDA copy-engine bandwidth: device/device and ATS pinned-host paths.
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA: %s line %d\n",cudaGetErrorString(e),__LINE__);exit(1);}}while(0)
constexpr int RUNS=9;
template<class F>double best(cudaStream_t stream,F f){cudaEvent_t a,b;cudaEventCreate(&a);cudaEventCreate(&b);f();CK(cudaStreamSynchronize(stream));std::vector<float>v;for(int r=0;r<RUNS;++r){cudaEventRecord(a,stream);f();cudaEventRecord(b,stream);cudaEventSynchronize(b);float x;cudaEventElapsedTime(&x,a,b);v.push_back(x);}std::sort(v.begin(),v.end());cudaEventDestroy(a);cudaEventDestroy(b);return v[0];}
int main(int argc,char**argv){int dev=0;for(int i=1;i+1<argc;++i)if(!strcmp(argv[i],"--dev"))dev=atoi(argv[++i]);CK(cudaSetDevice(dev));cudaDeviceProp p{};cudaGetDeviceProperties(&p,dev);size_t bytes=512ull<<20;void*d0,*d1,*h;CK(cudaMalloc(&d0,bytes));CK(cudaMalloc(&d1,bytes));CK(cudaHostAlloc(&h,bytes,cudaHostAllocDefault));memset(h,1,bytes);cudaMemset(d0,0,bytes);cudaStream_t s;cudaStreamCreate(&s);
  printf("GPU %d: %s sm_%d%d, asyncEngineCount=%d, ATS host allocation %.0f MB\n",dev,p.name,p.major,p.minor,p.asyncEngineCount,bytes/1048576.0);printf("direction,GBps,time_ms\n");double dd=best(s,[&]{CK(cudaMemcpyAsync(d1,d0,bytes,cudaMemcpyDeviceToDevice,s));});double hd=best(s,[&]{CK(cudaMemcpyAsync(d0,h,bytes,cudaMemcpyHostToDevice,s));});double dh=best(s,[&]{CK(cudaMemcpyAsync(h,d0,bytes,cudaMemcpyDeviceToHost,s));});printf("D2D_payload,%.2f,%.4f\nH2D_pinned,%.2f,%.4f\nD2H_pinned,%.2f,%.4f\n",bytes/(dd*1e6),dd,bytes/(hd*1e6),hd,bytes/(dh*1e6),dh);cudaStreamDestroy(s);cudaFreeHost(h);cudaFree(d1);cudaFree(d0);}

// Sustained tcgen05 throughput, reusing the SASS-validated latency primitive.
#define main latency_tcgen05_main_not_used
#include "../Latency/06b_tcgen05.cu"
#undef main
#include <algorithm>

constexpr int BW_RUNS=7, BW_ITERS=4096;
template<class F> double bw_best(F f){cudaEvent_t a,b;CK(cudaEventCreate(&a));CK(cudaEventCreate(&b));f();CK(cudaDeviceSynchronize());std::vector<float>v;for(int r=0;r<BW_RUNS;++r){cudaEventRecord(a);f();cudaEventRecord(b);cudaEventSynchronize(b);float x;cudaEventElapsedTime(&x,a,b);v.push_back(x);}std::sort(v.begin(),v.end());cudaEventDestroy(a);cudaEventDestroy(b);return v[0];}
template<int N> void run_shape(int blocks){double ms=bw_best([&]{tcbench::k_tcgen05<N,0><<<blocks,128>>>(BW_ITERS,g_out);});double mma=double(blocks)*BW_ITERS*UNROLL;double flop=mma*(2.0*64*N*16);printf("M64N%dK16,%d,%.0f,%.4f,%.3f\n",N,blocks,mma,ms,flop/(ms*1e9));}
int main(int argc,char**argv){int dev=arg_dev(argc,argv,0);dev_header(dev);cudaDeviceProp p{};cudaGetDeviceProperties(&p,dev);if(p.major!=11||p.minor!=0){fprintf(stderr,"requires sm_110\n");return 2;}printf("shape,blocks,mma_instructions,time_ms,TFLOPs\n");run_shape<8>(p.multiProcessorCount);run_shape<16>(p.multiProcessorCount);run_shape<32>(p.multiProcessorCount);cudaFree(g_out);}

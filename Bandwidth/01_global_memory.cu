// Global-memory bandwidth and scaling microbenchmark.
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  fprintf(stderr, "CUDA error: %s (%s:%d)\n", cudaGetErrorString(e), __FILE__, __LINE__); \
  exit(1); } } while (0)

constexpr int THREADS = 256;
constexpr int RUNS = 7;

__device__ __forceinline__ float sum4(float4 v) { return v.x + v.y + v.z + v.w; }

__global__ void k_read(const float4* __restrict__ src, size_t n, float* sink) {
  float s = 0.0f;
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n; i += size_t(gridDim.x) * blockDim.x) s += sum4(src[i]);
  __shared__ float sm[THREADS];
  sm[threadIdx.x] = s; __syncthreads();
  for (int d = THREADS / 2; d; d >>= 1) {
    if (threadIdx.x < d) sm[threadIdx.x] += sm[threadIdx.x + d];
    __syncthreads();
  }
  if (threadIdx.x == 0) atomicAdd(sink, sm[0]);
}

__global__ void k_read64(const float2* __restrict__ src, size_t n, float* sink) {
  float s = 0.0f;
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n; i += size_t(gridDim.x) * blockDim.x) { float2 v = src[i]; s += v.x + v.y; }
  __shared__ float sm[THREADS]; sm[threadIdx.x] = s; __syncthreads();
  for (int d=THREADS/2; d; d>>=1) { if (threadIdx.x<d) sm[threadIdx.x]+=sm[threadIdx.x+d]; __syncthreads(); }
  if (threadIdx.x == 0) atomicAdd(sink, sm[0]);
}

__global__ void k_read32(const float* __restrict__ src, size_t n, float* sink) {
  float s = 0.0f;
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n; i += size_t(gridDim.x) * blockDim.x) s += src[i];
  __shared__ float sm[THREADS]; sm[threadIdx.x] = s; __syncthreads();
  for (int d=THREADS/2; d; d>>=1) { if (threadIdx.x<d) sm[threadIdx.x]+=sm[threadIdx.x+d]; __syncthreads(); }
  if (threadIdx.x == 0) atomicAdd(sink, sm[0]);
}

__global__ void k_write(float4* dst, size_t n, float seed) {
  float4 v = make_float4(seed, seed + 1, seed + 2, seed + 3);
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n; i += size_t(gridDim.x) * blockDim.x) dst[i] = v;
}

__global__ void k_copy(const float4* __restrict__ src, float4* __restrict__ dst, size_t n) {
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n; i += size_t(gridDim.x) * blockDim.x) dst[i] = src[i];
}

__global__ void k_stride(const float* __restrict__ src, size_t loads,
                         unsigned stride, float* sink) {
  float s = 0.0f;
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < loads; i += size_t(gridDim.x) * blockDim.x) s += src[i * stride];
  __shared__ float sm[THREADS];
  sm[threadIdx.x] = s; __syncthreads();
  for (int d = THREADS / 2; d; d >>= 1) {
    if (threadIdx.x < d) sm[threadIdx.x] += sm[threadIdx.x + d];
    __syncthreads();
  }
  if (threadIdx.x == 0) atomicAdd(sink, sm[0]);
}

template<class F> double best_ms(F launch) {
  cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
  launch(); CK(cudaDeviceSynchronize());
  std::vector<float> times;
  for (int r = 0; r < RUNS; ++r) {
    CK(cudaEventRecord(a)); launch(); CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
    float ms; CK(cudaEventElapsedTime(&ms, a, b)); times.push_back(ms);
  }
  CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
  std::sort(times.begin(), times.end());
  return times.front();
}

static double gbps(size_t bytes, double ms) { return double(bytes) / (ms * 1.0e6); }

int main(int argc, char** argv) {
  int dev = 0;
  for (int i = 1; i + 1 < argc; ++i) if (!strcmp(argv[i], "--dev")) dev = atoi(argv[++i]);
  CK(cudaSetDevice(dev));
  cudaDeviceProp p{}; CK(cudaGetDeviceProperties(&p, dev));
  printf("GPU %d: %s sm_%d%d, %d SM, L2 %.0f MB\n", dev, p.name, p.major, p.minor,
         p.multiProcessorCount, p.l2CacheSize / 1048576.0);

  size_t free_b = 0, total_b = 0; CK(cudaMemGetInfo(&free_b, &total_b));
  size_t bytes = 512ull << 20;
  if (free_b < 2 * bytes + (512ull << 20)) bytes = 256ull << 20;
  bytes &= ~(sizeof(float4) - 1);
  size_t n4 = bytes / sizeof(float4);
  float4 *a, *b; float* sink;
  CK(cudaMalloc(&a, bytes)); CK(cudaMalloc(&b, bytes)); CK(cudaMalloc(&sink, sizeof(float)));
  CK(cudaMemset(a, 1, bytes)); CK(cudaMemset(b, 0, bytes)); CK(cudaMemset(sink, 0, 4));
  printf("Allocation per array: %.0f MB; timing uses %d runs, minimum stable time\n",
         bytes / 1048576.0, RUNS);

  printf("\n=== Scaling: 128-bit vector access over device allocation ===\n");
  printf("blocks,blocks_per_sm,read_GBps,write_GBps,copy_payload_GBps,copy_fabric_GBps\n");
  for (int mul : {1, 2, 4, 8}) {
    int blocks = p.multiProcessorCount * mul;
    double tr = best_ms([&]{ k_read <<<blocks, THREADS>>>(a, n4, sink); });
    double tw = best_ms([&]{ k_write<<<blocks, THREADS>>>(b, n4, float(mul)); });
    double tc = best_ms([&]{ k_copy <<<blocks, THREADS>>>(a, b, n4); });
    CK(cudaGetLastError());
    printf("%d,%d,%.2f,%.2f,%.2f,%.2f\n", blocks, mul, gbps(bytes,tr), gbps(bytes,tw),
           gbps(bytes,tc), gbps(2*bytes,tc));
  }

  printf("\n=== Read access width at 8 blocks/SM ===\n");
  printf("access_bits,requested_GBps\n");
  int sat_blocks = p.multiProcessorCount * 8;
  double t32 = best_ms([&]{ k_read32<<<sat_blocks,THREADS>>>((float*)a, bytes/4, sink); });
  double t64 = best_ms([&]{ k_read64<<<sat_blocks,THREADS>>>((float2*)a, bytes/8, sink); });
  double t128 = best_ms([&]{ k_read<<<sat_blocks,THREADS>>>(a, n4, sink); });
  printf("32,%.2f\n64,%.2f\n128,%.2f\n", gbps(bytes,t32), gbps(bytes,t64), gbps(bytes,t128));

  printf("\n=== Coalescing: full-span strided float loads ===\n");
  printf("stride_elements,stride_bytes,requested_GBps,time_ms\n");
  size_t nf = bytes / sizeof(float);
  for (unsigned stride : {1u, 2u, 4u, 8u, 16u, 32u}) {
    // Traverse the full 512 MB span exactly once. Requested bytes shrink with
    // stride while memory transactions do not, exposing coalescing efficiency
    // without modulo wrap turning large strides into an L2-resident loop.
    size_t loads = nf / stride;
    int blocks = p.multiProcessorCount * 8;
    double t = best_ms([&]{ k_stride<<<blocks,THREADS>>>((float*)a, loads, stride, sink); });
    printf("%u,%u,%.2f,%.4f\n", stride, stride*4, gbps(loads*4,t), t);
  }

  float h = 0; CK(cudaMemcpy(&h, sink, 4, cudaMemcpyDeviceToHost));
  printf("\nobservable_sink=%g\n", h);
  CK(cudaFree(sink)); CK(cudaFree(b)); CK(cudaFree(a));
  return 0;
}

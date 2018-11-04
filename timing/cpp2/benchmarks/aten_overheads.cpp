#include <iostream>
#include <chrono>

#include <benchmark/benchmark.h>
#include <torch/torch.h>

static int WIPE_NO_CACHE = 0;
static int WIPE_L1_CACHE = 1;
static int WIPE_L1_L2_CACHE = 2;
static int WIPE_L1_L2_L3_CACHE = 3;

static int DONT_RUN_WORKLOAD = 0;
static int RUN_WORKLOAD = 1;

static uint32_t wipe_dcache(size_t wipe_size) {
  uint32_t* wipe_buffer = nullptr;

  if (wipe_buffer == nullptr) {
    /*
      On c1.small.x86 (Packet)
      L1 Data 32K (x4)
      L1 Instruction 32K (x4)
      L2 Unified 256K (x4)
      L3 Unified 8192K (x1)
      L1 + L2: 288K
      L1 + L2 + L3: 8480K
    */
    wipe_buffer = static_cast<uint32_t*>(malloc(wipe_size));
    AT_ASSERT(wipe_buffer != nullptr);
  }
  uint32_t hash = 0;
  for (uint32_t i = 0; i * sizeof(uint32_t) < wipe_size; i += 8) {
    hash ^= wipe_buffer[i];
    wipe_buffer[i] = hash;
  }
  /* Make sure compiler doesn't optimize the loop away */
  return hash;
}

static void BM_TensorDim(benchmark::State& state) {
  int wipe_level = state.range(0);
  bool run_workload = (state.range(1) == RUN_WORKLOAD);

  auto options = at::TensorOptions(at::kCPU);

  // initialize some cuda...
  auto tmp = at::empty({0}, options);
  int64_t res = 0;

  for (auto _ : state) {
    if (wipe_level == WIPE_NO_CACHE) {
      // no-op
    } else if (wipe_level == WIPE_L1_CACHE) {  // Wipe L1 cache
      wipe_dcache(32 * 1024);
    } else if (wipe_level == WIPE_L1_L2_CACHE) {  // Wipe L1 + L2 cache
      wipe_dcache((32 + 256) * 1024);
    } else if (wipe_level == WIPE_L1_L2_L3_CACHE) {  // Wipe L1 + L2 + L3 cache
      wipe_dcache((32 + 256 + 8192) * 1024);
    }
    
    auto start = std::chrono::high_resolution_clock::now();

    if (run_workload) {
      benchmark::DoNotOptimize(res = tmp.dim());
    }

    auto end   = std::chrono::high_resolution_clock::now();
    auto elapsed_seconds =
      std::chrono::duration_cast<std::chrono::duration<double>>(
        end - start);

    state.SetIterationTime(elapsed_seconds.count());
  }
  std::ostream cnull(0);
  cnull << res;
}
// No wipe
BENCHMARK(BM_TensorDim)->Args({WIPE_NO_CACHE, DONT_RUN_WORKLOAD})->UseManualTime()->Iterations(1000000);
BENCHMARK(BM_TensorDim)->Args({WIPE_NO_CACHE, RUN_WORKLOAD})->UseManualTime()->Iterations(1000000);

// Wipe L1
BENCHMARK(BM_TensorDim)->Args({WIPE_L1_CACHE, DONT_RUN_WORKLOAD})->UseManualTime()->Iterations(100000);
BENCHMARK(BM_TensorDim)->Args({WIPE_L1_CACHE, RUN_WORKLOAD})->UseManualTime()->Iterations(100000);

// Wipe L1 + L2
BENCHMARK(BM_TensorDim)->Args({WIPE_L1_L2_CACHE, DONT_RUN_WORKLOAD})->UseManualTime()->Iterations(10000);
BENCHMARK(BM_TensorDim)->Args({WIPE_L1_L2_CACHE, RUN_WORKLOAD})->UseManualTime()->Iterations(10000);

// Wipe L1 + L2 + L3
BENCHMARK(BM_TensorDim)->Args({WIPE_L1_L2_L3_CACHE, DONT_RUN_WORKLOAD})->UseManualTime()->Iterations(1000);
BENCHMARK(BM_TensorDim)->Args({WIPE_L1_L2_L3_CACHE, RUN_WORKLOAD})->UseManualTime()->Iterations(1000);

// static void BM_VariableDim(benchmark::State& state) {
//   auto options = at::TensorOptions(at::kCPU);

//   // initialize some cuda...
//   auto tmp = torch::empty({0}, options);
//   int64_t res = 0;

//   for (auto _ : state) {
//     benchmark::DoNotOptimize(res = tmp.dim());
//   }
//   std::ostream cnull(0);
//   cnull << res;
// }
// BENCHMARK(BM_VariableDim);

// static void BM_TensorNumel(benchmark::State& state) {
//   auto options = at::TensorOptions(at::kCPU);

//   // initialize some cuda...
//   auto tmp = at::empty({0}, options);
//   int64_t res = 0;

//   for (auto _ : state) {
//     benchmark::DoNotOptimize(res = tmp.numel());
//   }
//   std::ostream cnull(0);
//   cnull << res;
// }
// BENCHMARK(BM_TensorNumel);

// static void BM_VariableNumel(benchmark::State& state) {
//   auto options = at::TensorOptions(at::kCPU);

//   // initialize some cuda...
//   auto tmp = torch::empty({0}, options);
//   int64_t res = 0;

//   for (auto _ : state) {
//     benchmark::DoNotOptimize(res = tmp.numel());
//   }
//   std::ostream cnull(0);
//   cnull << res;
// }
// BENCHMARK(BM_VariableNumel);

// static void BM_TensorSize(benchmark::State& state) {
//   auto options = at::TensorOptions(at::kCPU);

//   // initialize some cuda...
//   auto tmp = at::empty({0}, options);
//   int64_t res = 0;

//   for (auto _ : state) {
//     benchmark::DoNotOptimize(res = tmp.size(0));
//   }
//   std::ostream cnull(0);
//   cnull << res;
// }
// BENCHMARK(BM_TensorSize);

// static void BM_VariableSize(benchmark::State& state) {
//   auto options = at::TensorOptions(at::kCPU);

//   // initialize some cuda...
//   auto tmp = torch::empty({0}, options);
//   int64_t res = 0;

//   for (auto _ : state) {
//     benchmark::DoNotOptimize(res = tmp.size(0));
//   }
//   std::ostream cnull(0);
//   cnull << res;
// }
// BENCHMARK(BM_VariableSize);

// static void BM_TensorSizes(benchmark::State& state) {
//   auto options = at::TensorOptions(at::kCPU);

//   // initialize some cuda...
//   auto tmp = at::empty({0}, options);
//   at::IntList res;

//   for (auto _ : state) {
//     benchmark::DoNotOptimize(res = tmp.sizes());
//   }
//   std::ostream cnull(0);
//   cnull << res;
// }
// BENCHMARK(BM_TensorSizes);

// static void BM_VariableSizes(benchmark::State& state) {
//   auto options = at::TensorOptions(at::kCPU);

//   // initialize some cuda...
//   auto tmp = torch::empty({0}, options);
//   at::IntList res;

//   for (auto _ : state) {
//     benchmark::DoNotOptimize(res = tmp.sizes());
//   }
//   std::ostream cnull(0);
//   cnull << res;
// }
// BENCHMARK(BM_VariableSizes);

// static void BM_EmptyTensorNoopResize(benchmark::State& state) {
//   auto options = at::TensorOptions(at::kCPU);
//   std::vector<long int> sizes({0});

//   // initialize some cuda...
//   auto tmp = at::empty({0}, options);
//   tmp.resize_(sizes);

//   for (auto _ : state) {
//     tmp.resize_(sizes);
//   }
// }
// BENCHMARK(BM_EmptyTensorNoopResize);

// static void BM_EmptyVariableNoopResize(benchmark::State& state) {
//   auto options = at::TensorOptions(at::kCPU);
//   std::vector<long int> sizes({0});

//   // initialize some cuda...
//   auto tmp = torch::empty({0}, options);
//   tmp.resize_(sizes);

//   for (auto _ : state) {
//     tmp.resize_(sizes);
//   }
// }
// BENCHMARK(BM_EmptyVariableNoopResize);

// static void BM_TensorNoopResize(benchmark::State& state) {
//   auto options = at::TensorOptions(at::kCPU);
//   std::vector<long int> sizes({64, 2048});

//   // initialize some cuda...
//   auto tmp = at::empty({0}, options);
//   tmp.resize_(sizes);

//   for (auto _ : state) {
//     tmp.resize_(sizes);
//   }
// }
// BENCHMARK(BM_TensorNoopResize);

// static void BM_VariableNoopResize(benchmark::State& state) {
//   auto options = at::TensorOptions(at::kCPU);
//   std::vector<long int> sizes({64, 2048});

//   // initialize some cuda...
//   auto tmp = torch::empty({0}, options);
//   tmp.resize_(sizes);

//   for (auto _ : state) {
//     tmp.resize_(sizes);
//   }
// }
// BENCHMARK(BM_VariableNoopResize);

BENCHMARK_MAIN();


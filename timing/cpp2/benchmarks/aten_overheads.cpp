#include <iostream>
#include <benchmark/benchmark.h>
#include <torch/torch.h>

// TODO: is -O0 a better idea for preventing loading cached value? How to let GCC evaluate the function every time?
// TODO: read https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html

static uint32_t wipe_cache() {
  static uint32_t* wipe_buffer = nullptr;
  static size_t wipe_size = 0;

  if (wipe_buffer == nullptr) {
    /*
      On g3.8xlarge:
      L1d cache:             32K
      L1i cache:             32K
      L2 cache:              256K
      L3 cache:              46080K
      Data cache total:      46368K
    */
    wipe_size = 32 * 1024;  // 32K
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

static void BM_WipeCache(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCPU);

  for (auto _ : state) {
    benchmark::DoNotOptimize(wipe_cache());
  }
}
BENCHMARK(BM_WipeCache);

static void BM_TensorDim(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCPU);

  // initialize some cuda...
  auto tmp = at::empty({0}, options);
  int64_t res = 0;

  for (auto _ : state) {
    benchmark::DoNotOptimize(wipe_cache());
    benchmark::DoNotOptimize(res = tmp.dim());
  }
  std::ostream cnull(0);
  cnull << res;
}
BENCHMARK(BM_TensorDim);

static void BM_VariableDim(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCPU);

  // initialize some cuda...
  auto tmp = torch::empty({0}, options);
  int64_t res = 0;

  for (auto _ : state) {
    benchmark::DoNotOptimize(res = tmp.dim());
  }
  std::ostream cnull(0);
  cnull << res;
}
BENCHMARK(BM_VariableDim);

static void BM_TensorNumel(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCPU);

  // initialize some cuda...
  auto tmp = at::empty({0}, options);
  int64_t res = 0;

  for (auto _ : state) {
    benchmark::DoNotOptimize(res = tmp.numel());
  }
  std::ostream cnull(0);
  cnull << res;
}
BENCHMARK(BM_TensorNumel);

static void BM_VariableNumel(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCPU);

  // initialize some cuda...
  auto tmp = torch::empty({0}, options);
  int64_t res = 0;

  for (auto _ : state) {
    benchmark::DoNotOptimize(res = tmp.numel());
  }
  std::ostream cnull(0);
  cnull << res;
}
BENCHMARK(BM_VariableNumel);

static void BM_TensorSize(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCPU);

  // initialize some cuda...
  auto tmp = at::empty({0}, options);
  int64_t res = 0;

  for (auto _ : state) {
    benchmark::DoNotOptimize(res = tmp.size(0));
  }
  std::ostream cnull(0);
  cnull << res;
}
BENCHMARK(BM_TensorSize);

static void BM_VariableSize(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCPU);

  // initialize some cuda...
  auto tmp = torch::empty({0}, options);
  int64_t res = 0;

  for (auto _ : state) {
    benchmark::DoNotOptimize(res = tmp.size(0));
  }
  std::ostream cnull(0);
  cnull << res;
}
BENCHMARK(BM_VariableSize);

static void BM_TensorSizes(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCPU);

  // initialize some cuda...
  auto tmp = at::empty({0}, options);
  at::IntList res;

  for (auto _ : state) {
    benchmark::DoNotOptimize(res = tmp.sizes());
  }
  std::ostream cnull(0);
  cnull << res;
}
BENCHMARK(BM_TensorSizes);

static void BM_VariableSizes(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCPU);

  // initialize some cuda...
  auto tmp = torch::empty({0}, options);
  at::IntList res;

  for (auto _ : state) {
    benchmark::DoNotOptimize(res = tmp.sizes());
  }
  std::ostream cnull(0);
  cnull << res;
}
BENCHMARK(BM_VariableSizes);

static void BM_EmptyTensorNoopResize(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCPU);
  std::vector<long int> sizes({0});

  // initialize some cuda...
  auto tmp = at::empty({0}, options);
  tmp.resize_(sizes);

  for (auto _ : state) {
    tmp.resize_(sizes);
  }
}
BENCHMARK(BM_EmptyTensorNoopResize);

static void BM_EmptyVariableNoopResize(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCPU);
  std::vector<long int> sizes({0});

  // initialize some cuda...
  auto tmp = torch::empty({0}, options);
  tmp.resize_(sizes);

  for (auto _ : state) {
    tmp.resize_(sizes);
  }
}
BENCHMARK(BM_EmptyVariableNoopResize);

static void BM_TensorNoopResize(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCPU);
  std::vector<long int> sizes({64, 2048});

  // initialize some cuda...
  auto tmp = at::empty({0}, options);
  tmp.resize_(sizes);

  for (auto _ : state) {
    tmp.resize_(sizes);
  }
}
BENCHMARK(BM_TensorNoopResize);

static void BM_VariableNoopResize(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCPU);
  std::vector<long int> sizes({64, 2048});

  // initialize some cuda...
  auto tmp = torch::empty({0}, options);
  tmp.resize_(sizes);

  for (auto _ : state) {
    tmp.resize_(sizes);
  }
}
BENCHMARK(BM_VariableNoopResize);

BENCHMARK_MAIN();


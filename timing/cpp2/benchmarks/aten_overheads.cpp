#include <iostream>
#include <benchmark/benchmark.h>
#include <torch/torch.h>

static void BM_TensorDim(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCPU);

  // initialize some cuda...
  auto tmp = at::empty({0}, options);

  for (auto _ : state) {
    benchmark::DoNotOptimize(tmp.dim());
  }
}
BENCHMARK(BM_TensorDim);

static void BM_VariableDim(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCPU);

  // initialize some cuda...
  auto tmp = torch::empty({0}, options);

  for (auto _ : state) {
    benchmark::DoNotOptimize(tmp.dim());
  }
}
BENCHMARK(BM_VariableDim);

static void BM_TensorNumel(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCPU);

  // initialize some cuda...
  auto tmp = at::empty({0}, options);

  for (auto _ : state) {
    benchmark::DoNotOptimize(tmp.numel());
  }
}
BENCHMARK(BM_TensorNumel);

static void BM_VariableNumel(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCPU);

  // initialize some cuda...
  auto tmp = torch::empty({0}, options);

  for (auto _ : state) {
    benchmark::DoNotOptimize(tmp.numel());
  }
}
BENCHMARK(BM_VariableNumel);

static void BM_TensorSize(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCPU);

  // initialize some cuda...
  auto tmp = at::empty({0}, options);

  for (auto _ : state) {
    benchmark::DoNotOptimize(tmp.size(0));
  }
}
BENCHMARK(BM_TensorSize);

static void BM_VariableSize(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCPU);

  // initialize some cuda...
  auto tmp = torch::empty({0}, options);

  for (auto _ : state) {
    benchmark::DoNotOptimize(tmp.size(0));
  }
}
BENCHMARK(BM_VariableSize);

static void BM_TensorSizes(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCPU);

  // initialize some cuda...
  auto tmp = at::empty({0}, options);

  for (auto _ : state) {
    benchmark::DoNotOptimize(tmp.sizes());
  }
}
BENCHMARK(BM_TensorSizes);

static void BM_VariableSizes(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCPU);

  // initialize some cuda...
  auto tmp = torch::empty({0}, options);

  for (auto _ : state) {
    benchmark::DoNotOptimize(tmp.sizes());
  }
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


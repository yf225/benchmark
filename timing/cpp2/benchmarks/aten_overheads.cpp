#include <iostream>
#include <benchmark/benchmark.h>
#include <torch/torch.h>
#include <ATen/cuda/CUDAContext.h>
#include <typeinfo>

static void BM_VariableClassName(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCPU);

  auto a_at = at::ones({2, 2});
  auto a_th = torch::ones({2, 2});

  std::cout << "typeid(a_at): " << typeid(a_at).name() << "\n";
  std::cout << "typeid(a_th): " << typeid(a_th).name() << "\n";
}
BENCHMARK(BM_VariableClassName);

static void BM_TensorNoGrad(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCPU);

  auto a = at::ones({2, 2}, at::requires_grad());
  auto b = at::randn({2, 2}, at::requires_grad());
  auto c = a + b;
  c.backward(); // a.grad() will now hold the gradient of c w.r.t. a.
  std::cout << "a.grad(): " << a.grad() << "\n";
  std::cout << "b.grad(): " << b.grad() << "\n";
}
BENCHMARK(BM_TensorNoGrad);

static void BM_VariableNoGrad(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCPU);

  auto a = torch::ones({2, 2});
  auto b = torch::randn({2, 2}, at::requires_grad());
  auto c = a + b;
  c.backward(); // a.grad() will now hold the gradient of c w.r.t. a.
  std::cout << "a.grad(): " << a.grad() << "\n";
  std::cout << "b.grad(): " << b.grad() << "\n";
}
BENCHMARK(BM_VariableNoGrad);

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

static void BM_AtenEmptyCuda(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = at::empty({0}, options);

  for (auto _ : state) {
    auto tensor = at::native::empty_cuda({0}, options);
  }
}
BENCHMARK(BM_AtenEmptyCuda);

static void BM_AtenEmpty(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCPU);

  // initialize some cuda...
  auto tmp = at::empty({0}, options);

  for (auto _ : state) {
    auto tensor = at::empty({0}, options);
  }
}
BENCHMARK(BM_AtenEmpty);

static void BM_VariableEmpty(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCPU);

  // initialize some cuda...
  auto tmp = torch::empty({0}, options);

  for (auto _ : state) {
    auto tensor = torch::empty({0}, options);
  }
}
BENCHMARK(BM_VariableEmpty);

static void BM_AtenEmptyResize(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCPU);
  std::vector<long int> sizes({64, 2048});

  // initialize some cuda...
  auto tmp = at::empty({0}, options);
  tmp.resize_(sizes);

  for (auto _ : state) {
    auto tensor = at::empty({0}, options);
    tensor.resize_(sizes);
  }
}
BENCHMARK(BM_AtenEmptyResize);

static void BM_VariableEmptyResize(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCPU);
  std::vector<long int> sizes({64, 2048});
  std::vector<long int> zero({0});

  // initialize some cuda...
  auto tmp = torch::empty(zero, options);
  tmp.resize_(sizes);

  for (auto _ : state) {
    auto tensor = torch::empty(zero, options);
    tensor.resize_(sizes);
  }
}
BENCHMARK(BM_VariableEmptyResize);

static void BM_AtenEmptyNoResize(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCPU);
  std::vector<long int> sizes({64, 2048});

  // initialize some cuda...
  auto tmp = at::empty({0}, options);
  tmp.resize_(sizes);

  for (auto _ : state) {
    auto tensor = at::empty(sizes, options);
  }
}
BENCHMARK(BM_AtenEmptyNoResize);

static void BM_VariableEmptyNoResize(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCPU);
  std::vector<long int> sizes({64, 2048});
  std::vector<long int> zero({0});

  // initialize some cuda...
  auto tmp = torch::empty(zero, options);
  tmp.resize_(sizes);

  for (auto _ : state) {
    benchmark::DoNotOptimize(torch::empty(sizes, options));
  }
}
BENCHMARK(BM_VariableEmptyNoResize);

BENCHMARK_MAIN();


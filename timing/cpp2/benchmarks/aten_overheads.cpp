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

static size_t get_cache_size(int wipe_level) {
  /*
    On c1.small.x86 (E3-1240 V5) (Packet)
    L1 Data 32K (x4)
    L1 Instruction 32K (x4)
    L2 Unified 256K (x4)
    L3 Unified 8192K (x1)
  */
  if (wipe_level == WIPE_NO_CACHE) {
    return 0;
  } else if (wipe_level == WIPE_L1_CACHE) {  // Wipe L1 cache
    return 32 * 1024;
  } else if (wipe_level == WIPE_L1_L2_CACHE) {  // Wipe L1 + L2 cache
    return (32 + 256) * 1024;
  } else if (wipe_level == WIPE_L1_L2_L3_CACHE) {  // Wipe L1 + L2 + L3 cache
    return (32 + 256 + 8192) * 1024;
  }
}

static uint32_t* wipe_dcache_setup(int wipe_level) {
  uint32_t* wipe_buffer = nullptr;
  size_t wipe_size = get_cache_size(wipe_level);

  if (wipe_buffer == nullptr) {
    wipe_buffer = static_cast<uint32_t*>(malloc(wipe_size));
    AT_ASSERT(wipe_buffer != nullptr);
  }
  uint32_t hash = 0;
  for (uint32_t i = 0; i * sizeof(uint32_t) < wipe_size; i += 8) {
    hash ^= std::rand();
    wipe_buffer[i] = hash;
  }
  /* Make sure compiler doesn't optimize the loop away */
  return wipe_buffer;
}

static void wipe_dcache_teardown(int wipe_level, uint32_t* wipe_buffer) {
  size_t wipe_size = get_cache_size(wipe_level);

  if (wipe_size > 0) {
    std::ostream cnull(0);
    for (uint32_t i = 0; i * sizeof(uint32_t) < wipe_size; i += 8) {
      cnull << wipe_buffer[i];
    }
    free(wipe_buffer);
  }
}

static int random_in_range(int min, int max) {
  static bool first = true;
  if (first) {
    srand( time(NULL) ); //seeding for the first time only!
    first = false;
  }
  return min + rand() % (( max + 1 ) - min);
}

static void BM_TensorDim(benchmark::State& state) {
  int wipe_level = state.range(0);
  bool run_workload = (state.range(1) == RUN_WORKLOAD);
  int64_t res = 0;

  for (auto _ : state) {
    // Workload setup
    std::vector<int64_t> tensor_sizes = {};
    for (int i = 0; i < random_in_range(1, 5); i++) {
      tensor_sizes.push_back(2);
    }
    auto tmp = at::empty(tensor_sizes, at::TensorOptions(at::kCPU));

    uint32_t* wipe_buffer = wipe_dcache_setup(wipe_level);
    auto start = std::chrono::high_resolution_clock::now();

    if (run_workload) {
      // Workload
      benchmark::DoNotOptimize(res = tmp.dim());
    }

    auto end   = std::chrono::high_resolution_clock::now();
    auto elapsed_seconds =
      std::chrono::duration_cast<std::chrono::duration<double>>(
        end - start);
    state.SetIterationTime(elapsed_seconds.count());
    wipe_dcache_teardown(wipe_level, wipe_buffer);
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

static void BM_VariableDim(benchmark::State& state) {
  int wipe_level = state.range(0);
  bool run_workload = (state.range(1) == RUN_WORKLOAD);
  int64_t res = 0;

  for (auto _ : state) {
    // Workload setup
    std::vector<int64_t> tensor_sizes = {};
    for (int i = 0; i < random_in_range(1, 5); i++) {
      tensor_sizes.push_back(2);
    }
    auto tmp = torch::empty(tensor_sizes, at::TensorOptions(at::kCPU));

    uint32_t* wipe_buffer = wipe_dcache_setup(wipe_level);
    auto start = std::chrono::high_resolution_clock::now();

    if (run_workload) {
      // Workload
      benchmark::DoNotOptimize(res = tmp.dim());
    }

    auto end   = std::chrono::high_resolution_clock::now();
    auto elapsed_seconds =
      std::chrono::duration_cast<std::chrono::duration<double>>(
        end - start);
    state.SetIterationTime(elapsed_seconds.count());
    wipe_dcache_teardown(wipe_level, wipe_buffer);
  }
  std::ostream cnull(0);
  cnull << res;
}
// No wipe
BENCHMARK(BM_VariableDim)->Args({WIPE_NO_CACHE, DONT_RUN_WORKLOAD})->UseManualTime()->Iterations(1000000);
BENCHMARK(BM_VariableDim)->Args({WIPE_NO_CACHE, RUN_WORKLOAD})->UseManualTime()->Iterations(1000000);
// Wipe L1
BENCHMARK(BM_VariableDim)->Args({WIPE_L1_CACHE, DONT_RUN_WORKLOAD})->UseManualTime()->Iterations(100000);
BENCHMARK(BM_VariableDim)->Args({WIPE_L1_CACHE, RUN_WORKLOAD})->UseManualTime()->Iterations(100000);
// Wipe L1 + L2
BENCHMARK(BM_VariableDim)->Args({WIPE_L1_L2_CACHE, DONT_RUN_WORKLOAD})->UseManualTime()->Iterations(10000);
BENCHMARK(BM_VariableDim)->Args({WIPE_L1_L2_CACHE, RUN_WORKLOAD})->UseManualTime()->Iterations(10000);
// Wipe L1 + L2 + L3
BENCHMARK(BM_VariableDim)->Args({WIPE_L1_L2_L3_CACHE, DONT_RUN_WORKLOAD})->UseManualTime()->Iterations(1000);
BENCHMARK(BM_VariableDim)->Args({WIPE_L1_L2_L3_CACHE, RUN_WORKLOAD})->UseManualTime()->Iterations(1000);

static void BM_TensorNumel(benchmark::State& state) {
  int wipe_level = state.range(0);
  bool run_workload = (state.range(1) == RUN_WORKLOAD);
  int64_t res = 0;

  for (auto _ : state) {
    // Workload setup
    std::vector<int64_t> tensor_sizes = {};
    for (int i = 0; i < random_in_range(1, 5); i++) {
      tensor_sizes.push_back(2);
    }
    auto tmp = at::empty(tensor_sizes, at::TensorOptions(at::kCPU));

    uint32_t* wipe_buffer = wipe_dcache_setup(wipe_level);
    auto start = std::chrono::high_resolution_clock::now();

    if (run_workload) {
      // Workload
      benchmark::DoNotOptimize(res = tmp.numel());
    }

    auto end   = std::chrono::high_resolution_clock::now();
    auto elapsed_seconds =
      std::chrono::duration_cast<std::chrono::duration<double>>(
        end - start);
    state.SetIterationTime(elapsed_seconds.count());
    wipe_dcache_teardown(wipe_level, wipe_buffer);
  }
  std::ostream cnull(0);
  cnull << res;
}
// No wipe
BENCHMARK(BM_TensorNumel)->Args({WIPE_NO_CACHE, DONT_RUN_WORKLOAD})->UseManualTime()->Iterations(1000000);
BENCHMARK(BM_TensorNumel)->Args({WIPE_NO_CACHE, RUN_WORKLOAD})->UseManualTime()->Iterations(1000000);
// Wipe L1
BENCHMARK(BM_TensorNumel)->Args({WIPE_L1_CACHE, DONT_RUN_WORKLOAD})->UseManualTime()->Iterations(100000);
BENCHMARK(BM_TensorNumel)->Args({WIPE_L1_CACHE, RUN_WORKLOAD})->UseManualTime()->Iterations(100000);
// Wipe L1 + L2
BENCHMARK(BM_TensorNumel)->Args({WIPE_L1_L2_CACHE, DONT_RUN_WORKLOAD})->UseManualTime()->Iterations(10000);
BENCHMARK(BM_TensorNumel)->Args({WIPE_L1_L2_CACHE, RUN_WORKLOAD})->UseManualTime()->Iterations(10000);
// Wipe L1 + L2 + L3
BENCHMARK(BM_TensorNumel)->Args({WIPE_L1_L2_L3_CACHE, DONT_RUN_WORKLOAD})->UseManualTime()->Iterations(1000);
BENCHMARK(BM_TensorNumel)->Args({WIPE_L1_L2_L3_CACHE, RUN_WORKLOAD})->UseManualTime()->Iterations(1000);

static void BM_VariableNumel(benchmark::State& state) {
  int wipe_level = state.range(0);
  bool run_workload = (state.range(1) == RUN_WORKLOAD);
  int64_t res = 0;

  for (auto _ : state) {
    // Workload setup
    std::vector<int64_t> tensor_sizes = {};
    for (int i = 0; i < random_in_range(1, 5); i++) {
      tensor_sizes.push_back(2);
    }
    auto tmp = torch::empty(tensor_sizes, at::TensorOptions(at::kCPU));

    uint32_t* wipe_buffer = wipe_dcache_setup(wipe_level);
    auto start = std::chrono::high_resolution_clock::now();

    if (run_workload) {
      // Workload
      benchmark::DoNotOptimize(res = tmp.numel());
    }

    auto end   = std::chrono::high_resolution_clock::now();
    auto elapsed_seconds =
      std::chrono::duration_cast<std::chrono::duration<double>>(
        end - start);
    state.SetIterationTime(elapsed_seconds.count());
    wipe_dcache_teardown(wipe_level, wipe_buffer);
  }
  std::ostream cnull(0);
  cnull << res;
}
// No wipe
BENCHMARK(BM_VariableNumel)->Args({WIPE_NO_CACHE, DONT_RUN_WORKLOAD})->UseManualTime()->Iterations(1000000);
BENCHMARK(BM_VariableNumel)->Args({WIPE_NO_CACHE, RUN_WORKLOAD})->UseManualTime()->Iterations(1000000);
// Wipe L1
BENCHMARK(BM_VariableNumel)->Args({WIPE_L1_CACHE, DONT_RUN_WORKLOAD})->UseManualTime()->Iterations(100000);
BENCHMARK(BM_VariableNumel)->Args({WIPE_L1_CACHE, RUN_WORKLOAD})->UseManualTime()->Iterations(100000);
// Wipe L1 + L2
BENCHMARK(BM_VariableNumel)->Args({WIPE_L1_L2_CACHE, DONT_RUN_WORKLOAD})->UseManualTime()->Iterations(10000);
BENCHMARK(BM_VariableNumel)->Args({WIPE_L1_L2_CACHE, RUN_WORKLOAD})->UseManualTime()->Iterations(10000);
// Wipe L1 + L2 + L3
BENCHMARK(BM_VariableNumel)->Args({WIPE_L1_L2_L3_CACHE, DONT_RUN_WORKLOAD})->UseManualTime()->Iterations(1000);
BENCHMARK(BM_VariableNumel)->Args({WIPE_L1_L2_L3_CACHE, RUN_WORKLOAD})->UseManualTime()->Iterations(1000);

static void BM_TensorSize(benchmark::State& state) {
  int wipe_level = state.range(0);
  bool run_workload = (state.range(1) == RUN_WORKLOAD);
  int64_t res = 0;

  for (auto _ : state) {
    // Workload setup
    std::vector<int64_t> tensor_sizes = {random_in_range(1, 5)};
    auto tmp = at::empty(tensor_sizes, at::TensorOptions(at::kCPU));

    uint32_t* wipe_buffer = wipe_dcache_setup(wipe_level);
    auto start = std::chrono::high_resolution_clock::now();

    if (run_workload) {
      // Workload
      benchmark::DoNotOptimize(res = tmp.size(0));
    }

    auto end   = std::chrono::high_resolution_clock::now();
    auto elapsed_seconds =
      std::chrono::duration_cast<std::chrono::duration<double>>(
        end - start);
    state.SetIterationTime(elapsed_seconds.count());
    wipe_dcache_teardown(wipe_level, wipe_buffer);
  }
  std::ostream cnull(0);
  cnull << res;
}
// No wipe
BENCHMARK(BM_TensorSize)->Args({WIPE_NO_CACHE, DONT_RUN_WORKLOAD})->UseManualTime()->Iterations(1000000);
BENCHMARK(BM_TensorSize)->Args({WIPE_NO_CACHE, RUN_WORKLOAD})->UseManualTime()->Iterations(1000000);
// Wipe L1
BENCHMARK(BM_TensorSize)->Args({WIPE_L1_CACHE, DONT_RUN_WORKLOAD})->UseManualTime()->Iterations(100000);
BENCHMARK(BM_TensorSize)->Args({WIPE_L1_CACHE, RUN_WORKLOAD})->UseManualTime()->Iterations(100000);
// Wipe L1 + L2
BENCHMARK(BM_TensorSize)->Args({WIPE_L1_L2_CACHE, DONT_RUN_WORKLOAD})->UseManualTime()->Iterations(10000);
BENCHMARK(BM_TensorSize)->Args({WIPE_L1_L2_CACHE, RUN_WORKLOAD})->UseManualTime()->Iterations(10000);
// Wipe L1 + L2 + L3
BENCHMARK(BM_TensorSize)->Args({WIPE_L1_L2_L3_CACHE, DONT_RUN_WORKLOAD})->UseManualTime()->Iterations(1000);
BENCHMARK(BM_TensorSize)->Args({WIPE_L1_L2_L3_CACHE, RUN_WORKLOAD})->UseManualTime()->Iterations(1000);

static void BM_VariableSize(benchmark::State& state) {
  int wipe_level = state.range(0);
  bool run_workload = (state.range(1) == RUN_WORKLOAD);
  int64_t res = 0;

  for (auto _ : state) {
    // Workload setup
    std::vector<int64_t> tensor_sizes = {random_in_range(1, 5)};
    auto tmp = torch::empty(tensor_sizes, at::TensorOptions(at::kCPU));

    uint32_t* wipe_buffer = wipe_dcache_setup(wipe_level);
    auto start = std::chrono::high_resolution_clock::now();

    if (run_workload) {
      // Workload
      benchmark::DoNotOptimize(res = tmp.size(0));
    }

    auto end   = std::chrono::high_resolution_clock::now();
    auto elapsed_seconds =
      std::chrono::duration_cast<std::chrono::duration<double>>(
        end - start);
    state.SetIterationTime(elapsed_seconds.count());
    wipe_dcache_teardown(wipe_level, wipe_buffer);
  }
  std::ostream cnull(0);
  cnull << res;
}
// No wipe
BENCHMARK(BM_VariableSize)->Args({WIPE_NO_CACHE, DONT_RUN_WORKLOAD})->UseManualTime()->Iterations(1000000);
BENCHMARK(BM_VariableSize)->Args({WIPE_NO_CACHE, RUN_WORKLOAD})->UseManualTime()->Iterations(1000000);
// Wipe L1
BENCHMARK(BM_VariableSize)->Args({WIPE_L1_CACHE, DONT_RUN_WORKLOAD})->UseManualTime()->Iterations(100000);
BENCHMARK(BM_VariableSize)->Args({WIPE_L1_CACHE, RUN_WORKLOAD})->UseManualTime()->Iterations(100000);
// Wipe L1 + L2
BENCHMARK(BM_VariableSize)->Args({WIPE_L1_L2_CACHE, DONT_RUN_WORKLOAD})->UseManualTime()->Iterations(10000);
BENCHMARK(BM_VariableSize)->Args({WIPE_L1_L2_CACHE, RUN_WORKLOAD})->UseManualTime()->Iterations(10000);
// Wipe L1 + L2 + L3
BENCHMARK(BM_VariableSize)->Args({WIPE_L1_L2_L3_CACHE, DONT_RUN_WORKLOAD})->UseManualTime()->Iterations(1000);
BENCHMARK(BM_VariableSize)->Args({WIPE_L1_L2_L3_CACHE, RUN_WORKLOAD})->UseManualTime()->Iterations(1000);

static void BM_TensorSizes(benchmark::State& state) {
  int wipe_level = state.range(0);
  bool run_workload = (state.range(1) == RUN_WORKLOAD);
  at::IntList res;

  for (auto _ : state) {
    // Workload setup
    std::vector<int64_t> tensor_sizes = {random_in_range(1, 5)};
    auto tmp = at::empty(tensor_sizes, at::TensorOptions(at::kCPU));

    uint32_t* wipe_buffer = wipe_dcache_setup(wipe_level);
    auto start = std::chrono::high_resolution_clock::now();

    if (run_workload) {
      // Workload
      benchmark::DoNotOptimize(res = tmp.sizes());
    }

    auto end   = std::chrono::high_resolution_clock::now();
    auto elapsed_seconds =
      std::chrono::duration_cast<std::chrono::duration<double>>(
        end - start);
    state.SetIterationTime(elapsed_seconds.count());
    wipe_dcache_teardown(wipe_level, wipe_buffer);
  }
  std::ostream cnull(0);
  cnull << res;
}
// No wipe
BENCHMARK(BM_TensorSizes)->Args({WIPE_NO_CACHE, DONT_RUN_WORKLOAD})->UseManualTime()->Iterations(1000000);
BENCHMARK(BM_TensorSizes)->Args({WIPE_NO_CACHE, RUN_WORKLOAD})->UseManualTime()->Iterations(1000000);
// Wipe L1
BENCHMARK(BM_TensorSizes)->Args({WIPE_L1_CACHE, DONT_RUN_WORKLOAD})->UseManualTime()->Iterations(100000);
BENCHMARK(BM_TensorSizes)->Args({WIPE_L1_CACHE, RUN_WORKLOAD})->UseManualTime()->Iterations(100000);
// Wipe L1 + L2
BENCHMARK(BM_TensorSizes)->Args({WIPE_L1_L2_CACHE, DONT_RUN_WORKLOAD})->UseManualTime()->Iterations(10000);
BENCHMARK(BM_TensorSizes)->Args({WIPE_L1_L2_CACHE, RUN_WORKLOAD})->UseManualTime()->Iterations(10000);
// Wipe L1 + L2 + L3
BENCHMARK(BM_TensorSizes)->Args({WIPE_L1_L2_L3_CACHE, DONT_RUN_WORKLOAD})->UseManualTime()->Iterations(1000);
BENCHMARK(BM_TensorSizes)->Args({WIPE_L1_L2_L3_CACHE, RUN_WORKLOAD})->UseManualTime()->Iterations(1000);

static void BM_VariableSizes(benchmark::State& state) {
  int wipe_level = state.range(0);
  bool run_workload = (state.range(1) == RUN_WORKLOAD);
  at::IntList res;

  for (auto _ : state) {
    // Workload setup
    std::vector<int64_t> tensor_sizes = {random_in_range(1, 5)};
    auto tmp = torch::empty(tensor_sizes, at::TensorOptions(at::kCPU));

    uint32_t* wipe_buffer = wipe_dcache_setup(wipe_level);
    auto start = std::chrono::high_resolution_clock::now();

    if (run_workload) {
      // Workload
      benchmark::DoNotOptimize(res = tmp.sizes());
    }

    auto end   = std::chrono::high_resolution_clock::now();
    auto elapsed_seconds =
      std::chrono::duration_cast<std::chrono::duration<double>>(
        end - start);
    state.SetIterationTime(elapsed_seconds.count());
    wipe_dcache_teardown(wipe_level, wipe_buffer);
  }
  std::ostream cnull(0);
  cnull << res;
}
// No wipe
BENCHMARK(BM_VariableSizes)->Args({WIPE_NO_CACHE, DONT_RUN_WORKLOAD})->UseManualTime()->Iterations(1000000);
BENCHMARK(BM_VariableSizes)->Args({WIPE_NO_CACHE, RUN_WORKLOAD})->UseManualTime()->Iterations(1000000);
// Wipe L1
BENCHMARK(BM_VariableSizes)->Args({WIPE_L1_CACHE, DONT_RUN_WORKLOAD})->UseManualTime()->Iterations(100000);
BENCHMARK(BM_VariableSizes)->Args({WIPE_L1_CACHE, RUN_WORKLOAD})->UseManualTime()->Iterations(100000);
// Wipe L1 + L2
BENCHMARK(BM_VariableSizes)->Args({WIPE_L1_L2_CACHE, DONT_RUN_WORKLOAD})->UseManualTime()->Iterations(10000);
BENCHMARK(BM_VariableSizes)->Args({WIPE_L1_L2_CACHE, RUN_WORKLOAD})->UseManualTime()->Iterations(10000);
// Wipe L1 + L2 + L3
BENCHMARK(BM_VariableSizes)->Args({WIPE_L1_L2_L3_CACHE, DONT_RUN_WORKLOAD})->UseManualTime()->Iterations(1000);
BENCHMARK(BM_VariableSizes)->Args({WIPE_L1_L2_L3_CACHE, RUN_WORKLOAD})->UseManualTime()->Iterations(1000);

static void BM_EmptyTensorNoopResize(benchmark::State& state) {
  int wipe_level = state.range(0);
  bool run_workload = (state.range(1) == RUN_WORKLOAD);

  for (auto _ : state) {
    // Workload setup
    std::vector<long int> sizes({0});
    auto tmp = at::empty({0}, at::TensorOptions(at::kCPU));
    tmp.resize_(sizes);

    uint32_t* wipe_buffer = wipe_dcache_setup(wipe_level);
    auto start = std::chrono::high_resolution_clock::now();

    if (run_workload) {
      // Workload
      tmp.resize_(sizes);
    }

    auto end   = std::chrono::high_resolution_clock::now();
    auto elapsed_seconds =
      std::chrono::duration_cast<std::chrono::duration<double>>(
        end - start);
    state.SetIterationTime(elapsed_seconds.count());
    wipe_dcache_teardown(wipe_level, wipe_buffer);
  }
}
// No wipe
BENCHMARK(BM_EmptyTensorNoopResize)->Args({WIPE_NO_CACHE, DONT_RUN_WORKLOAD})->UseManualTime()->Iterations(1000000);
BENCHMARK(BM_EmptyTensorNoopResize)->Args({WIPE_NO_CACHE, RUN_WORKLOAD})->UseManualTime()->Iterations(1000000);
// Wipe L1
BENCHMARK(BM_EmptyTensorNoopResize)->Args({WIPE_L1_CACHE, DONT_RUN_WORKLOAD})->UseManualTime()->Iterations(100000);
BENCHMARK(BM_EmptyTensorNoopResize)->Args({WIPE_L1_CACHE, RUN_WORKLOAD})->UseManualTime()->Iterations(100000);
// Wipe L1 + L2
BENCHMARK(BM_EmptyTensorNoopResize)->Args({WIPE_L1_L2_CACHE, DONT_RUN_WORKLOAD})->UseManualTime()->Iterations(10000);
BENCHMARK(BM_EmptyTensorNoopResize)->Args({WIPE_L1_L2_CACHE, RUN_WORKLOAD})->UseManualTime()->Iterations(10000);
// Wipe L1 + L2 + L3
BENCHMARK(BM_EmptyTensorNoopResize)->Args({WIPE_L1_L2_L3_CACHE, DONT_RUN_WORKLOAD})->UseManualTime()->Iterations(1000);
BENCHMARK(BM_EmptyTensorNoopResize)->Args({WIPE_L1_L2_L3_CACHE, RUN_WORKLOAD})->UseManualTime()->Iterations(1000);

static void BM_EmptyVariableNoopResize(benchmark::State& state) {
  int wipe_level = state.range(0);
  bool run_workload = (state.range(1) == RUN_WORKLOAD);

  for (auto _ : state) {
    // Workload setup
    std::vector<long int> sizes({0});
    auto tmp = torch::empty({0}, at::TensorOptions(at::kCPU));
    tmp.resize_(sizes);

    uint32_t* wipe_buffer = wipe_dcache_setup(wipe_level);
    auto start = std::chrono::high_resolution_clock::now();

    if (run_workload) {
      // Workload
      tmp.resize_(sizes);
    }

    auto end   = std::chrono::high_resolution_clock::now();
    auto elapsed_seconds =
      std::chrono::duration_cast<std::chrono::duration<double>>(
        end - start);
    state.SetIterationTime(elapsed_seconds.count());
    wipe_dcache_teardown(wipe_level, wipe_buffer);
  }
}
// No wipe
BENCHMARK(BM_EmptyVariableNoopResize)->Args({WIPE_NO_CACHE, DONT_RUN_WORKLOAD})->UseManualTime()->Iterations(1000000);
BENCHMARK(BM_EmptyVariableNoopResize)->Args({WIPE_NO_CACHE, RUN_WORKLOAD})->UseManualTime()->Iterations(1000000);
// Wipe L1
BENCHMARK(BM_EmptyVariableNoopResize)->Args({WIPE_L1_CACHE, DONT_RUN_WORKLOAD})->UseManualTime()->Iterations(100000);
BENCHMARK(BM_EmptyVariableNoopResize)->Args({WIPE_L1_CACHE, RUN_WORKLOAD})->UseManualTime()->Iterations(100000);
// Wipe L1 + L2
BENCHMARK(BM_EmptyVariableNoopResize)->Args({WIPE_L1_L2_CACHE, DONT_RUN_WORKLOAD})->UseManualTime()->Iterations(10000);
BENCHMARK(BM_EmptyVariableNoopResize)->Args({WIPE_L1_L2_CACHE, RUN_WORKLOAD})->UseManualTime()->Iterations(10000);
// Wipe L1 + L2 + L3
BENCHMARK(BM_EmptyVariableNoopResize)->Args({WIPE_L1_L2_L3_CACHE, DONT_RUN_WORKLOAD})->UseManualTime()->Iterations(1000);
BENCHMARK(BM_EmptyVariableNoopResize)->Args({WIPE_L1_L2_L3_CACHE, RUN_WORKLOAD})->UseManualTime()->Iterations(1000);

static void BM_TensorNoopResize(benchmark::State& state) {
  int wipe_level = state.range(0);
  bool run_workload = (state.range(1) == RUN_WORKLOAD);

  for (auto _ : state) {
    // Workload setup
    std::vector<long int> sizes({64, 2048});
    auto tmp = at::empty({0}, at::TensorOptions(at::kCPU));
    tmp.resize_(sizes);

    uint32_t* wipe_buffer = wipe_dcache_setup(wipe_level);
    auto start = std::chrono::high_resolution_clock::now();

    if (run_workload) {
      // Workload
      tmp.resize_(sizes);
    }

    auto end   = std::chrono::high_resolution_clock::now();
    auto elapsed_seconds =
      std::chrono::duration_cast<std::chrono::duration<double>>(
        end - start);
    state.SetIterationTime(elapsed_seconds.count());
    wipe_dcache_teardown(wipe_level, wipe_buffer);
  }
}
// No wipe
BENCHMARK(BM_TensorNoopResize)->Args({WIPE_NO_CACHE, DONT_RUN_WORKLOAD})->UseManualTime()->Iterations(1000000);
BENCHMARK(BM_TensorNoopResize)->Args({WIPE_NO_CACHE, RUN_WORKLOAD})->UseManualTime()->Iterations(1000000);
// Wipe L1
BENCHMARK(BM_TensorNoopResize)->Args({WIPE_L1_CACHE, DONT_RUN_WORKLOAD})->UseManualTime()->Iterations(100000);
BENCHMARK(BM_TensorNoopResize)->Args({WIPE_L1_CACHE, RUN_WORKLOAD})->UseManualTime()->Iterations(100000);
// Wipe L1 + L2
BENCHMARK(BM_TensorNoopResize)->Args({WIPE_L1_L2_CACHE, DONT_RUN_WORKLOAD})->UseManualTime()->Iterations(10000);
BENCHMARK(BM_TensorNoopResize)->Args({WIPE_L1_L2_CACHE, RUN_WORKLOAD})->UseManualTime()->Iterations(10000);
// Wipe L1 + L2 + L3
BENCHMARK(BM_TensorNoopResize)->Args({WIPE_L1_L2_L3_CACHE, DONT_RUN_WORKLOAD})->UseManualTime()->Iterations(1000);
BENCHMARK(BM_TensorNoopResize)->Args({WIPE_L1_L2_L3_CACHE, RUN_WORKLOAD})->UseManualTime()->Iterations(1000);

static void BM_VariableNoopResize(benchmark::State& state) {
  int wipe_level = state.range(0);
  bool run_workload = (state.range(1) == RUN_WORKLOAD);

  for (auto _ : state) {
    // Workload setup
    std::vector<long int> sizes({64, 2048});
    auto tmp = torch::empty({0}, at::TensorOptions(at::kCPU));
    tmp.resize_(sizes);

    uint32_t* wipe_buffer = wipe_dcache_setup(wipe_level);
    auto start = std::chrono::high_resolution_clock::now();

    if (run_workload) {
      // Workload
      tmp.resize_(sizes);
    }

    auto end   = std::chrono::high_resolution_clock::now();
    auto elapsed_seconds =
      std::chrono::duration_cast<std::chrono::duration<double>>(
        end - start);
    state.SetIterationTime(elapsed_seconds.count());
    wipe_dcache_teardown(wipe_level, wipe_buffer);
  }
}
// No wipe
BENCHMARK(BM_VariableNoopResize)->Args({WIPE_NO_CACHE, DONT_RUN_WORKLOAD})->UseManualTime()->Iterations(1000000);
BENCHMARK(BM_VariableNoopResize)->Args({WIPE_NO_CACHE, RUN_WORKLOAD})->UseManualTime()->Iterations(1000000);
// Wipe L1
BENCHMARK(BM_VariableNoopResize)->Args({WIPE_L1_CACHE, DONT_RUN_WORKLOAD})->UseManualTime()->Iterations(100000);
BENCHMARK(BM_VariableNoopResize)->Args({WIPE_L1_CACHE, RUN_WORKLOAD})->UseManualTime()->Iterations(100000);
// Wipe L1 + L2
BENCHMARK(BM_VariableNoopResize)->Args({WIPE_L1_L2_CACHE, DONT_RUN_WORKLOAD})->UseManualTime()->Iterations(10000);
BENCHMARK(BM_VariableNoopResize)->Args({WIPE_L1_L2_CACHE, RUN_WORKLOAD})->UseManualTime()->Iterations(10000);
// Wipe L1 + L2 + L3
BENCHMARK(BM_VariableNoopResize)->Args({WIPE_L1_L2_L3_CACHE, DONT_RUN_WORKLOAD})->UseManualTime()->Iterations(1000);
BENCHMARK(BM_VariableNoopResize)->Args({WIPE_L1_L2_L3_CACHE, RUN_WORKLOAD})->UseManualTime()->Iterations(1000);

BENCHMARK_MAIN();

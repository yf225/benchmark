#include <iostream>
#include <benchmark/benchmark.h>
#include <torch/torch.h>
#include <ATen/cuda/CUDAContext.h>

static int iterations = 1;

static void BM_TensorTypeId(benchmark::State& state) {
{
  auto options = at::TensorOptions(at::kCUDA);
  std::vector<long int> sizes({64, 2048});

  // initialize some cuda...
  auto tmp = at::empty({0}, options);

  std::cout << "Dry run started!\n";
  tmp.unsafeGetTensorImpl()->type_id();
  std::cout << "Dry run is done!\n";
}

  auto options = at::TensorOptions(at::kCUDA);
  std::vector<long int> sizes({64, 2048});

  // initialize some cuda...
  auto tmp = at::empty({0}, options);

  for (auto _ : state) {
    tmp.unsafeGetTensorImpl()->type_id();
  }
}
BENCHMARK(BM_TensorTypeId)->Iterations(iterations);

static void BM_TensorType(benchmark::State& state) {
{
  auto options = at::TensorOptions(at::kCUDA);
  std::vector<long int> sizes({64, 2048});

  // initialize some cuda...
  auto tmp = at::empty({0}, options);

  std::cout << "Dry run started!\n";
  tmp.type();
  std::cout << "Dry run is done!\n";
}

  auto options = at::TensorOptions(at::kCUDA);
  std::vector<long int> sizes({64, 2048});

  // initialize some cuda...
  auto tmp = at::empty({0}, options);

  for (auto _ : state) {
    tmp.type();
  }
}
BENCHMARK(BM_TensorType)->Iterations(iterations);

static void BM_THCCachingAllocatorAllocate(benchmark::State& state) {
{
  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  int size = 64 * 2048;
  auto tmp = at::empty({size}, options);
  auto* impl = tmp.unsafeGetTensorImpl();

  // allocate memory once so that caching allocator has it.
  {
    at::DataPtr data = impl->storage().allocator()->allocate(size * 4);
  }

  std::cout << "Dry run started!\n";
  at::DataPtr data = impl->storage().allocator()->allocate(size * 4);
  std::cout << "Dry run is done!\n";
}

  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  int size = 64 * 2048;
  auto tmp = at::empty({size}, options);
  auto* impl = tmp.unsafeGetTensorImpl();

  // allocate memory once so that caching allocator has it.
  {
    at::DataPtr data = impl->storage().allocator()->allocate(size * 4);
  }

  for (auto _ : state) {
    at::DataPtr data = impl->storage().allocator()->allocate(size * 4);
  }
}
BENCHMARK(BM_THCCachingAllocatorAllocate)->Iterations(iterations);



static void BM_TensorIsCuda(benchmark::State& state) {
{
  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = at::empty({0}, options);

  std::cout << "Dry run started!\n";
  tmp.is_cuda();
  std::cout << "Dry run is done!\n";
}

  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = at::empty({0}, options);

  for (auto _ : state) {
    benchmark::DoNotOptimize(tmp.is_cuda());
  }
}
BENCHMARK(BM_TensorIsCuda)->Iterations(iterations);

static void BM_TensorDim(benchmark::State& state) {
{
  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = at::empty({0}, options);

  std::cout << "Dry run started!\n";
  tmp.dim();
  std::cout << "Dry run is done!\n";
}

  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = at::empty({0}, options);

  for (auto _ : state) {
    benchmark::DoNotOptimize(tmp.dim());
  }
}
BENCHMARK(BM_TensorDim)->Iterations(iterations);

static void BM_TensorIsSparse(benchmark::State& state) {
{
  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = at::empty({0}, options);

  std::cout << "Dry run started!\n";
  tmp.is_sparse();
  std::cout << "Dry run is done!\n";
}

  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = at::empty({0}, options);

  for (auto _ : state) {
    benchmark::DoNotOptimize(tmp.is_sparse());
  }
}
BENCHMARK(BM_TensorIsSparse)->Iterations(iterations);

static void BM_TensorTypeIsCuda(benchmark::State& state) {
{
  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = at::empty({0}, options);

  std::cout << "Dry run started!\n";
  tmp.type().is_cuda();
  std::cout << "Dry run is done!\n";
}

  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = at::empty({0}, options);

  for (auto _ : state) {
    benchmark::DoNotOptimize(tmp.type().is_cuda());
  }
}
BENCHMARK(BM_TensorTypeIsCuda)->Iterations(iterations);

static void BM_TensorNumel(benchmark::State& state) {
{
  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = at::empty({0}, options);

  std::cout << "Dry run started!\n";
  tmp.numel();
  std::cout << "Dry run is done!\n";
}

  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = at::empty({0}, options);

  for (auto _ : state) {
    benchmark::DoNotOptimize(tmp.numel());
  }
}
BENCHMARK(BM_TensorNumel)->Iterations(iterations);

static void BM_CudaAPIGetDevice(benchmark::State& state) {
{
  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = at::empty({0}, options);
  int32_t device;

  std::cout << "Dry run started!\n";
  cudaGetDevice(&device);
  std::cout << "Dry run is done!\n";
}

  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = at::empty({0}, options);
  int32_t device;

  for (auto _ : state) {
    benchmark::DoNotOptimize(cudaGetDevice(&device));
  }
}
BENCHMARK(BM_CudaAPIGetDevice)->Iterations(iterations);

static void BM_CudaAPISetDevice(benchmark::State& state) {
{
  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = at::empty({0}, options);
  int32_t device;
  cudaGetDevice(&device);

  std::cout << "Dry run started!\n";
  cudaSetDevice(device);
  std::cout << "Dry run is done!\n";
}

  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = at::empty({0}, options);
  int32_t device;
  cudaGetDevice(&device);

  for (auto _ : state) {
    benchmark::DoNotOptimize(cudaSetDevice(device));
  }
}
BENCHMARK(BM_CudaAPISetDevice)->Iterations(iterations);

static void BM_DynamicCUDAInterfaceGetDevice(benchmark::State& state) {
{
  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = at::empty({0}, options);
  int32_t device;

  std::cout << "Dry run started!\n";
  at::detail::DynamicCUDAInterface::get_device(&device);
  std::cout << "Dry run is done!\n";
}

  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = at::empty({0}, options);
  int32_t device;

  for (auto _ : state) {
    at::detail::DynamicCUDAInterface::get_device(&device);
  }
}
BENCHMARK(BM_DynamicCUDAInterfaceGetDevice)->Iterations(iterations);

static void BM_DynamicCUDAInterfaceSetDevice(benchmark::State& state) {
{
  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = at::empty({0}, options);
  int32_t device;
  cudaGetDevice(&device);

  std::cout << "Dry run started!\n";
  at::detail::DynamicCUDAInterface::set_device(device);
  std::cout << "Dry run is done!\n";
}

  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = at::empty({0}, options);
  int32_t device;
  cudaGetDevice(&device);

  for (auto _ : state) {
    at::detail::DynamicCUDAInterface::set_device(device);
  }
}
BENCHMARK(BM_DynamicCUDAInterfaceSetDevice)->Iterations(iterations);

static void BM_StorageImplGetDevice(benchmark::State& state) {
{
  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = at::empty({0}, options);
  auto* storage_impl = tmp.unsafeGetTensorImpl()->storage().unsafeGetStorageImpl();

  std::cout << "Dry run started!\n";
  storage_impl->device().index();
  std::cout << "Dry run is done!\n";
}

  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = at::empty({0}, options);
  auto* storage_impl = tmp.unsafeGetTensorImpl()->storage().unsafeGetStorageImpl();

  for (auto _ : state) {
    benchmark::DoNotOptimize(storage_impl->device().index());
  }
}
BENCHMARK(BM_StorageImplGetDevice)->Iterations(iterations);

static void BM_TensorImplGetDevice(benchmark::State& state) {
{
  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = at::empty({0}, options);
  auto* tensor_impl = tmp.unsafeGetTensorImpl();

  std::cout << "Dry run started!\n";
  tensor_impl->storage().unsafeGetStorageImpl()->device().index();
  std::cout << "Dry run is done!\n";
}

  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = at::empty({0}, options);
  auto* tensor_impl = tmp.unsafeGetTensorImpl();

  for (auto _ : state) {
    benchmark::DoNotOptimize(
        tensor_impl->storage().unsafeGetStorageImpl()->device().index());
  }
}
BENCHMARK(BM_TensorImplGetDevice)->Iterations(iterations);

static void BM_TensorGetDeviceDirect(benchmark::State& state) {
{
  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = at::empty({0}, options);

  std::cout << "Dry run started!\n";
  tmp.unsafeGetTensorImpl()->storage().unsafeGetStorageImpl()->device().index();
  std::cout << "Dry run is done!\n";
}

  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = at::empty({0}, options);

  for (auto _ : state) {
    benchmark::DoNotOptimize(
        tmp.unsafeGetTensorImpl()->storage().unsafeGetStorageImpl()->device().index());
  }
}
BENCHMARK(BM_TensorGetDeviceDirect)->Iterations(iterations);


//static void BM_THGetDevice(benchmark::State& state) {
//  auto options = at::TensorOptions(at::kCUDA);
//
//  // initialize some cuda...
//  auto tmp = at::empty({0}, options);
//
//  for (auto _ : state) {
//    benchmark::DoNotOptimize(at::_th_get_device(tmp));
//  }
//
//}
//BENCHMARK(BM_THGetDevice)->Iterations(iterations);

static void BM_TensorGetDevice(benchmark::State& state) {
{
  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = at::empty({0}, options);

  std::cout << "Dry run started!\n";
  tmp.get_device();
  std::cout << "Dry run is done!\n";
}

  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = at::empty({0}, options);

  for (auto _ : state) {
    benchmark::DoNotOptimize(tmp.get_device());
  }
}
BENCHMARK(BM_TensorGetDevice)->Iterations(iterations);

static void BM_DeviceGuardCtor(benchmark::State& state) {
{
  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = at::empty({0}, options);
  void* mem = malloc(sizeof(at::DeviceGuard));

  new (mem) at::DeviceGuard(tmp);

  std::cout << "Dry run started!\n";
  free(mem);
  std::cout << "Dry run is done!\n";
}

  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = at::empty({0}, options);
  void* mem = malloc(sizeof(at::DeviceGuard));

  for (auto _ : state) {
    benchmark::DoNotOptimize(new (mem) at::DeviceGuard(tmp));
  }

  free(mem);
}
BENCHMARK(BM_DeviceGuardCtor)->Iterations(iterations);

static void BM_DeviceGuard(benchmark::State& state) {
{
  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = at::empty({0}, options);

  std::cout << "Dry run started!\n";
  {
    const at::DeviceGuard guard(tmp);
  }
  std::cout << "Dry run is done!\n";
}

  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = at::empty({0}, options);

  for (auto _ : state) {
    {
      const at::DeviceGuard guard(tmp);
    }
  }
}
BENCHMARK(BM_DeviceGuard)->Iterations(iterations);

static void BM_EmptyTensorNoopResize(benchmark::State& state) {
{
  auto options = at::TensorOptions(at::kCUDA);
  std::vector<long int> sizes({0});

  // initialize some cuda...
  auto tmp = at::empty({0}, options);
  tmp.resize_(sizes);
  
  std::cout << "Dry run started!\n";
  tmp.resize_(sizes);
  std::cout << "Dry run is done!\n";
}

  auto options = at::TensorOptions(at::kCUDA);
  std::vector<long int> sizes({0});

  // initialize some cuda...
  auto tmp = at::empty({0}, options);
  tmp.resize_(sizes);

  for (auto _ : state) {
    tmp.resize_(sizes);
  }
}
BENCHMARK(BM_EmptyTensorNoopResize)->Iterations(iterations);

//static void BM_NoopEmptyResizeNoDispatch(benchmark::State& state) {
//  auto options = at::TensorOptions(at::kCUDA);
//  std::vector<long int> sizes({0});
//
//  // initialize some cuda...
//  auto tmp = at::empty({0}, options);
//  tmp.resize_(sizes);
//
//  for (auto _ : state) {
//    at::native::resize__cuda(tmp, sizes);
//  }
//}
//BENCHMARK(BM_NoopEmptyResizeNoDispatch)->Iterations(iterations);

static void BM_TensorNoopResize(benchmark::State& state) {
{
  auto options = at::TensorOptions(at::kCUDA);
  std::vector<long int> sizes({64, 2048});

  // initialize some cuda...
  auto tmp = at::empty({0}, options);
  tmp.resize_(sizes);

  std::cout << "Dry run started!\n";
  tmp.resize_(sizes);
  std::cout << "Dry run is done!\n";
}

  auto options = at::TensorOptions(at::kCUDA);
  std::vector<long int> sizes({64, 2048});

  // initialize some cuda...
  auto tmp = at::empty({0}, options);
  tmp.resize_(sizes);

  for (auto _ : state) {
    tmp.resize_(sizes);
  }
}
BENCHMARK(BM_TensorNoopResize)->Iterations(iterations);

static void BM_TensorAsStrided(benchmark::State& state) {
{
  auto tensor = at::rand({2400});
  std::vector<long int> strides({1, 300});
  std::vector<long int> sizes({300, 8});

  std::cout << "Dry run started!\n";
  tensor.as_strided(strides, sizes);
  std::cout << "Dry run is done!\n";
}

  auto tensor = at::rand({2400});
  std::vector<long int> strides({1, 300});
  std::vector<long int> sizes({300, 8});

  for (auto _ : state)
    benchmark::DoNotOptimize(tensor.as_strided(strides, sizes));
}
BENCHMARK(BM_TensorAsStrided)->Iterations(iterations);

static void BM_AtenEmptyCuda(benchmark::State& state) {
{
  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = at::empty({0}, options);

  std::cout << "Dry run started!\n";
  auto tensor = at::native::empty_cuda({0}, options);
  std::cout << "Dry run is done!\n";
}

  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = at::empty({0}, options);

  for (auto _ : state) {
    auto tensor = at::native::empty_cuda({0}, options);
  }
}
BENCHMARK(BM_AtenEmptyCuda)->Iterations(iterations);

static void BM_AtenEmpty(benchmark::State& state) {
{
  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = at::empty({0}, options);

  std::cout << "Dry run started!\n";
  auto tensor = at::empty({0}, options);
  std::cout << "Dry run is done!\n";
}

  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = at::empty({0}, options);

  for (auto _ : state) {
    auto tensor = at::empty({0}, options);
  }
}
BENCHMARK(BM_AtenEmpty)->Iterations(iterations);

static void BM_VariableEmpty(benchmark::State& state) {
{
  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = torch::empty({0}, options);

  std::cout << "Dry run started!\n";
  auto tensor = torch::empty({0}, options);
  std::cout << "Dry run is done!\n";
}

  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = torch::empty({0}, options);

  for (auto _ : state) {
    auto tensor = torch::empty({0}, options);
  }
}
BENCHMARK(BM_VariableEmpty)->Iterations(iterations);

static void BM_AtenEmptyResize(benchmark::State& state) {
{
  auto options = at::TensorOptions(at::kCUDA);
  std::vector<long int> sizes({64, 2048});

  // initialize some cuda...
  auto tmp = at::empty({0}, options);
  tmp.resize_(sizes);

  std::cout << "Dry run started!\n";
  auto tensor = at::empty({0}, options);
  tensor.resize_(sizes);
  std::cout << "Dry run is done!\n";
}
  auto options = at::TensorOptions(at::kCUDA);
  std::vector<long int> sizes({64, 2048});

  // initialize some cuda...
  auto tmp = at::empty({0}, options);
  tmp.resize_(sizes);

  for (auto _ : state) {
    auto tensor = at::empty({0}, options);
    tensor.resize_(sizes);
  }
}
BENCHMARK(BM_AtenEmptyResize)->Iterations(iterations);

static void BM_AtenEmptyNoResize(benchmark::State& state) {
{
  auto options = at::TensorOptions(at::kCUDA);
  std::vector<long int> sizes({64, 2048});

  // initialize some cuda...
  auto tmp = at::empty({0}, options);
  tmp.resize_(sizes);

  std::cout << "Dry run started!\n";
  auto tensor = at::empty(sizes, options);
  std::cout << "Dry run is done!\n";
}

  auto options = at::TensorOptions(at::kCUDA);
  std::vector<long int> sizes({64, 2048});

  // initialize some cuda...
  auto tmp = at::empty({0}, options);
  tmp.resize_(sizes);

  for (auto _ : state) {
    auto tensor = at::empty(sizes, options);
  }
}
BENCHMARK(BM_AtenEmptyNoResize)->Iterations(iterations);


static void BM_VariableEmptyResize(benchmark::State& state) {
{
  auto options = at::TensorOptions(at::kCUDA);
  std::vector<long int> sizes({64, 2048});
  std::vector<long int> zero({0});

  // initialize some cuda...
  auto tmp = torch::empty(zero, options);
  tmp.resize_(sizes);

  std::cout << "Dry run started!\n";
  auto tensor = torch::empty(zero, options);
  tensor.resize_(sizes);
  std::cout << "Dry run is done!\n";
}

  auto options = at::TensorOptions(at::kCUDA);
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
BENCHMARK(BM_VariableEmptyResize)->Iterations(iterations);

static void BM_VariableEmptyNoResize(benchmark::State& state) {
{
  auto options = at::TensorOptions(at::kCUDA);
  std::vector<long int> sizes({64, 2048});
  std::vector<long int> zero({0});

  // initialize some cuda...
  auto tmp = torch::empty(zero, options);
  tmp.resize_(sizes);

  std::cout << "Dry run started!\n";
  torch::empty(sizes, options);
  std::cout << "Dry run is done!\n";
}

  auto options = at::TensorOptions(at::kCUDA);
  std::vector<long int> sizes({64, 2048});
  std::vector<long int> zero({0});

  // initialize some cuda...
  auto tmp = torch::empty(zero, options);
  tmp.resize_(sizes);

  for (auto _ : state) {
    benchmark::DoNotOptimize(torch::empty(sizes, options));
  }
}
BENCHMARK(BM_VariableEmptyNoResize)->Iterations(iterations);


static void BM_MakeStorage(benchmark::State& state) {
{
  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = torch::empty({0}, options);

  std::cout << "Dry run started!\n";
  c10::make_intrusive<at::StorageImpl>(
      at::scalarTypeToTypeMeta(options.dtype()),
      0,
      at::cuda::getCUDADeviceAllocator(),
      true);
  std::cout << "Dry run is done!\n";
}

  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = torch::empty({0}, options);

  for (auto _ : state) {
    benchmark::DoNotOptimize(
        c10::make_intrusive<at::StorageImpl>(
            at::scalarTypeToTypeMeta(options.dtype()),
            0,
            at::cuda::getCUDADeviceAllocator(),
            true));
  }
}
BENCHMARK(BM_MakeStorage)->Iterations(iterations);

static void BM_StorageCtor(benchmark::State& state) {
{
  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = torch::empty({0}, options);

  void* mem = malloc(sizeof(at::StorageImpl));

  std::cout << "Dry run started!\n";
  new (mem) at::StorageImpl(
      at::scalarTypeToTypeMeta(options.dtype()),
      0,
      at::cuda::getCUDADeviceAllocator(),
      true);
  std::cout << "Dry run is done!\n";

  free(mem);
}

  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = torch::empty({0}, options);

  void* mem = malloc(sizeof(at::StorageImpl));

  for (auto _ : state) {
    benchmark::DoNotOptimize(
        new (mem) at::StorageImpl(
            at::scalarTypeToTypeMeta(options.dtype()),
            0,
            at::cuda::getCUDADeviceAllocator(),
            true));
  }

  free(mem);
}
BENCHMARK(BM_StorageCtor)->Iterations(iterations);

static void BM_MallocOverhead(benchmark::State& state) {
  for (auto _ : state) {
    benchmark::DoNotOptimize(malloc(1));
  }
}
BENCHMARK(BM_MallocOverhead)->Iterations(iterations);

static void BM_StorageMalloc(benchmark::State& state) {
  for (auto _ : state) {
    // NB: leaks memory
    benchmark::DoNotOptimize(malloc(sizeof(at::StorageImpl)));
  }
}
BENCHMARK(BM_StorageMalloc)->Iterations(iterations);

static void BM_ScalarTypeToTypeMeta(benchmark::State& state) {
{
  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = torch::empty({0}, options);

  std::cout << "Dry run started!\n";
  at::scalarTypeToTypeMeta(options.dtype());
  std::cout << "Dry run is done!\n";
}

  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = torch::empty({0}, options);

  for (auto _ : state) {
    benchmark::DoNotOptimize(
        at::scalarTypeToTypeMeta(options.dtype()));
  }
}
BENCHMARK(BM_ScalarTypeToTypeMeta)->Iterations(iterations);

static void BM_MakeTensorFromStorage(benchmark::State& state) {
{
  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = torch::empty({0}, options);

  auto storage = c10::make_intrusive<at::StorageImpl>(
            at::scalarTypeToTypeMeta(options.dtype()),
            0,
            at::cuda::getCUDADeviceAllocator(),
            true);

  std::cout << "Dry run started!\n";
  at::detail::make_tensor<at::TensorImpl>(storage, at::CUDATensorId(), false);
  std::cout << "Dry run is done!\n";
}

  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = torch::empty({0}, options);

  auto storage = c10::make_intrusive<at::StorageImpl>(
            at::scalarTypeToTypeMeta(options.dtype()),
            0,
            at::cuda::getCUDADeviceAllocator(),
            true);

  for (auto _ : state) {
    benchmark::DoNotOptimize(
        at::detail::make_tensor<at::TensorImpl>(storage, at::CUDATensorId(), false));
  }
}
BENCHMARK(BM_MakeTensorFromStorage)->Iterations(iterations);

static void BM_MakeVariableFromTensor(benchmark::State& state) {
{
  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = torch::empty({0}, options);

  auto storage_impl = c10::make_intrusive<at::StorageImpl>(
            at::scalarTypeToTypeMeta(options.dtype()),
            0,
            at::cuda::getCUDADeviceAllocator(),
            true);
  auto tensor = at::detail::make_tensor<at::TensorImpl>(
      storage_impl, at::CUDATensorId(), false);

  std::cout << "Dry run started!\n";
  torch::autograd::make_variable(tensor, false);
  std::cout << "Dry run is done!\n";
}

  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = torch::empty({0}, options);

  auto storage_impl = c10::make_intrusive<at::StorageImpl>(
            at::scalarTypeToTypeMeta(options.dtype()),
            0,
            at::cuda::getCUDADeviceAllocator(),
            true);
  auto tensor = at::detail::make_tensor<at::TensorImpl>(
      storage_impl, at::CUDATensorId(), false);

  for (auto _ : state) {
    benchmark::DoNotOptimize(
        torch::autograd::make_variable(tensor, false));
  }
}
BENCHMARK(BM_MakeVariableFromTensor)->Iterations(iterations);



static void BM_CheckedTensorUnwrap(benchmark::State& state) {
{
  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = at::empty({0}, options);

  std::cout << "Dry run started!\n";
  at::checked_tensor_unwrap(tmp,"self",1, false, at::Backend::CUDA, at::ScalarType::Float);
  std::cout << "Dry run is done!\n";
}

  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = at::empty({0}, options);

  for (auto _ : state) {
    benchmark::DoNotOptimize(
        at::checked_tensor_unwrap(tmp,"self",1, false, at::Backend::CUDA, at::ScalarType::Float));
  }
}
BENCHMARK(BM_CheckedTensorUnwrap)->Iterations(iterations);

BENCHMARK_MAIN();

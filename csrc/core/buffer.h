#pragma once

#ifdef USE_ROCM
#include "core/hip_defs.h"
#include "core/hip_cuda_dtype_defs.h"
#else
#include <cuda_runtime.h>
#endif

#include "core/cuda_utils.h"

namespace pplx {

template <typename T> class HostBuffer;
template <typename T> class DeviceBuffer;

/// Handle to a host-allocated buffer.
template <typename T> class HostBuffer final {
public:
  HostBuffer(size_t size)
      : size_(size) {
    CUDACHECK(cudaMallocHost(  (void **) &data_, size * sizeof(T)));
  }

  HostBuffer(const DeviceBuffer<T> &device_buffer);

  HostBuffer(HostBuffer &&other)
      : size_(other.size_) {
    std::swap(data_, other.data_);
  }

  HostBuffer(const HostBuffer &) = delete;

  ~HostBuffer() {
    cudaFreeHost(data_);
    data_ = nullptr;
  }

  HostBuffer &operator=(const HostBuffer &) = delete;

  HostBuffer &operator=(HostBuffer &&other) {
    std::swap(size_, other.size_);
    std::swap(data_, other.data_);
    return *this;
  }

  size_t size() const { return size_; }
  T &operator[](size_t i) { return data_[i]; }
  const T &operator[](size_t i) const { return data_[i]; }

  const T *get() const { return data_; }

  void copyFromDevice(const DeviceBuffer<T> &device_buffer);

private:
  size_t size_;
  T *data_ = nullptr;
};

/// Handle to a device-allocated buffer.
template <typename T> class DeviceBuffer final {
public:
  DeviceBuffer(size_t size)
      : size_(size) {
    CUDACHECK(cudaMalloc(&data_, size * sizeof(T)));
  }

  DeviceBuffer(const HostBuffer<T> &host_buffer);

  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer(DeviceBuffer &&other) = delete;

  ~DeviceBuffer() { cudaFree(data_); }

  DeviceBuffer &operator=(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(DeviceBuffer &&other) = delete;

  size_t size() const { return size_; }
  T *operator&() { return data_; }

  const T *get() const { return data_; }
  T *get() { return data_; }

  void copyFromHost(const HostBuffer<T> &host_buffer);

  void copyFromHost(const T *host_data, size_t num_elements) {
    CUDACHECK(cudaMemcpy(data_, host_data, num_elements * sizeof(T), cudaMemcpyHostToDevice));
  }

private:
  size_t size_;
  T *data_;
};

template <typename T>
HostBuffer<T>::HostBuffer(const DeviceBuffer<T> &device_buffer)
    : size_(device_buffer.size()) {
  CUDACHECK(cudaMallocHost( (void**) &data_, size_ * sizeof(T)));
  CUDACHECK(cudaMemcpy(data_, device_buffer.get(), size_ * sizeof(T), cudaMemcpyDeviceToHost));
}

template <typename T>
DeviceBuffer<T>::DeviceBuffer(const HostBuffer<T> &host_buffer)
    : size_(host_buffer.size()) {
  CUDACHECK(cudaMalloc(&data_, size_ * sizeof(T)));
  CUDACHECK(cudaMemcpy(data_, host_buffer.get(), size_ * sizeof(T), cudaMemcpyHostToDevice));
}

template <typename T> void HostBuffer<T>::copyFromDevice(const DeviceBuffer<T> &device_buffer) {
  CUDACHECK(cudaMemcpy(data_, device_buffer.get(), size_ * sizeof(T), cudaMemcpyDeviceToHost));
}

template <typename T> void DeviceBuffer<T>::copyFromHost(const HostBuffer<T> &host_buffer) {
  CUDACHECK(cudaMemcpy(data_, host_buffer.get(), size_ * sizeof(T), cudaMemcpyHostToDevice));
}

template <typename T> struct Strided1D {
  T *data;
  size_t strideElem;

  template <typename S>
  Strided1D(DeviceBuffer<S> &data, size_t strideElem)
      : data(reinterpret_cast<T *>(data.get())),
        strideElem(strideElem) {}

  Strided1D(T *data, size_t strideElem)
      : data(data),
        strideElem(strideElem) {}
};

template <typename T> struct Strided2D {
  T *data;
  size_t strideElem;
  size_t strideRow;

  template <typename S>
  Strided2D(DeviceBuffer<S> &data, size_t strideElem, size_t strideRow)
      : data(reinterpret_cast<T *>(data.get())),
        strideElem(strideElem),
        strideRow(strideRow) {}

  Strided2D(T *data, size_t strideElem, size_t strideRow)
      : data(data),
        strideElem(strideElem),
        strideRow(strideRow) {}
};

template <typename T> struct Strided3D {
  T *data;
  size_t strideElem;
  size_t strideRow;
  size_t strideCol;

  template <typename S>
  Strided3D(DeviceBuffer<S> &data, size_t strideElem, size_t strideRow, size_t strideCol)
      : data(reinterpret_cast<T *>(data.get())),
        strideElem(strideElem),
        strideRow(strideRow),
        strideCol(strideCol) {}

  Strided3D(T *data, size_t strideElem, size_t strideRow, size_t strideCol)
      : data(data),
        strideElem(strideElem),
        strideRow(strideRow),
        strideCol(strideCol) {}
};

} // namespace pplx

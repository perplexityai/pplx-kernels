# Install Driver and Dependencies

Here's a summary of the software and drivers required for running pplx-kernels on a single-node or multi-node cluster with Mellanox ConnectX or AWS Elastic Fabric Adapter (EFA) network interfaces. Configure your system and software accordingly.

| Software                  | Single-node | Multi-node with ConnectX | Multi-node with EFA |
|---------------------------|-------------|--------------------------|---------------------|
| NVIDIA Driver             | Y           | Y                        | Y                   |
| modprobe.d/nvidia.conf    |             | Y                        |                     |
| CUDA Toolkit              | Y           | Y                        | Y                   |
| RDMA-Core                 |             | Y                        | Y                   |
| UCX Library               |             | Y                        | Y                   |
| NIXL Library              |             | Y                        | Y                   |
| DOCA (GPUNetIO)           |             | Y                        |                     |
| GDRCopy Driver            |             | Y                        | Y                   |
| GDRCopy Library           |             | Y                        | Y                   |
| NVSHMEM Library           | Y           | Y                        | Y                   |
| NVSHMEM_USE_GDRCOPY       |             | 1                        | 1                   |
| NVSHMEM_IBRC_SUPPORT      |             | 1                        |                     |
| NVSHMEM_IBGDA_SUPPORT     |             | 1                        |                     |
| NVSHMEM_LIBFABRIC_SUPPORT |             |                          | 1                   |
| Libfabric Library         |             |                          | Y                   |
| EFA Driver                |             |                          | Y                   |

## NVIDIA Driver Config

### NIXL + UCX + DOCA Configuration

For ConnectX systems, internode communication can use NIXL + UCX with DOCA GPUNetIO. The following sections detail the installation and configuration.

To use IBGDA, NVIDIA Driver needs to be configured to allow GPU to initiate communication.

```bash
echo 'options nvidia NVreg_EnableStreamMemOPs=1 NVreg_RegistryDwords="PeerMappingOverride=1;"' | sudo tee -a /etc/modprobe.d/nvidia.conf
sudo update-initramfs -u
sudo reboot
```

#### System Packages for NIXL/UCX/DOCA

Install common build dependencies (Ubuntu 24.04 example):

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential cmake gcc g++ ninja-build pkg-config \
  autoconf automake libtool git valgrind \
  linux-headers-$(uname -r) \
  meson python3-pip python3-dev python3-docutils \
  libnl-3-dev libnl-route-3-dev \
  libglib2.0-dev libssl-dev libzip-dev libjson-c-dev \
  libpcap-dev libjsoncpp-dev curl
```

#### RDMA-Core

```bash
git clone https://github.com/linux-rdma/rdma-core.git
cd rdma-core
git checkout v58.0
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/opt/rdma
make -j"$(nproc)"
sudo make install
```

Set `RDMA_HOME=/opt/rdma`.

#### DOCA

Install DOCA with GPUNetIO and set `DOCA_HOME` to its install prefix. Refer to NVIDIA DOCA SDK documentation for obtaining packages and prerequisites.

Example install layout expectation:
- Headers at `$DOCA_HOME/include`
- Libraries at `$DOCA_HOME/lib/x86_64-linux-gnu`

#### UCX

Build UCX with CUDA, GDRCopy, and DOCA support.

```bash
git clone https://github.com/openucx/ucx.git
cd ucx
./autogen.sh
./contrib/configure-release-mt --prefix=/opt/ucx \
  --with-verbs=/opt/rdma \
  --with-cuda=/usr/local/cuda \
  --with-gdrcopy=/opt/gdrcopy \
  --with-doca=${DOCA_HOME:-no}
make -j"$(nproc)"
sudo make install
```

Set `UCX_HOME=/opt/ucx`.

#### NIXL

Follow the build instructions in the NIXL repository: [NIXL repository](https://github.com/ai-dynamo/nixl). Example:

```bash
pip3 install --upgrade meson pybind11
git clone https://github.com/ai-dynamo/nixl.git
cd nixl
meson setup build -Ducx_path=/opt/ucx -Dprefix=/opt/nixl --buildtype=release
sudo ninja install -C build
```

Set `NIXL_HOME=/opt/nixl`.

#### Runtime Environment

Set up environment variables (adjust paths to your installation):

```bash
export CUDA_HOME=/usr/local/cuda
export RDMA_HOME=/opt/rdma
export DOCA_HOME=/opt/doca
export UCX_HOME=/opt/ucx
export NIXL_HOME=/opt/nixl

export LD_LIBRARY_PATH="$UCX_HOME/lib:$NIXL_HOME/lib:$CUDA_HOME/lib64:${DOCA_HOME:+$DOCA_HOME/lib/x86_64-linux-gnu:}$LD_LIBRARY_PATH"
export LD_PRELOAD="${DOCA_HOME:+$DOCA_HOME/lib/x86_64-linux-gnu/libdoca_common.so:$DOCA_HOME/lib/x86_64-linux-gnu/libdoca_gpunetio.so:$DOCA_HOME/lib/x86_64-linux-gnu/libdoca_verbs.so}"
```

#### UCX Network Device Selection

On some systems UCX may not automatically pick the desired NIC per GPU. Explicitly set `UCX_NET_DEVICES` to map each GPU to the correct NIC and port.

Shell example (single-node, 1:1 GPU-to-NIC mapping):

```bash
# Adjust NIC names (mlx5_X) and port (:1) as needed
export UCX_NET_DEVICES="cuda${LOCAL_RANK}-mlx5_${LOCAL_RANK}:1"
```

Python example (custom mapping list):

```python
import os
local_rank = int(os.environ.get("LOCAL_RANK", "0"))
pxb_nics = ["mlx5_0", "mlx5_3", "mlx5_4", "mlx5_5", "mlx5_6", "mlx5_9", "mlx5_10", "mlx5_11"]
os.environ['UCX_NET_DEVICES'] = f'cuda{local_rank}-{pxb_nics[local_rank]}:1'
```

Project example (as used in `tests/bench_all_to_all.py`):

```python
import os
# Map each GPU to a specific mlx5 device and include additional NICs
pxb_nics_eos = ["mlx5_0", "mlx5_3", "mlx5_4", "mlx5_5", "mlx5_6", "mlx5_9", "mlx5_10", "mlx5_11"]
tcp_nics_eos = ',ibp154s0,ibp192s0,ibp206s0,ibp220s0,ibp94s0'
local_rank = int(os.environ.get("LOCAL_RANK", "0"))
os.environ['UCX_NET_DEVICES'] = f'cuda{local_rank}-{pxb_nics_eos[local_rank]}:1' + tcp_nics_eos
```

Adjust device names to match your system (`mlx5_*`, `ibp*`, and the `:1` port).

This configuration aligns pplx-kernels internode usage with NIXL + UCX. Adjust paths and options to match your deployment.

## GDRCopy

GDRCopy is needed for multi-node.

```bash
sudo apt-get install -y build-essential devscripts debhelper fakeroot pkg-config dkms
wget -O gdrcopy-v2.4.4.tar.gz https://github.com/NVIDIA/gdrcopy/archive/refs/tags/v2.4.4.tar.gz
tar xf gdrcopy-v2.4.4.tar.gz
cd gdrcopy-2.4.4/
sudo make prefix=/opt/gdrcopy -j$(nproc) install

cd packages/
CUDA=/usr/local/cuda ./build-deb-packages.sh
sudo dpkg -i gdrdrv-dkms_2.4.4_amd64.Ubuntu22_04.deb \
             gdrcopy-tests_2.4.4_amd64.Ubuntu22_04+cuda12.6.deb \
             gdrcopy_2.4.4_amd64.Ubuntu22_04.deb \
             libgdrapi_2.4.4_amd64.Ubuntu22_04.deb
```

Verify installation:

```bash
/opt/gdrcopy/bin/gdrcopy_copybw
```

## NVSHMEM

There are many configurations for NVSHMEM.
Besides the required configurations listed on the top of this page, here are some additional optional features:

* NVSHMEM_MPI_SUPPORT: For MPI support
* NVSHMEM_PMIX_SUPPORT: For PMIx support (e.g., slurm)
* NVSHMEM_BUILD_HYDRA_LAUNCHER: For Hydra launcher

Change the following options accordingly.

```bash
wget https://developer.nvidia.com/downloads/assets/secure/nvshmem/nvshmem_src_3.2.5-1.txz
mkdir nvshmem_src_3.2.5-1
tar xf nvshmem_src_3.2.5-1.txz -C nvshmem_src_3.2.5-1
cd nvshmem_src_3.2.5-1/nvshmem_src
mkdir -p build
cd build
cmake \
    -DNVSHMEM_PREFIX=/opt/nvshmem-3.2.5 \
    -DCMAKE_CUDA_ARCHITECTURES=90a \
    -DNVSHMEM_MPI_SUPPORT=1 \
    -DNVSHMEM_PMIX_SUPPORT=1 \
    -DNVSHMEM_LIBFABRIC_SUPPORT=1 \
    -DNVSHMEM_IBRC_SUPPORT=1 \
    -DNVSHMEM_IBGDA_SUPPORT=1 \
    -DNVSHMEM_BUILD_TESTS=1 \
    -DNVSHMEM_BUILD_EXAMPLES=1 \
    -DNVSHMEM_BUILD_HYDRA_LAUNCHER=1 \
    -DNVSHMEM_BUILD_TXZ_PACKAGE=1 \
    -DMPI_HOME=/opt/amazon/openmpi \
    -DPMIX_HOME=/opt/amazon/pmix \
    -DGDRCOPY_HOME=/opt/gdrcopy \
    -DLIBFABRIC_HOME=/opt/amazon/efa \
    -G Ninja \
    ..
ninja build
sudo ninja install
```

After installation, add the following environment variables:

```bash
export NVSHMEM_HOME=/opt/nvshmem-3.2.5
export LD_LIBRARY_PATH=$NVSHMEM_HOME/lib:$LD_LIBRARY_PATH

# For single-node
export NVSHMEM_REMOTE_TRANSPORT=none

# For multi-node with ConnectX
export NVSHMEM_REMOTE_TRANSPORT=ibrc
export NVSHMEM_IB_ENABLE_IBGDA=1

# For multi-node with EFA
export NVSHMEM_REMOTE_TRANSPORT=libfabric
export NVSHMEM_LIBFABRIC_PROVIDER=efa
```

To install Hydra launcher:

```bash
cd nvshmem_src_3.2.5-1/nvshmem_src/
sed -i 's/^make/make -j/' scripts/install_hydra.sh
sudo bash scripts/install_hydra.sh hydra-build /opt/hydra
```

Verify installation:

```bash
# Using Hydra:
/opt/hydra/bin/nvshmrun.hydra -genvlist LD_LIBRARY_PATH -hosts host1,host2 -n 2 -ppn 1 /opt/nvshmem-3.2.5/bin/perftest/device/pt-to-pt/shmem_put_latency

# Using MPI:
NVSHMEM_BOOTSTRAP=MPI mpirun -x LD_LIBRARY_PATH -x NVSHMEM_BOOTSTRAP -H host1,host2 /opt/nvshmem-3.2.5/bin/perftest/device/pt-to-pt/shmem_put_latency
```

# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Automated release workflow with GitHub Actions
- Semantic versioning support
- This CHANGELOG.md file

### Changed
- Repository now supports stable tags and releases

## [0.1.0] - 2025-10-19

### Added
- NVSHMEM4Py Integration - replaced custom python bindings with official Python language binding for NVSHMEM
- Cuda Graph support
- Flexible transportation layers: NVLink, IBGDA, IBRC, EFA
- Overlapping communication and computation
- Comprehensive testing and benchmarking suite
- C++ testing support with CMake
- Multi-node testing and benchmarking capabilities

### Changed
- Migrated from static linking of host-side initialization (libnvshmem.a) to dynamic linking (libnvshmem_host.so)
- Updated README.md with comprehensive installation and usage instructions
- Improved project structure with proper Python packaging

### Fixed
- Fixed potential bugs in internode dispatch kernel (shared memory allocation and synchronization)
- Fixed build linkage errors
- Fixed NVSHMEM external variable instances issues
- Memory leaks in Distributed::allToAllImpl (freed srcBuffer and dstBuffer)
- Added device guard for kernel calls

### Security
- Proper linking of NVSHMEM to sublibraries as interface to prevent multiple instances

---

## Version History Overview

This project follows [Semantic Versioning](https://semver.org/):
- **MAJOR** version for incompatible API changes
- **MINOR** version for backwards-compatible functionality additions  
- **PATCH** version for backwards-compatible bug fixes

For consumers of pplx-kernels, you can now use version tags instead of pinning to specific SHAs:

```bash
# Install specific version
pip install git+https://github.com/perplexityai/pplx-kernels.git@v0.1.0

# Or in requirements.txt
git+https://github.com/perplexityai/pplx-kernels.git@v0.1.0
```

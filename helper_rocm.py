# Copyright 2025 FlashFloat authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
import re
import subprocess

import torch

try:
    from torch.utils.cpp_extension import ROCM_HOME
except:
    raise RuntimeError(
        "Base env does not provide Torch with support of ROCM SDK. Exit."
    )

HIP_VERSION_PAT = r"HIP version: (\S+)"
HIP_SDK_ROOT = "/opt/rocm"


def is_hip(hip_sdk_root=None) -> bool:
    SDK_ROOT = f"{hip_sdk_root or HIP_SDK_ROOT}"

    def _check_sdk_installed() -> bool:
        # return True if this dir points to a directory or symbolic link
        return os.path.isdir(SDK_ROOT)

    if not _check_sdk_installed():
        return False, None

    # we provide torch for the base env, check whether it is valid installation
    result = subprocess.run(
        [f"{SDK_ROOT}/bin/rocminfo | grep -o -m1 'gfx.*'"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        shell=True,
    )

    if result.returncode != 0:
        print("Use AMD pytorch, but no devices found!")
        return False, None

    target_amdgpu_arch = result.stdout.strip()
    print(f"target AMD gpu arch {target_amdgpu_arch}")
    return True, [target_amdgpu_arch]


# currently only support MI30X (MI308X, MI300XA) datacenter intelligent computing accelerator
_is_hip, target_amdgpu_arch = is_hip()

if _is_hip:
    assert ROCM_HOME is not None, "ROCM_HOME is not set"

    ROCM_HOME = os.environ.get("ROCM_HOME", ROCM_HOME)


def get_hipcc_rocm_version(hip_sdk_root=None):
    assert _is_hip

    SDK_ROOT = f"{hip_sdk_root or HIP_SDK_ROOT}"

    result = subprocess.run(
        [f"{SDK_ROOT}/bin/hipcc", "--version"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    # Check if the command was executed successfully
    if result.returncode != 0:
        print("Error running 'hipcc --version'")
        return None

    # Extract the version using a regular expression
    match = re.search(HIP_VERSION_PAT, result.stdout)
    if match:
        # Return the version string
        return match.group(1)
    else:
        print("Could not find HIP version in the output")
        return None


amd_libraries = ["hiprtc", "amdhip64"]

for flag in [
    # "-D__HIP_NO_HALF_OPERATORS__=1",
    # "-D__HIP_NO_HALF_CONVERSIONS__=1",
]:
    try:
        from torch.utils.cpp_extension import COMMON_HIPCC_FLAGS

        COMMON_HIPCC_FLAGS.remove(flag)
    except ValueError:
        pass

hipcc_flags = [
    "-D__HIP_PLATFORM_AMD__=1",
    f"--offload-arch={';'.join(target_amdgpu_arch if target_amdgpu_arch is not None else [])}",
]


def get_hip_libraries():
    return amd_libraries


def get_hipcc_flags():
    return hipcc_flags

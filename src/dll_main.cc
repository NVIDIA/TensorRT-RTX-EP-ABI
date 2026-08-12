// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// DllMain for the NV TensorRT-RTX ONNX Runtime Execution Provider.
//
// Purpose: pin this DLL into the process so it is never unloaded after the
// first DLL_PROCESS_ATTACH. This eliminates two App Verifier issues observed
// against earlier builds:
//
//   1. VERIFIER STOP 0x900 (APPLICATION_VERIFIER_LEAK_ALLOCATION)
//      The statically-linked protobuf in this DLL builds a global
//      DescriptorPool during `_static_init2_` (the generated AddDescriptorsImpl
//      callbacks). When the loader unmaps this DLL (e.g. after ORT tears down
//      the EpFactory at the end of device enumeration), those absl btree /
//      std::string allocations are still alive but owned by an unloaded
//      module - AVRF reports a leak attributed to this DLL.
//
//   2. VERIFIER STOP 0x304 (waiting on a thread handle in DllMain)
//      Calling google::protobuf::ShutdownProtobufLibrary() from
//      DLL_PROCESS_DETACH to release the descriptor pool is unsafe because
//      protobuf's shutdown path uses absl synchronization primitives that
//      can wait on thread handles - illegal under the loader lock.
//
// Pinning sidesteps both: GET_MODULE_HANDLE_EX_FLAG_PIN bumps the module's
// load count past the "permanent" threshold so subsequent FreeLibrary() calls
// become no-ops. The DLL stays in the process for its full lifetime, which
// matches how ORT uses EP DLLs in practice anyway.

#ifdef _WIN32

#include <windows.h>

extern "C" BOOL WINAPI DllMain(HINSTANCE /*hinstDLL*/, DWORD fdwReason, LPVOID /*lpReserved*/)
{
    if (fdwReason == DLL_PROCESS_ATTACH)
    {
        HMODULE self = nullptr;
        // Use DllMain's own address as the anchor so we pin *this* DLL regardless
        // of where it was loaded from.
        //
        // IMPORTANT: do NOT combine GET_MODULE_HANDLE_EX_FLAG_PIN with
        // GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT - the loader honours
        // UNCHANGED_REFCOUNT and silently skips the pin, so the DLL still
        // unloads and AVRF 0x900 fires.
        //
        // Failure here is non-fatal: the worst case is reverting to the prior
        // (unpinned) behaviour, so we don't propagate the error.
        (void)::GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_PIN | GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS,
                                   reinterpret_cast<LPCWSTR>(&DllMain), &self);
    }
    return TRUE;
}

#endif  // _WIN32

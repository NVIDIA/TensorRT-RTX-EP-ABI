# TensorRT RTX Execution Provider C++ Coding Guidelines

The TensorRT RTX Execution Provider C++ Coding Guidelines are derived from several sources, primarily:

- [NVIDIA TensorRT Coding Guidelines](https://github.com/NVIDIA/TensorRT/blob/main/CODING-GUIDELINES.md)
- [AUTOSAR C++ 2014](https://www.autosar.org/fileadmin/user_upload/standards/adaptive/17-03/AUTOSAR_RS_CPP14Guidelines.pdf)
- [MISRA C++ 2008](https://www.misra.org.uk/Activities/MISRAC/tabid/171/Default.aspx)
- [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html)

---

## Table of Contents

- [Language Standard](#language-standard)
- [Namespaces](#namespaces)
- [Constants](#constants)
- [Literals](#literals)
- [Brace Notation](#brace-notation)
- [Naming Conventions](#naming-conventions)
- [Tabs vs Spaces](#tabs-vs-spaces)
- [Formatting](#formatting)
- [Pointers and Memory Allocation](#pointers-and-memory-allocation)
- [Comments](#comments)
- [Disabling Code](#disabling-code)
- [Exceptions](#exceptions)
- [Casts](#casts)
- [Forward Declarations and Extern Variables](#forward-declarations-and-extern-variables)
- [Structures and Classes](#structures-and-classes)
- [Preprocessor Directives](#preprocessor-directives)
- [Signed vs Unsigned Integers](#signed-vs-unsigned-integers)
- [CUDA and TensorRT Specific Guidelines](#cuda-and-tensorrt-specific-guidelines)
- [ONNX Runtime Integration Guidelines](#onnx-runtime-integration-guidelines)
- [NVIDIA Copyright](#nvidia-copyright)
- [Appendix](#appendix)

---

## Language Standard

1. This project uses **C++17** as the language standard.
2. Use modern C++ features where appropriate (smart pointers, range-based for loops, auto keyword, etc.).
3. Ensure compatibility with MSVC (Windows), GCC (Linux), and Clang compilers.

---

## Namespaces

1. *MISRA C++: 2008 Rule 7-3-1*
   Global namespace shall only contain main, namespace declarations and extern "C" declarations. Use explicit or anonymous namespaces for everything else.

2. All TensorRT RTX EP code should be contained within the `trt_rtx_ep` namespace:
```cpp
namespace trt_rtx_ep
{
// ... implementation code ...
} // namespace trt_rtx_ep
```

3. Closing braces of namespaces should have a comment indicating the namespace it closes:
```cpp
namespace trt_rtx_ep
{
namespace utils
{
// ...
} // namespace utils
} // namespace trt_rtx_ep
```

4. Do not use `using namespace` directives in header files.

---

## Constants

1. Prefer `const` or `constexpr` variables over `#defines` whenever possible, as the latter are not visible to the compiler.

2. *MISRA C++: 2008 Rule 7-1-1 and 7-1-2*
   A variable that is not modified after its initialization should be declared as `const`.

3. Use `constexpr` for compile-time constants:
```cpp
constexpr size_t kMaxBatchSize = 256;
constexpr int32_t kDefaultDeviceId = 0;
```

4. For naming of constants, see the [Naming Conventions](#naming-conventions) section.

---

## Literals

1. Except `0` (only used in comparison for checking signedness/existence/emptiness), `nullptr`, `true`, and `false`, all other literals should only be used for variable initialization.

   Example - **Bad**:
```cpp
if (nbInputs == 2U) { /*...*/ }
```

   Example - **Good**:
```cpp
constexpr size_t kNbInputsWithBias = 2U;
if (nbInputs == kNbInputsWithBias) { /*...*/ }
```

2. *MISRA C++: 2008 Rule 2-13-4*
   Literal suffixes should be uppercase. For example, use `1234L` instead of `1234l`.

---

## Brace Notation

1. Use the [Allman indentation](https://en.wikipedia.org/wiki/Indent_style#Allman_style) style:
```cpp
if (condition)
{
    doSomething();
}
else
{
    doSomethingElse();
}
```

2. Put the semicolon for an empty `for` or `while` loop on a new line:
```cpp
while (condition)
    ;
```

3. *AUTOSAR C++14 Rule 6.6.3*, *MISRA C++: 2008 6-3-1*
   The statement forming the body of a `switch`, `while`, `do .. while` or `for` statement shall be a compound statement (use brace-delimited statements).

4. *AUTOSAR C++14 Rule 6.6.4*, *MISRA C++: 2008 Rule 6-4-1*
   `if` and `else` should always be followed by brace-delimited statements, even if empty or a single statement:
```cpp
// Good
if (condition)
{
    return true;
}

// Bad
if (condition)
    return true;
```

---

## Naming Conventions

### Filenames

* Use snake_case with lowercase letters: `tensorrt_rtx_execution_provider.cc`, `cuda_graph.h`
* Header files use `.h` extension, source files use `.cc` extension
* *NOTE*: All files involved in the compilation must have filenames that are case-insensitive unique.

### Types

* All types (including classes, structs, enums, type aliases) use PascalCase (CamelCase with uppercase first letter):
```cpp
class TensorrtRtxExecutionProvider;
struct ProviderOptions;
enum class DataType;
using HashValue = uint64_t;
```

### Local Variables, Methods, and Namespaces

* Use snake_case for local variables and function parameters:
```cpp
int device_id;
size_t num_inputs;
const void* data_ptr;
```

* Use camelCase (or PascalCase for interface methods) for methods:
```cpp
void ProcessInputs();
bool IsGraphCaptureEnabled() const;
```

* Use snake_case for namespaces:
```cpp
namespace trt_rtx_ep { }
namespace tensorrt_ptr { }
```

### Global Variables

* Non-static global variables: snake_case prefixed with 'g_':
```cpp
const OrtApi* g_ort_api = nullptr;
const OrtEpApi* g_ep_api = nullptr;
```

* Static or anonymous namespace global variables: snake_case prefixed with 's_':
```cpp
static std::once_flag s_init_once;
```

### Class Member Variables

* Private and protected members: use snake_case with trailing underscore:
```cpp
class TensorrtRtxExecutionProvider
{
private:
    int device_id_;
    std::string engine_cache_path_;
};
```

* Public members in struct-like classes (POD types) may omit the trailing underscore:
```cpp
struct ComputeState {
    std::string fused_node_name;
    int device_id;
    bool is_dynamic_shape;
};
```

> **Note**: This project follows the Google C++ Style Guide convention for member variables (trailing underscore for private members) rather than Hungarian notation (`m` prefix).

### Constants

* Enumerations, global constants, static constants, and magic-number constants use UPPER_SNAKE_CASE with prefix 'k':
```cpp
constexpr int32_t kDEFAULT_DEVICE_ID = 0;
constexpr size_t kMAX_BATCH_SIZE = 256;

enum class ErrorCode
{
    kSUCCESS = 0,
    kINVALID_ARGUMENT = 1,
    kOUT_OF_MEMORY = 2
};
```

* Function-scope constants that are not magic numbers are named like non-constant variables:
```cpp
const bool passed = validateInputs() && checkOutputs();
```

### Macros

* Use UPPER_SNAKE_CASE:
```cpp
#define TRT_RTX_EP_VERSION "1.0.0"
#define EXPORT_API __declspec(dllexport)
```

---

## Tabs vs Spaces

1. Use only spaces. Do not use tabs.
2. Indent 4 spaces at a time. This is enforced automatically if you format your code using the clang-format config.

---

## Formatting

1. Use the [LLVM clang-format](https://clang.llvm.org/docs/ClangFormat.html) tool for formatting your changes prior to submitting a PR.

2. Use a maximum of **120 characters per line**. The auto formatting tool will wrap longer lines.

3. Exceptions to formatting violations must be justified on a per-case basis. Bypassing formatting rules is discouraged, but can be achieved for exceptions as follows:
```cpp
// clang-format off
// .. Unformatted code ..
// clang-format on
```

4. Include order should be:
   1. Corresponding header file (for .cc files)
   2. Project headers
   3. ONNX Runtime headers
   4. TensorRT headers
   5. CUDA headers
   6. Standard library headers
   7. Third-party headers

```cpp
#include "tensorrt_rtx_execution_provider.h"

#include "tensorrt_rtx_provider_factory.h"
#include "utils/ep_utils.h"

#include "onnxruntime_cxx_api.h"

#include "NvInfer.h"
#include "NvOnnxParser.h"

#include <cuda_runtime.h>

#include <memory>
#include <string>
#include <vector>
```

---

## Pointers and Memory Allocation

1. *AUTOSAR C++ 2014: 18-5-2/3*
   Use smart pointers for allocating objects on the heap.

2. When picking a smart pointer:
   - Prefer `std::unique_ptr` for single resource ownership
   - Use `std::shared_ptr` for shared resource ownership
   - Use `std::weak_ptr` only in exceptional cases

3. Do not use smart pointers that have been deprecated in C++11 (`std::auto_ptr`).

4. For CUDA memory, use appropriate RAII wrappers or ensure proper cleanup in destructors.

5. For TensorRT objects, use the provided destroy functions or custom deleters:
```cpp
struct TrtDestroyer
{
    template <typename T>
    void operator()(T* obj) const
    {
        delete obj;
    }
};

using UniqueTrtPtr = std::unique_ptr<nvinfer1::ICudaEngine, TrtDestroyer>;
```

---

## Comments

1. C++ comments are required. C comments are not allowed except for special cases (inline parameter documentation).

2. C++ style for single-line comments:
```cpp
// This is a single line comment
```

3. In function calls where parameters are not obvious from inspection, use inline C comments:
```cpp
doSomeOperation(/* checkForErrors = */ false, /* asyncMode = */ true);
```

4. If the comment is a full sentence, it should be capitalized and punctuated properly.

5. Follow [Doxygen rules](http://www.doxygen.nl/manual/docblocks.html) for documenting class interfaces and function prototypes:
```cpp
//!
//! \brief Creates a TensorRT engine from an ONNX model.
//!
//! \param onnxModelPath Path to the ONNX model file.
//! \param maxBatchSize Maximum batch size for the engine.
//! \return Pointer to the created engine, or nullptr on failure.
//!
nvinfer1::ICudaEngine* createEngine(const std::string& onnxModelPath, int maxBatchSize);

struct TensorInfo
{
    std::string name;        //!< Name of the tensor
    nvinfer1::Dims dims;     //!< Dimensions of the tensor
    nvinfer1::DataType type; //!< Data type of the tensor
};
```

---

## Disabling Code

1. Use `#if` / `#endif` to disable code, preferably with a mnemonic condition:
```cpp
#if DEBUG_TENSORRT_INSTRUMENTATION
// ... debug code ...
#endif // DEBUG_TENSORRT_INSTRUMENTATION
```

2. *MISRA C++: 2008 Rule 2-7-2 and 2-7-3*
   Do NOT use comments to disable code. Use comments to explain code, not hide it.

3. For safety-critical code, avoid using compile-time expressions and DCE to disable code.

---

## Exceptions

1. Exceptions must not be thrown across library boundaries (DLL/SO boundaries).

2. Use error codes or status returns for API functions that cross library boundaries.

3. Internal helper functions may use exceptions for error handling.

4. Always catch exceptions at API boundaries and convert to appropriate error codes:
```cpp
OrtStatusPtr MyApiFunction()
{
    try
    {
        // ... implementation ...
        return nullptr; // Success
    }
    catch (const std::exception& e)
    {
        return OrtApis::CreateStatus(ORT_FAIL, e.what());
    }
}
```

---

## Casts

1. Use the least forceful cast necessary, or no cast if possible, to help the compiler diagnose unintended consequences.

2. Prefer C++ style casts over C-style casts:
   - `static_cast<T>()` for safe, well-defined conversions
   - `reinterpret_cast<T>()` for low-level reinterpretations (use sparingly)
   - `const_cast<T>()` for adding/removing const (use sparingly)
   - `dynamic_cast<T>()` for safe downcasting in class hierarchies

3. Avoid `dynamic_cast` in performance-critical code paths.

---

## Forward Declarations and Extern Variables

1. *MISRA C++: 2008 Rule 3-2-3*
   For safety-critical code, a type, object, or function used in multiple translation units shall be declared in one and only one file.

2. Place forward declarations in header files and include these header files as needed.

3. Minimize forward declarations; prefer including the full header when practical.

---

## Structures and Classes

1. *MISRA C++: 2008 Rule 14-7-1*
   All class templates, function templates, class template member functions and class template static members shall be instantiated at least once.

2. *MISRA C++: 2008 Rule 11-01*
   If a class is not a Plain Old Data Structure (POD), then its data members should be private.

3. Use the following member ordering in class declarations:
   1. Public types and type aliases
   2. Public static members
   3. Public constructors and destructor
   4. Public methods
   5. Protected members (same ordering)
   6. Private members (same ordering)

---

## Preprocessor Directives

1. *MISRA C++: 2008 Rule 16-0-2*
   `#define` and `#undef` of macros should be done only at global namespace.

2. Prefer `#if defined(...)` over `#ifdef`:
```cpp
#if defined(FOO) || defined(BAR)
void foo();
#endif // defined(FOO) || defined(BAR)
```

3. When nesting preprocessor directives, use indentation after the hash mark:
```cpp
#if defined(FOO)
#  if FOO == 0
#    define BAR 0
#  elif FOO == 1
#    define BAR 5
#  else
#    error "invalid FOO value"
#  endif
#endif
```

4. Use `#pragma once` for header file include guards (preferred for this project due to cross-platform simplicity).

5. *AUTOSAR C++ 2014: 7-1-6*
   Use `using` instead of `typedef`:
```cpp
// Good
using HashValue = uint64_t;

// Avoid
typedef uint64_t HashValue;
```

---

## Signed vs Unsigned Integers

1. Use signed integers instead of unsigned, except for:
   - Bitmap operations (use unsigned to avoid sign extension issues)
   - External library interfaces that expect unsigned integers
   - Loop comparisons against container sizes:
```cpp
for (size_t i = 0; i < tensors_.size(); ++i) // preferred style
```

2. Be explicit about integer types when interfacing with external APIs:
```cpp
// TensorRT uses int32_t for dimensions
int32_t batchSize = static_cast<int32_t>(inputShape[0]);
```

---

## CUDA and TensorRT Specific Guidelines

### CUDA Error Handling

1. Always check CUDA API return values:
```cpp
#define CUDA_CHECK(call)                                                         \
    do                                                                           \
    {                                                                            \
        cudaError_t status = call;                                               \
        if (status != cudaSuccess)                                               \
        {                                                                        \
            LOGS_DEFAULT(ERROR) << "CUDA error: " << cudaGetErrorString(status); \
            return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, cudaGetErrorString(status)); \
        }                                                                        \
    } while (0)
```

### TensorRT Resource Management

1. Use RAII patterns for TensorRT objects:
```cpp
class TrtEngine
{
public:
    explicit TrtEngine(nvinfer1::ICudaEngine* engine)
        : engine_(engine)
    {
    }

    ~TrtEngine()
    {
        if (engine_)
        {
            delete engine_;
        }
    }

    // Delete copy operations
    TrtEngine(const TrtEngine&) = delete;
    TrtEngine& operator=(const TrtEngine&) = delete;

    // Allow move operations
    TrtEngine(TrtEngine&& other) noexcept
        : engine_(other.engine_)
    {
        other.engine_ = nullptr;
    }

private:
    nvinfer1::ICudaEngine* engine_;
};
```

### Stream Management

1. Always pass CUDA streams explicitly rather than relying on the default stream.
2. Synchronize streams appropriately before accessing results on the CPU.

---

## ONNX Runtime Integration Guidelines

### OrtEpApi Conformance

1. All exported functions must follow the OrtEpApi interface specifications.
2. Use the provided status creation utilities for error reporting.
3. Handle all ORT callback functions safely and validate parameters.

### Memory Management

1. Memory allocated by ONNX Runtime should be freed by ONNX Runtime.
2. Memory allocated by the EP should be freed by the EP.
3. Never transfer memory ownership across the ORT-EP boundary without explicit documentation.

### Thread Safety

1. The EP factory must be thread-safe for concurrent EP instance creation.
2. Individual EP instances are not required to be thread-safe (ONNX Runtime handles synchronization).
3. Use `std::mutex` or `std::shared_mutex` for protecting shared resources.

---

## NVIDIA Copyright

All TensorRT RTX Execution Provider source files should contain an NVIDIA copyright header with SPDX identifiers. The following block should be prepended to the top of all source files (.cpp, .cc, .h, .cu, .py):

```cpp
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
```

---

## Appendix

### Abbreviation Words and Compound Words in Names

* Abbreviation words (usually fully-capitalized in literature) are treated as normal words without special capitalization:
  - `gpuAllocator` (GPU → gpu)
  - `cudaStream` (CUDA → cuda)
  - `onnxModel` (ONNX → onnx)

* Compound words can be abbreviated into fully capitalized letters:
  - `TRT` for TensorRT
  - `RTX` for RTX
  - `EP` for Execution Provider
  - `ORT` for ONNX Runtime

### Terminology

* **CUDA code**: Code that must be compiled with a CUDA compiler, including:
  - Declaration or definition of variables with `__device__`, `__managed__`, or `__constant__` keywords
  - Declaration or definition of device functions decorated with `__device__`
  - Declaration or definition of kernels decorated with `__global__`
  - Kernel launching with `<<<...>>>` syntax

* **Host code**: Code that runs on the CPU, including CUDA runtime API calls.

* **EP (Execution Provider)**: A plugin that provides hardware-accelerated execution for ONNX Runtime.

* **Engine**: A TensorRT optimized inference engine compiled from an ONNX model.

* **Context**: A TensorRT execution context used for inference.

### Common Pitfalls

1. **C headers**: Use C++ equivalents:
   - Use `<cstdint>` instead of `<stdint.h>`
   - Use `<cstring>` instead of `<string.h>`

2. **C library functions**: Prefer C++ alternatives:
   - Use `std::fill_n()` or brace initialization instead of `memset()`
   - Use `std::copy()` instead of `memcpy()` for non-trivial types

3. **Const correctness**: When specifying pointers to const data:
```cpp
char const* const errStr = getErrorStr(status);
```

4. **RAII violations**: Always use RAII for resource management. Avoid manual resource cleanup patterns.

5. **Thread safety**: Be aware of TensorRT's threading model and ensure proper synchronization.

---

## Tools and Automation

### Clang-Format

Run clang-format before submitting code:
```bash
# Format a single file
clang-format -i src/myfile.cc

# Format all source files
find src -name "*.cc" -o -name "*.h" | xargs clang-format -i
```

### Static Analysis

Consider using static analysis tools:
- Clang-Tidy for C++ best practices
- Cppcheck for additional static analysis
- NVIDIA Compute Sanitizer for CUDA code

---

*Last updated: December 2025*


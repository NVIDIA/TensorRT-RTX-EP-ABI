TensorRT RTX Execution Provider - Complete Build Guide
======================================================

This comprehensive guide will walk you through the entire process of building the TensorRT RTX Execution Provider for ONNX Runtime from source.

Table of Contents
-----------------

1. `Prerequisites <#prerequisites>`__
2. `Environment Setup <#environment-setup>`__
3. `Build Process <#build-process>`__
4. `Building with vcpkg <#building-with-vcpkg>`__
5. `Verification <#verification>`__
6. `Troubleshooting <#troubleshooting>`__
7. `Advanced Configuration <#advanced-configuration>`__
8. `Integration Guide <#integration-guide>`__

--------------

Prerequisites
-------------

Required Software
~~~~~~~~~~~~~~~~~

1. CMake (Version 3.20 or Higher)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Download:**
- Visit: https://cmake.org/download/
- Download the Windows x64 Installer (e.g., ``cmake-3.28.0-windows-x86_64.msi``)

**Installation:**
1. Run the installer
2. **Important**: During installation, select “Add CMake to the system PATH for all users”
3. Complete the installation

**Verification:**

.. code:: powershell

   cmake --version

Expected output: ``cmake version 3.20.0`` or higher

--------------

2. Visual Studio 2022
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Download:**
- Visit: https://visualstudio.microsoft.com/downloads/
- Download Visual Studio 2022 Community Edition (free)

**Installation:**
1. Run the Visual Studio Installer
2. Select the **“Desktop development with C++”** workload
3. In the “Individual components” tab, ensure these are selected:
- MSVC v143 - VS 2022 C++ x64/x86 build tools (or latest)
- Windows 10 SDK (10.0.19041.0 or later)
- C++ CMake tools for Windows
4. Complete the installation (requires ~7 GB)

**Verification:**
Open Developer Command Prompt for VS 2022 and run:

.. code:: powershell

   cl

Expected output: Microsoft C/C++ Optimizing Compiler Version information

--------------

3. NVIDIA CUDA Toolkit (Version 12.9 or Higher)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Download:**
- Visit: https://developer.nvidia.com/cuda-downloads
- Select:
- Operating System: Windows
- Architecture: x86_64
- Version: (your Windows version)
- Installer Type: exe (local) *recommended*

**Installation:**
1. Run the CUDA installer
2. Choose “Express Installation” (recommended) or “Custom”
3. Note the installation path (for example, ``C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2``)
4. The installer will install:
- CUDA Toolkit
- CUDA Samples
- CUDA Documentation
- NSight tools

**Verification:**

.. code:: powershell

   nvcc --version

Expected output: CUDA compilation tools release information

Check installation path:

.. code:: powershell

   dir "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\"

--------------

4. vcpkg (Optional)
^^^^^^^^^^^^^^^^^^^

vcpkg is needed when using the ``--use_vcpkg`` build flag. The build script can automatically clone and bootstrap vcpkg, but you can also set it up manually.

**Automatic Setup:**
The ``build.bat`` script will clone vcpkg into a ``vcpkg/`` directory adjacent to the project and bootstrap it automatically when ``--use_vcpkg`` is passed.

**Manual Setup:**

.. code:: powershell

   # Clone vcpkg adjacent to the project directory
   cd ..
   git clone https://github.com/microsoft/vcpkg.git
   cd vcpkg
   .\bootstrap-vcpkg.bat
   cd ..\trt-rtx-ep-abi

**Verification:**

.. code:: powershell

   ..\vcpkg\vcpkg --version

--------------

Required SDKs
~~~~~~~~~~~~~

1. ONNX Runtime SDK
^^^^^^^^^^^^^^^^^^^

**Download:**
1. Visit: https://github.com/microsoft/onnxruntime/releases
2. Find a supported release (1.24 or newer; CI uses 1.26.0)
3. Download: ``onnxruntime-win-x64-1.26.0.zip``

**Extraction:**

.. code:: powershell

   # Example: Extract to C:\SDK\
   Expand-Archive -Path "onnxruntime-win-x64-1.26.0.zip" -DestinationPath "C:\SDK\"

**Verification:**
Check the extracted directory structure:

.. code:: text

   C:\SDK\onnxruntime-win-x64-1.26.0\
   ├── include\
   │   ├── onnxruntime_c_api.h
   │   ├── onnxruntime_cxx_api.h
   │   └── ... (other headers)
   └── lib\
       ├── onnxruntime.dll
       ├── onnxruntime.lib
       └── ... (other libraries)

--------------

2. TensorRT RTX SDK
^^^^^^^^^^^^^^^^^^^

**Download:**
- Contact NVIDIA for access to TensorRT RTX SDK
- Download a release compatible with the installed CUDA major version. The CI configuration uses TensorRT RTX 1.5.0.114 for CUDA 13.2.
- Download the Windows SDK package

**Extraction:**

.. code:: powershell

   # Example: Extract to C:\SDK\
   Expand-Archive -Path "TensorRT-RTX-1.5.0.114.zip" -DestinationPath "C:\SDK\"

**Verification:**
Check the extracted directory structure:

.. code:: text

   C:\SDK\TensorRT-RTX-1.5.0.114\
   ├── include\
   │   ├── NvInfer.h
   │   ├── NvInferVersion.h
   │   └── ... (other headers)
   └── lib\
       ├── tensorrt_rtx_1_1.lib
       ├── tensorrt_onnxparser_rtx_1_1.lib
       └── ... (other libraries)

--------------

Environment Setup
-----------------

Verify All Prerequisites
~~~~~~~~~~~~~~~~~~~~~~~~

Before building, verify all prerequisites are correctly installed:

.. code:: powershell

   # Check CMake
   cmake --version

   # Check Visual Studio C++ compiler
   where cl

   # Check CUDA
   nvcc --version

   # Check SDK directories exist
   Test-Path "C:\SDK\onnxruntime-win-x64-1.26.0"
   Test-Path "C:\SDK\TensorRT-RTX-1.5.0.114"
   Test-Path "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2"

All checks should return successful results.

--------------

Build Process
-------------

Automated GitHub Actions build
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The repository's ``Build and test`` workflow builds Release configurations on
Ubuntu 24.04 and Windows Server 2025. The Linux job runs in NVIDIA's pinned
``nvidia/cuda:13.2.1-devel-ubuntu24.04`` container, while the Windows job
installs CUDA 13.2. Both jobs download ONNX Runtime 1.26.0 during CMake
configuration and download TensorRT RTX 1.5.0.114 from NVIDIA's SDK archives.

Compilation is required. Unit-test execution is informational because standard
GitHub-hosted runners do not provide an NVIDIA GPU: the CTest steps use
``continue-on-error`` and upload JUnit reports without changing the build
result.

For local builds, the TensorRT RTX SDK can be supplied either as an extracted
directory or as a complete archive URL:

.. code:: powershell

   build.bat --cuda_home "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2" ^
             --trt_rtx_url "https://developer.nvidia.com/downloads/trt/rtx_sdk/secure/1.5/TensorRT-RTX-1.5.0.114-Windows-amd64-cuda-13.2-Release-external.zip"

.. code:: bash

   ./build.sh --cuda_home /usr/local/cuda-13.2 \
              --trt_rtx_url "https://developer.nvidia.com/downloads/trt/rtx_sdk/secure/1.5/TensorRT-RTX-1.5.0.114-Linux-x86_64-cuda-13.2-Release-external.tar.zst"

When ``--onnxruntime_home`` is omitted, CMake downloads the configured ONNX
Runtime version automatically.

Method 1: Using the Build Script (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The easiest and most reliable way to build the project is using the provided ``build.bat`` script.

Step 1: Open Command Prompt
^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Press ``Win + R``

2. Type ``cmd`` and press Enter

3. Navigate to the project directory:

   .. code:: powershell

      cd C:\iraut\trt-rtx-ep-abi

Step 2: Run the Build Script
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: powershell

   build.bat --cuda_home "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2" ^
             --onnxruntime_home "C:\SDK\onnxruntime-win-x64-1.26.0" ^
             --trt_rtx_home "C:\SDK\TensorRT-RTX-1.5.0.114"

**Replace the paths** with your actual installation paths!

**Note:** Arguments can be provided in any order:

.. code:: powershell

   build.bat --trt_rtx_home "C:\SDK\TensorRT-RTX" --cuda_home "C:\CUDA" --onnxruntime_home "C:\onnxruntime"

**With custom build directory:**

.. code:: powershell

   build.bat --cuda_home "C:\CUDA" --onnxruntime_home "C:\onnxruntime" ^
             --trt_rtx_home "C:\TensorRT-RTX" --build_dir "C:\mybuild"

**With custom build configuration (Debug/Release/RelWithDebInfo):**

.. code:: powershell

   build.bat --cuda_home "C:\CUDA" --onnxruntime_home "C:\onnxruntime" ^
             --trt_rtx_home "C:\TensorRT-RTX" --config Debug

**Incremental build only (after code changes):**

.. code:: powershell

   build.bat --cuda_home "C:\CUDA" --onnxruntime_home "C:\onnxruntime" ^
             --trt_rtx_home "C:\TensorRT-RTX" --build

**Reconfigure and build (after CMakeLists.txt changes):**

.. code:: powershell

   build.bat --cuda_home "C:\CUDA" --onnxruntime_home "C:\onnxruntime" ^
             --trt_rtx_home "C:\TensorRT-RTX" --update --build

**Full clean rebuild (combine all actions):**

.. code:: powershell

   build.bat --cuda_home "C:\CUDA" --onnxruntime_home "C:\onnxruntime" ^
             --trt_rtx_home "C:\TensorRT-RTX" --clean --update --build

Step 3: Monitor Build Progress
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The script will:
1. ✓ Validate all paths exist
2. ✓ Display build configuration
3. ✓ Clean previous build directory (if exists)
4. ✓ Create new build directory
5. ✓ Run CMake configuration
6. ✓ Build the project with parallel compilation
7. ✓ Display success message with output location

**Expected Output:**

::

   ============================================================================
   Build Configuration:
     CUDA Toolkit:        C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2
     ONNX Runtime SDK:    C:\SDK\onnxruntime-win-x64-1.26.0
     TensorRT RTX SDK:    C:\SDK\TensorRT-RTX-1.5.0.114
     Build Directory:     build
     Build Config:        Release
     Source Directory:    C:\iraut\trt-rtx-ep-abi
     Actions:             clean + update + build
   ============================================================================

   Configuring project with CMake...
   -- Building for: Visual Studio 17 2022
   -- The CXX compiler identification is MSVC 19.38.33133.0
   -- Using ONNXRUNTIME_ROOT: C:\SDK\onnxruntime-win-x64-1.26.0
   -- Using TRT_RTX_ROOT: C:\SDK\TensorRT-RTX-1.5.0.114
   -- NV_TRT_MAJOR_RTX is 1
   -- NV_TRT_MINOR_RTX is 1
   ...
   -- Configuring done
   -- Generating done
   -- Build files have been written to: C:\iraut\trt-rtx-ep-abi\build

   Building project with parallel compilation...
   Microsoft (R) Build Engine version ...
   ...
   Build succeeded.
       0 Warning(s)
       0 Error(s)

   ============================================================================
   Build completed successfully!
   Output: build\Release\onnxruntime_providers_nv_tensorrt_rtx.dll
   ============================================================================

**Build Time:** Approximately 5-15 minutes depending on your system.

--------------

Method 2: Manual Build
~~~~~~~~~~~~~~~~~~~~~~

If you prefer manual control or need to customize the build:

Step 1: Create and Enter Build Directory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: powershell

   mkdir build
   cd build

Step 2: Configure with CMake
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: powershell

   cmake -G "Visual Studio 17 2022" -A x64 ^
         -DCUDAToolkit_ROOT="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2" ^
         -DONNXRUNTIME_ROOT="C:\SDK\onnxruntime-win-x64-1.26.0" ^
         -DTRT_RTX_ROOT="C:\SDK\TensorRT-RTX-1.5.0.114" ^
         ..

**For Visual Studio 2019:**

.. code:: powershell

   cmake -G "Visual Studio 16 2019" -A x64 ...

Step 3: Build Release Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: powershell

   cmake --build . --config Release --parallel

**For Debug Configuration:**

.. code:: powershell

   cmake --build . --config Debug --parallel

Step 4: Return to Project Root
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: powershell

   cd ..

--------------

Building with vcpkg
-------------------

By default, the build uses CMake’s FetchContent to download and build dependencies (ONNX, protobuf, abseil). You can alternatively use vcpkg as the package manager.

Using the Build Script
~~~~~~~~~~~~~~~~~~~~~~

Add the ``--use_vcpkg`` flag to any build command:

.. code:: powershell

   build.bat --cuda_home "C:\CUDA" --onnxruntime_home "C:\onnxruntime" ^
             --trt_rtx_home "C:\TensorRT-RTX" --use_vcpkg

This will:
1. Clone and bootstrap vcpkg (if not already present in ``../vcpkg/``)
2. Use vcpkg to resolve dependencies (protobuf, ONNX, abseil) via the project’s ``vcpkg.json`` manifest
3. Configure CMake with the vcpkg toolchain file and ``x64-windows-static-md`` triplet

Manual Build with vcpkg
~~~~~~~~~~~~~~~~~~~~~~~

.. code:: powershell

   mkdir build && cd build
   cmake -G "Visual Studio 17 2022" -A x64 ^
         -DCUDAToolkit_ROOT="C:\CUDA" ^
         -DONNXRUNTIME_ROOT="C:\onnxruntime" ^
         -DTRT_RTX_ROOT="C:\TensorRT-RTX" ^
         -DUSE_VCPKG=ON ^
         -DCMAKE_TOOLCHAIN_FILE="..\vcpkg\scripts\buildsystems\vcpkg.cmake" ^
         -DVCPKG_TARGET_TRIPLET=x64-windows-static-md ^
         -DVCPKG_HOST_TRIPLET=x64-windows ^
         ..
   cmake --build . --config Release --parallel

..

   **Note:** The first vcpkg build will take longer as it downloads and compiles the dependency packages. Subsequent builds reuse the cached packages.

--------------

Verification
------------

Build Outputs
~~~~~~~~~~~~~

After a successful build, verify the following files exist:

**Release Build:**

::

   build\Release\
   ├── onnxruntime_providers_nv_tensorrt_rtx.dll    # Main shared library
   ├── onnxruntime_providers_nv_tensorrt_rtx.lib    # Import library
   └── onnxruntime_providers_nv_tensorrt_rtx.pdb    # Debug symbols

**Debug Build:**

::

   build\Debug\
   ├── onnxruntime_providers_nv_tensorrt_rtx.dll
   ├── onnxruntime_providers_nv_tensorrt_rtx.lib
   └── onnxruntime_providers_nv_tensorrt_rtx.pdb

Verify DLL Dependencies
~~~~~~~~~~~~~~~~~~~~~~~

Check that the DLL was built correctly:

.. code:: powershell

   dumpbin /DEPENDENTS build\Release\onnxruntime_providers_nv_tensorrt_rtx.dll

Expected dependencies should include:
- ``onnxruntime.dll``
- ``cudart64_*.dll`` (CUDA runtime)
- ``tensorrt_rtx_1_1.dll`` (TensorRT RTX)
- Standard Windows DLLs (KERNEL32.dll, etc.)

Check Exported Symbols
~~~~~~~~~~~~~~~~~~~~~~

Verify the DLL exports the correct functions:

.. code:: powershell

   dumpbin /EXPORTS build\Release\onnxruntime_providers_nv_tensorrt_rtx.dll

Should show exported functions like:
- ``CreateExecutionProviderFactory``
- ``GetProviderInfo``
- etc.

--------------

Troubleshooting
---------------

CMake Configuration Errors
~~~~~~~~~~~~~~~~~~~~~~~~~~

Error: “ONNXRUNTIME_ROOT must be set via command line”
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Cause:** The ONNXRUNTIME_ROOT path was not provided to CMake.

**Solution:**

.. code:: powershell

   # Ensure you provide the -DONNXRUNTIME_ROOT flag
   cmake ... -DONNXRUNTIME_ROOT="C:\SDK\onnxruntime-win-x64-1.26.0" ...

--------------

Error: “TRT_RTX_ROOT must be set via command line”
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Cause:** The TRT_RTX_ROOT path was not provided to CMake.

**Solution:**

.. code:: powershell

   # Ensure you provide the -DTRT_RTX_ROOT flag
   cmake ... -DTRT_RTX_ROOT="C:\SDK\TensorRT-RTX-1.5.0.114" ...

--------------

Error: “Could not find CUDAToolkit”
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Cause:** CMake cannot locate the CUDA installation.

**Solutions:**

1. **Verify CUDA Installation:**

   .. code:: powershell

      dir "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\"

2. **Provide Explicit Path:**

   .. code:: powershell

      cmake ... -DCUDAToolkit_ROOT="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2" ...

3. **Check Environment Variables:**

   .. code:: powershell

      echo %CUDA_PATH%

   Should point to CUDA installation directory.

4. **Ensure CUDA bin is in PATH:**

   .. code:: powershell

      echo %PATH%

   Should contain ``C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin``

--------------

Library Not Found Errors
~~~~~~~~~~~~~~~~~~~~~~~~

Error: “Could not find onnxruntime library”
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Cause:** CMake cannot find ``onnxruntime.lib`` in the specified SDK.

**Solutions:**

1. **Verify Library Exists:**

   .. code:: powershell

      dir "C:\SDK\onnxruntime-win-x64-1.26.0\lib\onnxruntime.lib"

2. **Check SDK Extraction:** Ensure the ONNX Runtime SDK was fully extracted.

3. **Verify Path:** Double-check the ONNXRUNTIME_ROOT path points to the correct directory.

--------------

Error: “Could not find tensorrt_rtx_X_X library”
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Cause:** CMake cannot find the TensorRT RTX libraries.

**Solutions:**

1. **Verify Libraries Exist:**

   .. code:: powershell

      dir "C:\SDK\TensorRT-RTX-1.5.0.114\lib\"

   Should show files like:

   - ``tensorrt_rtx_1_1.lib``
   - ``tensorrt_onnxparser_rtx_1_1.lib``

2. **Check Version Numbers:** The CMakeLists.txt automatically detects version numbers from ``NvInferVersion.h``. Verify this file exists:

   .. code:: powershell

      dir "C:\SDK\TensorRT-RTX-1.5.0.114\include\NvInferVersion.h"

3. **Manual Version Check:** Open ``NvInferVersion.h`` and look for:

   .. code:: cpp

      #define TRT_MAJOR_RTX 1
      #define TRT_MINOR_RTX 1

--------------

Build Errors
~~~~~~~~~~~~

Error: Compilation errors related to C++ standard
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Cause:** Visual Studio or compiler doesn’t support C++20.

**Solution:**
- Use Visual Studio 2019 16.11+ (MSVC toolset v14.29+, ``_MSC_VER >= 1929``) or Visual Studio 2022 (MSVC toolset v14.30+, ``_MSC_VER >= 1930``)
- Update Visual Studio to the latest version
- Confirm the MSVC toolset version in the Visual Studio Installer’s individual-components list (or check ``_MSC_VER``) — ``/std:c++20`` requires the thresholds above

--------------

Error: Missing header files (e.g., “onnxruntime_c_api.h: No such file”)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Cause:** Include directories are not correctly configured.

**Solutions:**

1. **Clean and Rebuild:**

   .. code:: powershell

      rmdir /s /q build
      mkdir build
      cd build
      # Re-run CMake configuration

2. **Verify SDK Headers:**

   .. code:: powershell

      dir "C:\SDK\onnxruntime-win-x64-1.26.0\include\*.h"
      dir "C:\SDK\TensorRT-RTX-1.5.0.114\include\*.h"

--------------

Error: “Cannot open file ‘onnxruntime.lib’”
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Cause:** Linker cannot find the ONNX Runtime library.

**Solution:**
- Ensure the ONNXRUNTIME_ROOT path is correct
- Verify ``onnxruntime.lib`` exists in ``<ONNXRUNTIME_ROOT>\lib\``
- Clean build directory and reconfigure

--------------

General Build Failures
~~~~~~~~~~~~~~~~~~~~~~

Strategy: Clean Build
^^^^^^^^^^^^^^^^^^^^^

Often, build issues can be resolved by cleaning and starting fresh:

.. code:: powershell

   # From project root
   rmdir /s /q build

   # Run build script again (full clean build)
   build.bat --cuda_home "..." --onnxruntime_home "..." --trt_rtx_home "..."

   # Or specify a different build directory
   build.bat --cuda_home "..." --onnxruntime_home "..." --trt_rtx_home "..." --build_dir "build_clean"

   # For incremental builds (faster, keeps existing build)
   build.bat --cuda_home "..." --onnxruntime_home "..." --trt_rtx_home "..." --build

   # To update CMake config and build
   build.bat --cuda_home "..." --onnxruntime_home "..." --trt_rtx_home "..." --update --build

   # Full clean rebuild (all steps)
   build.bat --cuda_home "..." --onnxruntime_home "..." --trt_rtx_home "..." --clean --update --build

Strategy: Verbose Build Output
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For more detailed error information:

.. code:: powershell

   cmake --build build --config Release --parallel --verbose

--------------

Advanced Configuration
----------------------

Building for Debug
~~~~~~~~~~~~~~~~~~

**Using Build Script:**
Use the ``--config`` flag to specify the build configuration:

.. code:: powershell

   build.bat --cuda_home "..." --onnxruntime_home "..." --trt_rtx_home "..." --config Debug

**Available configurations:**
- ``Release`` - Optimized build (default)
- ``Debug`` - Debug build with symbols
- ``RelWithDebInfo`` - Release build with debug information

**Manual Build:**

.. code:: powershell

   cmake --build build --config Debug --parallel

Building with Ninja
~~~~~~~~~~~~~~~~~~~

For faster builds, use Ninja instead of Visual Studio:

1. **Install Ninja:**

   .. code:: powershell

      # Using Chocolatey
      choco install ninja

      # Or download from https://ninja-build.org/

2. **Configure with Ninja:**

   .. code:: powershell

      # Open Developer Command Prompt for VS 2022
      mkdir build
      cd build
      cmake -G "Ninja" ^
            -DCMAKE_BUILD_TYPE=Release ^
            -DCUDAToolkit_ROOT="..." ^
            -DONNXRUNTIME_ROOT="..." ^
            -DTRT_RTX_ROOT="..." ^
            ..

3. **Build:**

   .. code:: powershell

      ninja

Custom Build Options
~~~~~~~~~~~~~~~~~~~~

You can add additional CMake options:

.. code:: powershell

   # Enable verbose CMake output
   cmake ... -DCMAKE_VERBOSE_MAKEFILE=ON ...

   # Specify different build type
   cmake ... -DCMAKE_BUILD_TYPE=RelWithDebInfo ...

   # Change output directory
   cmake ... -DCMAKE_RUNTIME_OUTPUT_DIRECTORY="C:\MyOutput" ...

--------------

Integration Guide
-----------------

Using the Built Execution Provider
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After building, you need to integrate the execution provider with your application.

Step 1: Copy Required Files
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Copy the following to your application directory:

**From Build Output:**

::

   build\Release\onnxruntime_providers_nv_tensorrt_rtx.dll

**From ONNX Runtime SDK:**

::

   <ONNXRUNTIME_ROOT>\lib\onnxruntime.dll
   <ONNXRUNTIME_ROOT>\lib\onnxruntime_providers_shared.dll

**From TensorRT RTX SDK:**

::

   <TRT_RTX_ROOT>\lib\tensorrt_rtx_1_1.dll
   <TRT_RTX_ROOT>\lib\tensorrt_onnxparser_rtx_1_1.dll
   (and any other required DLLs)

**From CUDA Toolkit:**

::

   <CUDA_PATH>\bin\cudart64_*.dll
   (and other required CUDA DLLs)

Step 2: Load the Execution Provider
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Example C++ Code:**

.. code:: cpp

   #include <onnxruntime_cxx_api.h>
   #include <iostream>

   int main() {
       try {
           // Initialize ONNX Runtime environment
           Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "TensorRTRtxEP");
           
           // Create session options
           Ort::SessionOptions session_options;
           
           // Load TensorRT RTX execution provider
           // The specific API depends on how the provider exposes itself
           // Consult the provider's documentation for exact usage
           
           // Create session
           const wchar_t* model_path = L"path/to/your/model.onnx";
           Ort::Session session(env, model_path, session_options);
           
           std::cout << "Session created successfully with TensorRT RTX EP!" << std::endl;
           
           // Your inference code here...
           
       } catch (const Ort::Exception& e) {
           std::cerr << "ONNX Runtime error: " << e.what() << std::endl;
           return 1;
       }
       
       return 0;
   }

Step 3: Configure Runtime Environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Ensure all DLL dependencies are accessible:

**Option 1: Copy DLLs to Application Directory** (Simplest)

**Option 2: Add to System PATH** (Development)

.. code:: powershell

   $env:PATH += ";C:\SDK\onnxruntime-win-x64-1.26.0\lib"
   $env:PATH += ";C:\SDK\TensorRT-RTX-1.5.0.114\lib"
   $env:PATH += ";C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin"

**Option 3: Use Application Configuration** (Production)
- Package all required DLLs with your application installer
- Use delay-load DLLs with custom loading paths

--------------

Build Artifacts
---------------

Generated Files
~~~~~~~~~~~~~~~

After a successful build, the following artifacts are generated:

::

   build\
   ├── CMakeCache.txt                               # CMake configuration cache
   ├── CMakeFiles\                                  # CMake internal files
   ├── cmake_install.cmake                          # Install configuration
   ├── TensorRTRtxEp.vcxproj                       # Visual Studio project file
   ├── trt-rtx-ep.sln                              # Visual Studio solution file
   │
   ├── Release\                                     # Release build outputs
   │   ├── onnxruntime_providers_nv_tensorrt_rtx.dll     # Main library
   │   ├── onnxruntime_providers_nv_tensorrt_rtx.lib     # Import library
   │   └── onnxruntime_providers_nv_tensorrt_rtx.pdb     # Debug symbols
   │
   ├── Debug\                                       # Debug build outputs (if built)
   │   ├── onnxruntime_providers_nv_tensorrt_rtx.dll
   │   ├── onnxruntime_providers_nv_tensorrt_rtx.lib
   │   └── onnxruntime_providers_nv_tensorrt_rtx.pdb
   │
   └── _deps\                                       # Downloaded dependencies (FetchContent builds only)
       ├── onnx-build\                             # Built ONNX libraries
       ├── onnx-src\                               # ONNX source code
       ├── protobuf-build\                         # Built Protobuf libraries
       ├── protobuf-src\                           # Protobuf source code
       └── abseil-src\                             # Abseil libraries

..

   **Note:** When building with ``--use_vcpkg``, the ``_deps/`` directory is not created. Instead, dependencies are managed by vcpkg and stored in its own package cache.

--------------

Performance Optimization
------------------------

Build Optimizations
~~~~~~~~~~~~~~~~~~~

For optimal performance in production:

1. **Use Release Build:**

   .. code:: powershell

      cmake --build . --config Release

2. **Enable Link-Time Optimization (LTO):**
   Add to CMakeLists.txt:

   .. code:: cmake

      set_target_properties(TensorRTRtxEp PROPERTIES 
          INTERPROCEDURAL_OPTIMIZATION TRUE)

3. **Use Parallel Compilation:**
   Already enabled in build.bat with ``--parallel`` flag

--------------

Cleanup
-------

Cleaning Build Artifacts
~~~~~~~~~~~~~~~~~~~~~~~~

To remove all build artifacts and start fresh:

.. code:: powershell

   # Remove build directory
   rmdir /s /q build

   # Remove CMake cache (if exists in root)
   del CMakeCache.txt

   # Remove Visual Studio user files (optional)
   del /s *.user

Keeping Source Clean
~~~~~~~~~~~~~~~~~~~~

The build system is designed to keep all build artifacts in the ``build\`` directory. Your source tree remains clean and can be easily version controlled.

--------------

Summary
-------

Quick Reference
~~~~~~~~~~~~~~~

**Build Command:**

.. code:: powershell

   build.bat --cuda_home "<CUDA_PATH>" --onnxruntime_home "<ONNXRUNTIME_ROOT>" --trt_rtx_home "<TRT_RTX_ROOT>" [options]

**Required Arguments:**
\| Flag \| Description \|
\|——\|————-\|
\| ``--cuda_home`` \| Path to CUDA Toolkit installation \|
\| ``--onnxruntime_home`` \| Path to ONNX Runtime SDK root directory \|
\| ``--trt_rtx_home`` \| Path to TensorRT RTX SDK root directory \|

**Optional Arguments:**
\| Flag \| Description \|
\|——\|————-\|
\| ``--build_dir`` \| Build output directory (default: ``build``) \|
\| ``--config`` \| Build configuration: ``Debug``, ``Release`` (default), ``RelWithDebInfo`` \|
\| ``--clean`` \| Clean the build directory \|
\| ``--update`` \| Run CMake configuration \|
\| ``--build`` \| Compile the project \|
\| ``--use_vcpkg`` \| Use vcpkg package manager for dependencies \|
\| ``--production`` \| Enable production build with signature verification \|
\| ``-h``, ``--help``, ``/?`` \| Show help message \|

**Build Actions (can be combined, executed in order: clean → update → build):**
\| Flags \| Description \|
\|——-\|————-\|
\| (no flags) \| Full build: clean + update + build \|
\| ``--build`` \| Incremental build only (fastest for code changes) \|
\| ``--update --build`` \| Reconfigure and build (for CMakeLists.txt changes) \|
\| ``--clean --update`` \| Clean and reconfigure only (no build) \|
\| ``--clean --update --build`` \| Full build (same as no flags) \|

**Output Location:**

::

   build\<CONFIG>\onnxruntime_providers_nv_tensorrt_rtx.dll

Where ``<CONFIG>`` is ``Release``, ``Debug``, or ``RelWithDebInfo`` based on ``--config`` flag.

**Common Issues:**
- ✓ Verify all three SDK paths are correct
- ✓ Ensure CUDA is in system PATH
- ✓ Use Visual Studio 2022 or 2019
- ✓ Clean build directory if configuration changes

**Next Steps:**
1. Build the library
2. Copy DLLs to your application
3. Load the execution provider in your code
4. Run inference with TensorRT RTX acceleration!

--------------

Additional Resources
--------------------

- **ONNX Runtime Documentation:** https://onnxruntime.ai/docs/
- **TensorRT Documentation:** https://docs.nvidia.com/deeplearning/tensorrt/
- **CUDA Toolkit Documentation:** https://docs.nvidia.com/cuda/
- **CMake Documentation:** https://cmake.org/documentation/

--------------

Support and Issues
------------------

If you encounter issues not covered in this guide:

1. **Check Build Output:** Read the complete CMake and build output carefully
2. **Verify Prerequisites:** Ensure all required software and SDKs are installed
3. **Clean Build:** Try removing the build directory and building from scratch
4. **Check Versions:** Ensure SDK versions are compatible
5. **Review CMakeLists.txt:** Check if paths and versions match your setup

--------------

| **Document Version:** 1.1
| **Last Updated:** January 2026
| **Tested With:**
  - Visual Studio 2022 (17.8)
  - CMake 3.28.0
  - CUDA Toolkit 12.9
  - ONNX Runtime 1.23.2
  - TensorRT RTX 1.3.1.9
  - vcpkg (baseline c82f746)

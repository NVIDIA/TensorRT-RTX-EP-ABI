@echo off
setlocal enabledelayedexpansion

REM ============================================================================
REM Build Script for TensorRT RTX Execution Provider
REM ============================================================================
REM This script builds the TensorRT RTX Execution Provider library
REM
REM Usage:
REM   build.bat --cuda_home <PATH> (--trt_rtx_home <PATH> ^| --trt_rtx_url <URL>) [options]
REM
REM Build Actions (can be combined, executed in order: clean -> update -> build):
REM   (no flags)      - Full build: clean + update + build
REM   --clean         - Clean the build directory
REM   --update        - Run CMake configuration
REM   --build         - Compile the project
REM
REM Example:
REM   build.bat --cuda_home "C:\CUDA" --onnxruntime_home "C:\onnx" --trt_rtx_home "C:\TRT"
REM   build.bat --cuda_home "C:\CUDA" --onnxruntime_home "C:\onnx" --trt_rtx_home "C:\TRT" --build
REM   build.bat --cuda_home "C:\CUDA" --onnxruntime_home "C:\onnx" --trt_rtx_home "C:\TRT" --clean --update --build
REM
REM Arguments can be provided in any order.
REM ============================================================================

REM Initialize variables
set "CUDA_TOOLKIT_PATH="
set "ONNXRUNTIME_ROOT="
set "TRT_RTX_ROOT="
set "TRT_RTX_DOWNLOAD_URL="
set "BUILD_DIR=build"
set "BUILD_CONFIG=Release"
set "DO_CLEAN=0"
set "DO_UPDATE=0"
set "DO_BUILD=0"
set "DO_PRODUCTION=0"
set "TRT_RTX_EP_VERSION="
set "FLAGS_SPECIFIED=0"
set "BUILD_FAILED=0"
set "USE_VCPKG=OFF"
set "ARCH=x64"
set "VCPKG_TARGET_TRIPLET="
set "VCPKG_HOST_TRIPLET="
set "VCPKG_TOOLCHAIN_FILE="
set "DO_BUILD_WHEEL=0"
set "WHEEL_OUTPUT_DIR="

REM Parse named arguments
:parse_args
if "%~1"=="" goto :check_args

if /i "%~1"=="--cuda_home" (
    set "CUDA_TOOLKIT_PATH=%~2"
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="--onnxruntime_home" (
    set "ONNXRUNTIME_ROOT=%~2"
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="--trt_rtx_home" (
    set "TRT_RTX_ROOT=%~2"
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="--trt_rtx_url" (
    set "TRT_RTX_DOWNLOAD_URL=%~2"
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="--build_dir" (
    set "BUILD_DIR=%~2"
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="--config" (
    set "BUILD_CONFIG=%~2"
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="--clean" (
    set "DO_CLEAN=1"
    set "FLAGS_SPECIFIED=1"
    shift
    goto :parse_args
)
if /i "%~1"=="--update" (
    set "DO_UPDATE=1"
    set "FLAGS_SPECIFIED=1"
    shift
    goto :parse_args
)
if /i "%~1"=="--build" (
    set "DO_BUILD=1"
    set "FLAGS_SPECIFIED=1"
    shift
    goto :parse_args
)
if /i "%~1"=="--production" (
    set "DO_PRODUCTION=1"
    shift
    goto :parse_args
)
if /i "%~1"=="--use_vcpkg" (
    set "USE_VCPKG=ON"
    set "VCPKG_TARGET_TRIPLET=x64-windows-static-md"
    set "VCPKG_HOST_TRIPLET=x64-windows"
    set "VCPKG_TOOLCHAIN_FILE=..\vcpkg\scripts\buildsystems\vcpkg.cmake"
    shift
    goto :parse_args
)
if /i "%~1"=="--version" (
    set "TRT_RTX_EP_VERSION=%~2"
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="--build_wheel" (
    set "DO_BUILD_WHEEL=1"
    shift
    goto :parse_args
)
if /i "%~1"=="--wheel_dir" (
    set "WHEEL_OUTPUT_DIR=%~2"
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="-h" goto :usage
if /i "%~1"=="--help" goto :usage
if /i "%~1"=="/?" goto :usage

echo ERROR: Unknown argument: %~1
goto :usage

:check_args
REM Check if all arguments are provided
if "%CUDA_TOOLKIT_PATH%"=="" (
    echo ERROR: CUDA Toolkit path is required! Use --cuda_home ^<path^>
    goto :usage
)

if "%TRT_RTX_ROOT%"=="" if "%TRT_RTX_DOWNLOAD_URL%"=="" (
    echo ERROR: TensorRT RTX SDK source is required!
    echo Use --trt_rtx_home ^<path^> or --trt_rtx_url ^<url^>
    goto :usage
)

REM Production builds require a version
if "%DO_PRODUCTION%"=="1" (
    if "%TRT_RTX_EP_VERSION%"=="" (
        echo ERROR: --production requires --version ^<M.m.p^>
        echo Example: build.bat --production --version 1.2.3 ...
        exit /b 1
    )
)

REM Auto-derive version from release branch name (e.g. rel-0.1 -> 0.1.0) when not
REM explicitly provided. This populates TRT_RTX_EP_VERSION for both the CMake build
REM and the wheel build so the version is stamped consistently in one place.
if "%TRT_RTX_EP_VERSION%"=="" (
    set "_BRANCH="
    for /f "delims=" %%B in ('git rev-parse --abbrev-ref HEAD 2^>nul') do set "_BRANCH=%%B"
    if "!_BRANCH:~0,4!"=="rel-" (
        set "_BRANCH_VER=!_BRANCH:rel-=!"
        for /f "tokens=1,2 delims=." %%M in ("!_BRANCH_VER!") do (
            if not "%%M"=="" if not "%%N"=="" set "TRT_RTX_EP_VERSION=%%M.%%N.0"
        )
    )
)

REM If no flags specified, do full build (clean + update + build)
if "%FLAGS_SPECIFIED%"=="0" (
    set "DO_CLEAN=1"
    set "DO_UPDATE=1"
    set "DO_BUILD=1"
)

REM Determine platform tag for wheel filename
set "PLATFORM_TAG=win_amd64"

REM Wheel build requires the C++ DLL; auto-enable --build if caller forgot it
if "%DO_BUILD_WHEEL%"=="1" (
    if "%DO_BUILD%"=="0" if "%FLAGS_SPECIFIED%"=="1" (
        echo [INFO] --build_wheel requires --build; enabling automatically.
        set "DO_BUILD=1"
    )
)

REM Validate build configuration
if /i not "%BUILD_CONFIG%"=="Debug" if /i not "%BUILD_CONFIG%"=="Release" if /i not "%BUILD_CONFIG%"=="RelWithDebInfo" (
    echo ERROR: Invalid build configuration: %BUILD_CONFIG%
    echo Valid options are: Debug, Release, RelWithDebInfo
    exit /b 1
)

REM Validate paths exist
if not exist "%CUDA_TOOLKIT_PATH%" (
    echo ERROR: CUDA Toolkit path does not exist: %CUDA_TOOLKIT_PATH%
    exit /b 1
)

if not "%ONNXRUNTIME_ROOT%"=="" if not exist "%ONNXRUNTIME_ROOT%" (
    echo ERROR: ONNX Runtime SDK root path does not exist: %ONNXRUNTIME_ROOT%
    exit /b 1
)

if not "%TRT_RTX_ROOT%"=="" if not exist "%TRT_RTX_ROOT%" (
    echo ERROR: TensorRT RTX SDK root path does not exist: %TRT_RTX_ROOT%
    exit /b 1
)

REM Store source directory (where CMakeLists.txt is located)
REM Note: %~dp0 includes a trailing backslash which can escape quotes in CMake commands
REM Adding a dot normalizes the path and removes the trailing backslash
set "SOURCE_DIR=%~dp0."

REM Build actions string for display
set "ACTIONS="
if "%DO_CLEAN%"=="1" set "ACTIONS=clean"
if "%DO_UPDATE%"=="1" (
    if defined ACTIONS (set "ACTIONS=%ACTIONS% + update") else (set "ACTIONS=update")
)
if "%DO_BUILD%"=="1" (
    if defined ACTIONS (set "ACTIONS=%ACTIONS% + build") else (set "ACTIONS=build")
)
if "%DO_BUILD_WHEEL%"=="1" (
    if defined ACTIONS (set "ACTIONS=%ACTIONS% + wheel") else (set "ACTIONS=wheel")
)

echo ============================================================================
echo Build Configuration:
echo   CUDA Toolkit:        %CUDA_TOOLKIT_PATH%
if "%ONNXRUNTIME_ROOT%"=="" (
    echo   ONNX Runtime SDK:    download 1.26.0
) else (
    echo   ONNX Runtime SDK:    %ONNXRUNTIME_ROOT%
)
if "%TRT_RTX_ROOT%"=="" (
    echo   TensorRT RTX SDK:    %TRT_RTX_DOWNLOAD_URL%
) else (
    echo   TensorRT RTX SDK:    %TRT_RTX_ROOT%
)
echo   Build Directory:     %BUILD_DIR%
echo   Build Config:        %BUILD_CONFIG%
echo   Source Directory:    %SOURCE_DIR%
echo   Actions:             %ACTIONS%
if "%DO_PRODUCTION%"=="1" (
echo   Production Build:    ENABLED ^(signature verification ON^)
echo   Version:             %TRT_RTX_EP_VERSION%
) else (
echo   Production Build:    DISABLED ^(test build, no signature verification^)
if not "%TRT_RTX_EP_VERSION%"=="" (
echo   Version:             %TRT_RTX_EP_VERSION%
) else (
echo   Version:             0.0.0 ^(default^)
)
)
echo   Target Architecture: %ARCH%
if "%DO_BUILD_WHEEL%"=="1" (
    if "%WHEEL_OUTPUT_DIR%"=="" (
        echo   Wheel Output:        %BUILD_DIR%\dist
    ) else (
        echo   Wheel Output:        %WHEEL_OUTPUT_DIR%
    )
)
echo ============================================================================
echo.

REM ============================================================================
REM Step 1: CLEAN (if requested)
REM ============================================================================
if "%DO_CLEAN%"=="1" (
    if exist "%BUILD_DIR%" (
        echo [CLEAN] Removing build directory: %BUILD_DIR%
        rmdir /s /q "%BUILD_DIR%"
        echo [CLEAN] Done.
        echo.
    ) else (
        echo [CLEAN] Build directory does not exist, nothing to clean.
        echo.
    )
)

REM ============================================================================
REM Step 2: UPDATE / CMake Configure (if requested)
REM ============================================================================
if "%DO_UPDATE%"=="1" (
    REM Create build directory if it doesn't exist
    if not exist "%BUILD_DIR%" (
        echo [UPDATE] Creating build directory: %BUILD_DIR%
        mkdir "%BUILD_DIR%" 2>nul
        if not exist "%BUILD_DIR%" (
            echo ERROR: Failed to create build directory: %BUILD_DIR%
            echo Please check if the path is valid and you have write permissions.
            exit /b 1
        )
    )

REM ============================================================================
REM Step 2.5: Install vcpkg (if requested)
REM ============================================================================
if "%USE_VCPKG%"=="ON" (
    REM Clone vcpkg
    if not exist "vcpkg" (
        git clone https://github.com/microsoft/vcpkg.git
        if !ERRORLEVEL! NEQ 0 (
            echo ERROR: Failed to clone vcpkg.
            exit /b 1
        )
        REM Init vcpkg
        pushd "vcpkg"
        call .\bootstrap-vcpkg.bat
        if !ERRORLEVEL! NEQ 0 (
            echo ERROR: Failed to bootstrap vcpkg.
            popd
            exit /b 1
        )
        REM Return to root
        popd
    )  
)

    
    cd /d "%BUILD_DIR%"
    
    echo [UPDATE] Configuring project with CMake...
    if "%DO_PRODUCTION%"=="1" (
        set "PRODUCTION_FLAG=-DTRT_RTX_EP_PRODUCTION_BUILD=ON"
    ) else (
        set "PRODUCTION_FLAG=-DTRT_RTX_EP_PRODUCTION_BUILD=OFF"
    )
    if not "%TRT_RTX_EP_VERSION%"=="" (
        set "VERSION_FLAG=-DTRT_RTX_EP_VERSION=%TRT_RTX_EP_VERSION%"
    ) else (
        set "VERSION_FLAG="
    )
    cmake -G "Visual Studio 17 2022" -A %ARCH% ^
          -DCUDAToolkit_ROOT="%CUDA_TOOLKIT_PATH%" ^
          -DONNXRUNTIME_ROOT="%ONNXRUNTIME_ROOT%" ^
          -DTRT_RTX_ROOT="%TRT_RTX_ROOT%" ^
          -DTRT_RTX_DOWNLOAD_URL="%TRT_RTX_DOWNLOAD_URL%" ^
          -DUSE_VCPKG="%USE_VCPKG%" ^
          -DCMAKE_TOOLCHAIN_FILE=%VCPKG_TOOLCHAIN_FILE% ^
          -DVCPKG_TARGET_TRIPLET=%VCPKG_TARGET_TRIPLET% ^
          -DVCPKG_HOST_TRIPLET=%VCPKG_HOST_TRIPLET% ^
          !PRODUCTION_FLAG! ^
          !VERSION_FLAG! ^
          "%SOURCE_DIR%"
    
    if !ERRORLEVEL! NEQ 0 (
        echo.
        echo ERROR: CMake configuration failed!
        set "BUILD_FAILED=1"
        cd /d "%SOURCE_DIR%"
        goto :end
    )
    echo [UPDATE] Done.
    echo.
    
    cd /d "%SOURCE_DIR%"
)

REM ============================================================================
REM Step 3: BUILD (if requested)
REM ============================================================================
if "%DO_BUILD%"=="1" (
    REM Check if build directory exists
    if not exist "%BUILD_DIR%" (
        echo ERROR: Build directory does not exist: %BUILD_DIR%
        echo Please run with --update first to configure CMake.
        set "BUILD_FAILED=1"
        goto :end
    )
    
    REM Check if CMake was configured (CMakeCache.txt exists)
    if not exist "%BUILD_DIR%\CMakeCache.txt" (
        echo ERROR: Not a CMake build directory ^(missing CMakeCache.txt^)
        echo Please run with --update first to configure CMake.
        set "BUILD_FAILED=1"
        goto :end
    )
    
    cd /d "%BUILD_DIR%"
    
    echo [BUILD] Building project with parallel compilation ^(%BUILD_CONFIG%^)...
    cmake --build . --config %BUILD_CONFIG% --parallel
    
    if !ERRORLEVEL! NEQ 0 (
        echo.
        echo ERROR: Build failed!
        set "BUILD_FAILED=1"
        cd /d "%SOURCE_DIR%"
        goto :end
    )
    echo [BUILD] Done.
    echo.
    
    cd /d "%SOURCE_DIR%"
)

REM ============================================================================
REM Step 4: BUILD PYTHON WHEEL (if requested)
REM ============================================================================
REM Run the wheel build at top-level (not inside a giant `(...)` block) — cmd.exe's
REM delayed expansion silently breaks inside long nested paren blocks, causing
REM `!VAR!` to evaluate literally and `if !ERRORLEVEL! NEQ 0` to always fire.
if not "%DO_BUILD_WHEEL%"=="1" goto :end_wheel
    REM Locate python/ — prefer inside trt-rtx-ep-abi/ (final repo layout),
    REM fall back to sibling directory (workspace layout during development)
    set "PYTHON_DIR="
    if exist "%SOURCE_DIR%\python\pyproject.toml" set "PYTHON_DIR=%SOURCE_DIR%\python"
    if not defined PYTHON_DIR if exist "%SOURCE_DIR%\..\python\pyproject.toml" set "PYTHON_DIR=%SOURCE_DIR%\..\python"
    if not defined PYTHON_DIR (
        echo ERROR: Cannot find python\ directory. Expected at %SOURCE_DIR%\python or %SOURCE_DIR%\..\python
        set "BUILD_FAILED=1"
        goto :end
    )
    set "EP_DLL=%SOURCE_DIR%\%BUILD_DIR%\%BUILD_CONFIG%\onnxruntime_providers_nv_tensorrt_rtx.dll"
    if "%TRT_RTX_EP_VERSION%"=="" (
        set "WHEEL_VERSION=0.0.0"
    ) else (
        set "WHEEL_VERSION=%TRT_RTX_EP_VERSION%"
    )
    if "%WHEEL_OUTPUT_DIR%"=="" set "WHEEL_OUTPUT_DIR=%~dp0%BUILD_DIR%\dist"

    REM Verify EP DLL was produced
    if not exist "%EP_DLL%" (
        echo ERROR: EP DLL not found at %EP_DLL%
        echo Run with --build ^(or no flags for full build^) before --build_wheel.
        set "BUILD_FAILED=1"
        goto :end
    )

    REM Require python in PATH. Use `call` so a pyenv-style python.bat shim
    REM returns control to this script instead of replacing it.
    call python --version >nul 2>&1
    set "_PY_RC=%ERRORLEVEL%"
    if not "%_PY_RC%"=="0" (
        echo ERROR: 'python' not found in PATH. Add Python to PATH and retry.
        echo Note: 'python -m build' must also be available ^(pip install build^).
        set "BUILD_FAILED=1"
        goto :end
    )

    REM Stage DLLs into package dir
    REM TRT RTX SDK layout: DLLs are in bin\, not lib\ (lib\ contains .lib import files)
    REM CUDA runtime: prefer bin\x64 (CUDA 12+), fall back to bin\ (older layouts).
    echo [WHEEL] Staging DLLs into package directory...
    set "CUDA_BIN_ARG="
    if exist "%CUDA_TOOLKIT_PATH%\bin\x64" (
        set "CUDA_BIN_ARG=--cuda-bin "%CUDA_TOOLKIT_PATH%\bin\x64""
    )
    if not defined CUDA_BIN_ARG if exist "%CUDA_TOOLKIT_PATH%\bin" (
        set "CUDA_BIN_ARG=--cuda-bin "%CUDA_TOOLKIT_PATH%\bin""
    )
    if not defined CUDA_BIN_ARG (
        echo [WHEEL] Warning: CUDA bin directory not found under %CUDA_TOOLKIT_PATH%; cudart will not be bundled.
    )
    call python "%PYTHON_DIR%\scripts\stage_windows_dlls.py" ^
        --ep-dll "%EP_DLL%" ^
        --trt-lib-dir "%TRT_RTX_ROOT%\bin" ^
        %CUDA_BIN_ARG%
    set "_STAGE_RC=%ERRORLEVEL%"
    if not "%_STAGE_RC%"=="0" (
        echo ERROR: DLL staging failed.
        echo If TRT RTX DLLs are not in %TRT_RTX_ROOT%\bin, set NV_TRT_RTX_LIB_DIR to the correct path.
        set "BUILD_FAILED=1"
        goto :end
    )

    REM Detect CUDA major version from toolkit path for package naming (e.g. v13.2 -> cu13)
    set "NV_CUDA_MAJOR="
    echo import re, sys > "%TEMP%\_cuda_ver.py"
    echo p = sys.argv[1] >> "%TEMP%\_cuda_ver.py"
    echo m = re.search^(r'[-v]^(\d+^)\.', p^) >> "%TEMP%\_cuda_ver.py"
    echo print^(m.group^(1^) if m else '', end=''^) >> "%TEMP%\_cuda_ver.py"
    for /f "usebackq delims=" %%m in (`python "%TEMP%\_cuda_ver.py" "%CUDA_TOOLKIT_PATH%" 2^>nul`) do set "NV_CUDA_MAJOR=%%m"
    del "%TEMP%\_cuda_ver.py" 2>nul
    if not "%NV_CUDA_MAJOR%"=="" (
        echo [WHEEL] CUDA major: %NV_CUDA_MAJOR% ^(package: onnxruntime-ep-nv-tensorrt-rtx-cu%NV_CUDA_MAJOR%^)
    ) else (
        echo [WHEEL] Warning: Could not detect CUDA major from path; package will use base name.
    )

    REM Write _version.py ^(overwrite; gitignored^)
    echo [WHEEL] Writing _version.py ^(version %WHEEL_VERSION%^)...
    > "%PYTHON_DIR%\onnxruntime_ep_nv_tensorrt_rtx\_version.py" echo __version__ = "%WHEEL_VERSION%"

    REM Clean stale intermediate build artefacts so DLLs from a previous staging
    REM run do not leak into the new wheel (build_py adds files but never removes them).
    if exist "%PYTHON_DIR%\build" (
        echo [WHEEL] Removing stale Python build cache: %PYTHON_DIR%\build
        rmdir /s /q "%PYTHON_DIR%\build"
    )

    REM Build wheel
    echo [WHEEL] Building Python wheel...
    call python -m build --wheel --no-isolation --outdir "%WHEEL_OUTPUT_DIR%" "%PYTHON_DIR%"
    set "_WHEEL_RC=%ERRORLEVEL%"
    if not "%_WHEEL_RC%"=="0" (
        echo ERROR: Wheel build failed.
        echo Ensure 'pip install build' has been run once.
        set "BUILD_FAILED=1"
        goto :end
    )
    echo [WHEEL] Done.
    echo.

    REM Build meta wheel (onnxruntime-ep-nv-tensorrt-rtx -> cu13 dependency)
    set "META_DIR=%PYTHON_DIR%\meta"
    if not exist "%META_DIR%\pyproject.toml" (
        echo [META] Warning: meta\ directory not found at %META_DIR%; skipping meta wheel.
        goto :end_wheel
    )
    echo [META] Writing meta\_version.txt ^(version %WHEEL_VERSION%^)...
    > "%META_DIR%\_version.txt" echo %WHEEL_VERSION%
    echo [META] Building meta wheel ^(onnxruntime-ep-nv-tensorrt-rtx^)...
    call python -m build --wheel --no-isolation --outdir "%WHEEL_OUTPUT_DIR%" "%META_DIR%"
    set "_META_RC=%ERRORLEVEL%"
    if not "%_META_RC%"=="0" (
        echo ERROR: Meta wheel build failed.
        set "BUILD_FAILED=1"
        goto :end
    )
    echo [META] Done.
    echo.
:end_wheel

:end
echo ============================================================================
if "%BUILD_FAILED%"=="1" (
    echo Build FAILED!
    echo ============================================================================
    exit /b 1
)
echo Completed successfully!
if "%DO_BUILD%"=="1" (
    echo Output: %BUILD_DIR%\%BUILD_CONFIG%\onnxruntime_providers_nv_tensorrt_rtx.dll
)
if "%DO_BUILD_WHEEL%"=="1" (
    set "_wpkg=onnxruntime_ep_nv_tensorrt_rtx"
    if not "!NV_CUDA_MAJOR!"=="" set "_wpkg=onnxruntime_ep_nv_tensorrt_rtx_cu!NV_CUDA_MAJOR!"
    set "_wver=%TRT_RTX_EP_VERSION%"
    if "!_wver!"=="" set "_wver=0.0.0"
    echo Wheel:  %WHEEL_OUTPUT_DIR%\!_wpkg!-!_wver!-py3-none-%PLATFORM_TAG%.whl
    echo Meta:   %WHEEL_OUTPUT_DIR%\onnxruntime_ep_nv_tensorrt_rtx-!_wver!-py3-none-any.whl
)
echo ============================================================================
exit /b 0

:usage
echo.
echo Usage: build.bat --cuda_home ^<PATH^> ^(--trt_rtx_home ^<PATH^> ^| --trt_rtx_url ^<URL^>^) [options]
echo.
echo Required Arguments:
echo   --cuda_home ^<PATH^>         Path to CUDA Toolkit installation
echo   --trt_rtx_home ^<PATH^>      Path to TensorRT RTX SDK root directory, or:
echo   --trt_rtx_url ^<URL^>        Full TensorRT RTX SDK archive URL
echo.
echo Optional Arguments:
echo   --onnxruntime_home ^<PATH^>  ONNX Runtime SDK root ^(downloads 1.26.0 when omitted^)
echo   --build_dir ^<PATH^>         Build output directory (default: build)
echo   --config ^<TYPE^>            Build configuration (default: Release)
echo                              Options: Debug, Release, RelWithDebInfo
echo   --clean                    Clean the build directory
echo   --update                   Run CMake configuration
echo   --build                    Compile the project
echo   --version ^<M.m.p^>          Set EP version (e.g. 1.2.3). Required for --production
echo   --production               Enable production build with signature verification
echo   --use_vcpkg                Use VCPKG package manager

echo   --build_wheel              Build Python wheel after C++ DLL build
echo   --wheel_dir ^<PATH^>         Wheel output directory (default: ^<build_dir^>\dist)
echo                              Prerequisite: pip install build
echo   -h, --help, /?             Show this help message
echo.
echo Build Actions (can be combined, executed in order: clean -^> update -^> build):
echo   (no flags)                 Full build: clean + update + build
echo   --clean                    Only clean the build directory
echo   --update                   Only run CMake configuration
echo   --build                    Only compile (requires prior --update)
echo   --clean --update           Clean and reconfigure
echo   --update --build           Reconfigure and build (no clean)
echo   --clean --update --build   Full build (same as no flags)
echo.
echo Build Types:
echo   (default)                  Test build - signature verification disabled
echo   --production               Production build - NVIDIA signature verification enabled
echo.
echo Examples:
echo   Full clean build (default):
echo     build.bat --cuda_home "C:\CUDA" --onnxruntime_home "C:\onnx" --trt_rtx_home "C:\TRT"
echo.
echo   Incremental build only (after code changes):
echo     build.bat --cuda_home "C:\CUDA" --onnxruntime_home "C:\onnx" --trt_rtx_home "C:\TRT" --build
echo.
echo   Reconfigure and build (after CMakeLists.txt changes):
echo     build.bat --cuda_home "C:\CUDA" --onnxruntime_home "C:\onnx" --trt_rtx_home "C:\TRT" --update --build
echo.
echo   Clean and reconfigure only (no build):
echo     build.bat --cuda_home "C:\CUDA" --onnxruntime_home "C:\onnx" --trt_rtx_home "C:\TRT" --clean --update
echo.
echo   With custom build directory:
echo     build.bat --cuda_home "C:\CUDA" --onnxruntime_home "C:\onnx" --trt_rtx_home "C:\TRT" --build_dir "C:\out"
echo.
echo   Debug build with symbols:
echo     build.bat --cuda_home "C:\CUDA" --onnxruntime_home "C:\onnx" --trt_rtx_home "C:\TRT" --config Debug
echo.
echo   Release build with debug info:
echo     build.bat --cuda_home "C:\CUDA" --onnxruntime_home "C:\onnx" --trt_rtx_home "C:\TRT" --config RelWithDebInfo
echo.
echo   Production build (with signature verification and version):
echo     build.bat --cuda_home "C:\CUDA" --onnxruntime_home "C:\onnx" --trt_rtx_home "C:\TRT" --production --version 1.2.3
echo.
echo   Full build + Python wheel:
echo     build.bat --cuda_home "C:\CUDA" --onnxruntime_home "C:\onnx" --trt_rtx_home "C:\TRT" --version 1.5.0 --build_wheel
echo.
echo   Incremental C++ build + wheel (no clean/reconfigure):
echo     build.bat --cuda_home "C:\CUDA" --onnxruntime_home "C:\onnx" --trt_rtx_home "C:\TRT" --build --build_wheel --version 1.5.0
echo.
echo Arguments can be provided in any order.
echo.
exit /b 1

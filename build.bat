@echo off
setlocal enabledelayedexpansion

REM ============================================================================
REM Build Script for TensorRT RTX Execution Provider
REM ============================================================================
REM This script builds the TensorRT RTX Execution Provider library
REM
REM Usage:
REM   build.bat --cuda_home <PATH> --onnxruntime_home <PATH> --trt_rtx_home <PATH> [--version <M.m.p>] [options]
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

if "%ONNXRUNTIME_ROOT%"=="" (
    echo ERROR: ONNX Runtime SDK root path is required! Use --onnxruntime_home ^<path^>
    goto :usage
)

if "%TRT_RTX_ROOT%"=="" (
    echo ERROR: TensorRT RTX SDK root path is required! Use --trt_rtx_home ^<path^>
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

REM If no flags specified, do full build (clean + update + build)
if "%FLAGS_SPECIFIED%"=="0" (
    set "DO_CLEAN=1"
    set "DO_UPDATE=1"
    set "DO_BUILD=1"
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

if not exist "%ONNXRUNTIME_ROOT%" (
    echo ERROR: ONNX Runtime SDK root path does not exist: %ONNXRUNTIME_ROOT%
    exit /b 1
)

if not exist "%TRT_RTX_ROOT%" (
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

echo ============================================================================
echo Build Configuration:
echo   CUDA Toolkit:        %CUDA_TOOLKIT_PATH%
echo   ONNX Runtime SDK:    %ONNXRUNTIME_ROOT%
echo   TensorRT RTX SDK:    %TRT_RTX_ROOT%
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
echo ============================================================================
exit /b 0

:usage
echo.
echo Usage: build.bat --cuda_home ^<PATH^> --onnxruntime_home ^<PATH^> --trt_rtx_home ^<PATH^> [--version ^<M.m.p^>] [options]
echo.
echo Required Arguments:
echo   --cuda_home ^<PATH^>         Path to CUDA Toolkit installation
echo   --onnxruntime_home ^<PATH^>  Path to ONNX Runtime SDK root directory
echo   --trt_rtx_home ^<PATH^>      Path to TensorRT RTX SDK root directory
echo.
echo Optional Arguments:
echo   --build_dir ^<PATH^>         Build output directory (default: build)
echo   --config ^<TYPE^>            Build configuration (default: Release)
echo                              Options: Debug, Release, RelWithDebInfo
echo   --clean                    Clean the build directory
echo   --update                   Run CMake configuration
echo   --build                    Compile the project
echo   --version ^<M.m.p^>          Set EP version (e.g. 1.2.3). Required for --production
echo   --production               Enable production build with signature verification
echo   --use_vcpkg                Use VCPKG package manager
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
echo Arguments can be provided in any order.
echo.
exit /b 1

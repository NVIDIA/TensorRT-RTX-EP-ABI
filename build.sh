#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ============================================================================
# Build Script for TensorRT RTX Execution Provider (Linux)
# ============================================================================
# Mirrors build.bat — same flags, same default behaviour.
#
# Usage:
#   build.sh --cuda_home <PATH> (--trt_rtx_home <PATH> | --trt_rtx_url <URL>) [options]
#
# Build Actions (can be combined, executed in order: clean -> update -> build):
#   (no flags)      - Full build: clean + update + build
#   --clean         - Clean the build directory
#   --update        - Run CMake configuration
#   --build         - Compile the project
#
# Example:
#   ./build.sh --cuda_home /usr/local/cuda --onnxruntime_home ~/ORT --trt_rtx_home ~/TRT-RTX
#   ./build.sh --cuda_home /usr/local/cuda --onnxruntime_home ~/ORT --trt_rtx_home ~/TRT-RTX --build
#   ./build.sh --cuda_home /usr/local/cuda --onnxruntime_home ~/ORT --trt_rtx_home ~/TRT-RTX --version 1.4.0 --build_wheel
# ============================================================================

set -euo pipefail

# ----------------------------------------------------------------------------
# Defaults
# ----------------------------------------------------------------------------
CUDA_TOOLKIT_PATH=""
ONNXRUNTIME_ROOT=""
TRT_RTX_ROOT=""
TRT_RTX_DOWNLOAD_URL=""
BUILD_DIR="build"
BUILD_CONFIG="Release"
DO_CLEAN=0
DO_UPDATE=0
DO_BUILD=0
DO_PRODUCTION=0
TRT_RTX_EP_VERSION=""
FLAGS_SPECIFIED=0
DO_BUILD_WHEEL=0
WHEEL_OUTPUT_DIR=""

SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ----------------------------------------------------------------------------
# Usage
# ----------------------------------------------------------------------------
usage() {
    cat <<EOF

Usage: build.sh --cuda_home <PATH> (--trt_rtx_home <PATH> | --trt_rtx_url <URL>) [options]

Required Arguments:
  --cuda_home <PATH>         Path to CUDA Toolkit installation
  --trt_rtx_home <PATH>      Path to TensorRT RTX SDK root directory, or:
  --trt_rtx_url <URL>        Full TensorRT RTX SDK archive URL

Optional Arguments:
  --onnxruntime_home <PATH>  ONNX Runtime SDK root (downloads 1.26.0 when omitted)
  --build_dir <PATH>         Build output directory (default: build)
  --config <TYPE>            Build configuration (default: Release)
                             Options: Debug, Release, RelWithDebInfo
  --clean                    Clean the build directory
  --update                   Run CMake configuration
  --build                    Compile the project
  --version <M.m.p>          Set EP version (e.g. 1.4.0). Required for --production
  --production               Enable production build with signature verification
  --build_wheel              Build Python wheel after C++ SO build
  --wheel_dir <PATH>         Wheel output directory (default: <build_dir>/dist)
                             Prerequisite: pip install build
  -h, --help                 Show this help message

Build Actions (can be combined, executed in order: clean -> update -> build):
  (no flags)                 Full build: clean + update + build
  --clean                    Only clean the build directory
  --update                   Only run CMake configuration
  --build                    Only compile (requires prior --update)
  --clean --update           Clean and reconfigure
  --update --build           Reconfigure and build (no clean)
  --clean --update --build   Full build (same as no flags)

Examples:
  Full clean build (default):
    ./build.sh --cuda_home /usr/local/cuda-12.9 --onnxruntime_home ~/ORT --trt_rtx_home ~/TRT-RTX

  Incremental build only (after code changes):
    ./build.sh --cuda_home /usr/local/cuda-12.9 --onnxruntime_home ~/ORT --trt_rtx_home ~/TRT-RTX --build

  Reconfigure and build:
    ./build.sh --cuda_home /usr/local/cuda-12.9 --onnxruntime_home ~/ORT --trt_rtx_home ~/TRT-RTX --update --build

  Full build + Python wheel:
    ./build.sh --cuda_home /usr/local/cuda-12.9 --onnxruntime_home ~/ORT --trt_rtx_home ~/TRT-RTX --version 1.4.0 --build_wheel

  Incremental C++ build + wheel:
    ./build.sh --cuda_home /usr/local/cuda-12.9 --onnxruntime_home ~/ORT --trt_rtx_home ~/TRT-RTX --build --build_wheel --version 1.4.0

EOF
    exit 1
}

# ----------------------------------------------------------------------------
# Parse arguments
# ----------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --cuda_home)        CUDA_TOOLKIT_PATH="$2"; shift 2 ;;
        --onnxruntime_home) ONNXRUNTIME_ROOT="$2";  shift 2 ;;
        --trt_rtx_home)     TRT_RTX_ROOT="$2";      shift 2 ;;
        --trt_rtx_url)      TRT_RTX_DOWNLOAD_URL="$2"; shift 2 ;;
        --build_dir)        BUILD_DIR="$2";          shift 2 ;;
        --config)           BUILD_CONFIG="$2";       shift 2 ;;
        --clean)            DO_CLEAN=1; FLAGS_SPECIFIED=1; shift ;;
        --update)           DO_UPDATE=1; FLAGS_SPECIFIED=1; shift ;;
        --build)            DO_BUILD=1;  FLAGS_SPECIFIED=1; shift ;;
        --production)       DO_PRODUCTION=1;         shift ;;
        --version)          TRT_RTX_EP_VERSION="$2"; shift 2 ;;
        --build_wheel)      DO_BUILD_WHEEL=1;        shift ;;
        --wheel_dir)        WHEEL_OUTPUT_DIR="$2";   shift 2 ;;
        -h|--help)          usage ;;
        *) echo "ERROR: Unknown argument: $1"; usage ;;
    esac
done

# ----------------------------------------------------------------------------
# Validate required arguments
# ----------------------------------------------------------------------------
if [[ -z "$CUDA_TOOLKIT_PATH" ]]; then
    echo "ERROR: CUDA Toolkit path is required! Use --cuda_home <path>"
    usage
fi
if [[ -z "$TRT_RTX_ROOT" && -z "$TRT_RTX_DOWNLOAD_URL" ]]; then
    echo "ERROR: TensorRT RTX SDK source is required!"
    echo "Use --trt_rtx_home <path> or --trt_rtx_url <url>"
    usage
fi

if [[ "$DO_PRODUCTION" -eq 1 && -z "$TRT_RTX_EP_VERSION" ]]; then
    echo "ERROR: --production requires --version <M.m.p>"
    echo "Example: build.sh --production --version 1.4.0 ..."
    exit 1
fi

# Auto-derive version from release branch name (e.g. rel-0.1 -> 0.1.0) when not
# explicitly provided. This populates TRT_RTX_EP_VERSION for both the CMake build
# and the wheel build so the version is stamped consistently in one place.
if [[ -z "$TRT_RTX_EP_VERSION" ]]; then
    _branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "")
    _branch="${_branch%$'\r'}"
    if [[ "$_branch" =~ ^rel-([0-9]+\.[0-9]+)$ ]]; then
        TRT_RTX_EP_VERSION="${BASH_REMATCH[1]}.0"
    fi
fi

# Validate paths exist
if [[ ! -d "$CUDA_TOOLKIT_PATH" ]]; then
    echo "ERROR: CUDA Toolkit path does not exist: $CUDA_TOOLKIT_PATH"
    exit 1
fi
if [[ -n "$ONNXRUNTIME_ROOT" && ! -d "$ONNXRUNTIME_ROOT" ]]; then
    echo "ERROR: ONNX Runtime SDK root path does not exist: $ONNXRUNTIME_ROOT"
    exit 1
fi
if [[ -n "$TRT_RTX_ROOT" && ! -d "$TRT_RTX_ROOT" ]]; then
    echo "ERROR: TensorRT RTX SDK root path does not exist: $TRT_RTX_ROOT"
    exit 1
fi

# Validate build configuration
case "$BUILD_CONFIG" in
    Debug|Release|RelWithDebInfo) ;;
    *) echo "ERROR: Invalid build configuration: $BUILD_CONFIG"
       echo "Valid options are: Debug, Release, RelWithDebInfo"
       exit 1 ;;
esac

# If no flags specified, do full build (clean + update + build)
if [[ "$FLAGS_SPECIFIED" -eq 0 ]]; then
    DO_CLEAN=1
    DO_UPDATE=1
    DO_BUILD=1
fi

# --build_wheel requires --build; auto-enable if caller passed only --build_wheel
if [[ "$DO_BUILD_WHEEL" -eq 1 && "$DO_BUILD" -eq 0 && "$FLAGS_SPECIFIED" -eq 1 ]]; then
    echo "[INFO] --build_wheel requires --build; enabling automatically."
    DO_BUILD=1
fi

# Build actions string for display
ACTIONS=""
[[ "$DO_CLEAN" -eq 1 ]]       && ACTIONS="clean"
[[ "$DO_UPDATE" -eq 1 ]]      && ACTIONS="${ACTIONS:+$ACTIONS + }update"
[[ "$DO_BUILD" -eq 1 ]]       && ACTIONS="${ACTIONS:+$ACTIONS + }build"
[[ "$DO_BUILD_WHEEL" -eq 1 ]] && ACTIONS="${ACTIONS:+$ACTIONS + }wheel"

WHEEL_DISPLAY="${WHEEL_OUTPUT_DIR:-$BUILD_DIR/dist}"

echo "============================================================================"
echo "Build Configuration:"
echo "  CUDA Toolkit:        $CUDA_TOOLKIT_PATH"
echo "  ONNX Runtime SDK:    ${ONNXRUNTIME_ROOT:-download 1.26.0}"
echo "  TensorRT RTX SDK:    ${TRT_RTX_ROOT:-$TRT_RTX_DOWNLOAD_URL}"
echo "  Build Directory:     $BUILD_DIR"
echo "  Build Config:        $BUILD_CONFIG"
echo "  Source Directory:    $SOURCE_DIR"
echo "  Actions:             $ACTIONS"
if [[ "$DO_PRODUCTION" -eq 1 ]]; then
    echo "  Production Build:    ENABLED (signature verification ON)"
    echo "  Version:             $TRT_RTX_EP_VERSION"
else
    echo "  Production Build:    DISABLED (test build, no signature verification)"
    echo "  Version:             ${TRT_RTX_EP_VERSION:-0.0.0 (default)}"
fi
if [[ "$DO_BUILD_WHEEL" -eq 1 ]]; then
    echo "  Wheel Output:        $WHEEL_DISPLAY"
fi
echo "============================================================================"
echo

# ============================================================================
# Step 1: CLEAN
# ============================================================================
if [[ "$DO_CLEAN" -eq 1 ]]; then
    if [[ -d "$BUILD_DIR" ]]; then
        echo "[CLEAN] Removing build directory: $BUILD_DIR"
        rm -rf "$BUILD_DIR"
        echo "[CLEAN] Done."
    else
        echo "[CLEAN] Build directory does not exist, nothing to clean."
    fi
    echo
fi

# ============================================================================
# Step 2: UPDATE / CMake Configure
# ============================================================================
if [[ "$DO_UPDATE" -eq 1 ]]; then
    mkdir -p "$BUILD_DIR"

    PRODUCTION_FLAG="-DTRT_RTX_EP_PRODUCTION_BUILD=OFF"
    [[ "$DO_PRODUCTION" -eq 1 ]] && PRODUCTION_FLAG="-DTRT_RTX_EP_PRODUCTION_BUILD=ON"

    VERSION_FLAG=""
    [[ -n "$TRT_RTX_EP_VERSION" ]] && VERSION_FLAG="-DTRT_RTX_EP_VERSION=$TRT_RTX_EP_VERSION"

    echo "[UPDATE] Configuring project with CMake..."
    cmake -B "$BUILD_DIR" \
          -DCMAKE_BUILD_TYPE="$BUILD_CONFIG" \
          -DCUDAToolkit_ROOT="$CUDA_TOOLKIT_PATH" \
          -DONNXRUNTIME_ROOT="$ONNXRUNTIME_ROOT" \
          -DTRT_RTX_ROOT="$TRT_RTX_ROOT" \
          -DTRT_RTX_DOWNLOAD_URL="$TRT_RTX_DOWNLOAD_URL" \
          "$PRODUCTION_FLAG" \
          ${VERSION_FLAG:+"$VERSION_FLAG"} \
          "$SOURCE_DIR"

    echo "[UPDATE] Done."
    echo
fi

# ============================================================================
# Step 3: BUILD
# ============================================================================
if [[ "$DO_BUILD" -eq 1 ]]; then
    if [[ ! -d "$BUILD_DIR" ]]; then
        echo "ERROR: Build directory does not exist: $BUILD_DIR"
        echo "Please run with --update first to configure CMake."
        exit 1
    fi
    if [[ ! -f "$BUILD_DIR/CMakeCache.txt" ]]; then
        echo "ERROR: Not a CMake build directory (missing CMakeCache.txt)"
        echo "Please run with --update first to configure CMake."
        exit 1
    fi

    echo "[BUILD] Building project with parallel compilation ($BUILD_CONFIG)..."
    cmake --build "$BUILD_DIR" --parallel

    echo "[BUILD] Done."
    echo
fi

# ============================================================================
# Step 4: BUILD PYTHON WHEEL
# ============================================================================
if [[ "$DO_BUILD_WHEEL" -eq 1 ]]; then
    # Locate python/ — prefer inside trt-rtx-ep-abi/ (final repo layout),
    # fall back to sibling directory (workspace layout during development)
    if [[ -f "$SOURCE_DIR/python/pyproject.toml" ]]; then
        PYTHON_DIR="$SOURCE_DIR/python"
    elif [[ -f "$SOURCE_DIR/../python/pyproject.toml" ]]; then
        PYTHON_DIR="$(cd "$SOURCE_DIR/../python" && pwd)"
    else
        echo "ERROR: Cannot find python/ directory. Expected at $SOURCE_DIR/python or $SOURCE_DIR/../python"
        exit 1
    fi

    EP_SO="$SOURCE_DIR/$BUILD_DIR/libonnxruntime_providers_nv_tensorrt_rtx.so"
    WHEEL_VERSION="${TRT_RTX_EP_VERSION:-0.0.0}"
    WHEEL_OUTPUT_DIR="${WHEEL_OUTPUT_DIR:-$SOURCE_DIR/$BUILD_DIR/dist}"

    # Verify EP SO was produced
    if [[ ! -f "$EP_SO" ]]; then
        echo "ERROR: EP shared library not found at $EP_SO"
        echo "Run with --build (or no flags for full build) before --build_wheel."
        exit 1
    fi

    # Require python3.12 in PATH (preferred); fall back to python3
    if command -v python3.12 > /dev/null 2>&1; then
        PYTHON_BIN="python3.12"
    elif command -v python3 > /dev/null 2>&1; then
        PYTHON_BIN="python3"
    else
        echo "ERROR: Neither 'python3.12' nor 'python3' found in PATH."
        echo "Note: 'pip install build' must be available for the chosen interpreter."
        exit 1
    fi
    echo "[WHEEL] Using Python: $PYTHON_BIN ($(${PYTHON_BIN} --version))"

    # Stage SOs into package dir
    # TRT RTX SDK layout: SOs are in lib/
    echo "[WHEEL] Staging SOs into package directory..."
    "$PYTHON_BIN" "$PYTHON_DIR/scripts/stage_linux_so.py" \
        --ep-so "$EP_SO" \
        --trt-lib-dir "$TRT_RTX_ROOT/lib" || {
        echo "ERROR: SO staging failed."
        echo "If TRT RTX SOs are not in $TRT_RTX_ROOT/lib, set NV_TRT_RTX_LIB_DIR to the correct path."
        exit 1
    }

    # Detect CUDA major version from toolkit path for package naming (e.g. cuda-13.1 -> cu13)
    CUDA_MAJOR=$("$PYTHON_BIN" -c "
import re, sys
p = sys.argv[1]
m = re.search(r'[-v](\d+)\.', p)
print(m.group(1) if m else '', end='')
" "$CUDA_TOOLKIT_PATH" 2>/dev/null || echo "")
    if [[ -n "$CUDA_MAJOR" ]]; then
        export NV_CUDA_MAJOR="$CUDA_MAJOR"
        echo "[WHEEL] CUDA major: $CUDA_MAJOR (package: onnxruntime-ep-nv-tensorrt-rtx-cu${CUDA_MAJOR})"
    else
        echo "[WHEEL] Warning: Could not detect CUDA major from path; package will use base name."
    fi

    # Write _version.py (overwrite; gitignored)
    echo "[WHEEL] Writing _version.py (version $WHEEL_VERSION)..."
    echo "__version__ = \"$WHEEL_VERSION\"" > "$PYTHON_DIR/onnxruntime_ep_nv_tensorrt_rtx/_version.py"

    # Clean stale intermediate build artefacts so SOs from a previous staging
    # run do not leak into the new wheel (build_py adds files but never removes them).
    if [[ -d "$PYTHON_DIR/build" ]]; then
        echo "[WHEEL] Removing stale Python build cache: $PYTHON_DIR/build"
        rm -rf "$PYTHON_DIR/build"
    fi

    mkdir -p "$WHEEL_OUTPUT_DIR"

    # Build wheel
    echo "[WHEEL] Building Python wheel..."
    if ! "$PYTHON_BIN" -m build --wheel --no-isolation --outdir "$WHEEL_OUTPUT_DIR" "$PYTHON_DIR"; then
        echo "ERROR: Wheel build failed."
        echo "Ensure 'pip install build' has been run once."
        exit 1
    fi

    echo "[WHEEL] Done."
    echo

    # Build meta wheel (onnxruntime-ep-nv-tensorrt-rtx -> cu13 dependency)
    META_DIR="$PYTHON_DIR/meta"
    if [[ ! -f "$META_DIR/pyproject.toml" ]]; then
        echo "[META] Warning: meta/ directory not found at $META_DIR; skipping meta wheel."
    else
        echo "[META] Writing meta/_version.txt (version $WHEEL_VERSION)..."
        echo "$WHEEL_VERSION" > "$META_DIR/_version.txt"
        echo "[META] Building meta wheel (onnxruntime-ep-nv-tensorrt-rtx)..."
        if ! "$PYTHON_BIN" -m build --wheel --no-isolation --outdir "$WHEEL_OUTPUT_DIR" "$META_DIR"; then
            echo "ERROR: Meta wheel build failed."
            exit 1
        fi
        echo "[META] Done."
        echo
    fi
fi

# ============================================================================
echo "============================================================================"
echo "Completed successfully!"
if [[ "$DO_BUILD" -eq 1 ]]; then
    echo "Output: $BUILD_DIR/libonnxruntime_providers_nv_tensorrt_rtx.so"
fi
if [[ "$DO_BUILD_WHEEL" -eq 1 ]]; then
    _wheel_pkg="onnxruntime_ep_nv_tensorrt_rtx${CUDA_MAJOR:+_cu$CUDA_MAJOR}"
    WHEEL_PLATFORM=$("$PYTHON_BIN" -c \
        "import sysconfig; print(sysconfig.get_platform().replace('-','_').replace('.','_'))" \
        2>/dev/null || echo "linux_x86_64")
    _wver="${WHEEL_VERSION:-0.0.0}"
    echo "Wheel:  $WHEEL_OUTPUT_DIR/${_wheel_pkg}-${_wver}-py3-none-${WHEEL_PLATFORM}.whl"
    echo "Meta:   $WHEEL_OUTPUT_DIR/onnxruntime_ep_nv_tensorrt_rtx-${_wver}-py3-none-any.whl"
fi
echo "============================================================================"

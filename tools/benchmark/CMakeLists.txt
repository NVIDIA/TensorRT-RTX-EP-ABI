#Standalone cached - EP - context benchmark worker.It deliberately uses only
#public ORT APIs so it can compare independently - built provider DLLs.
add_executable(trt_ep_perf_test
        trt_ep_perf_test.cpp
        trt_ep_perf_test_command_line_parser.cpp
        trt_ep_perf_test_memory_monitor.cpp)
add_dependencies(trt_ep_perf_test TensorRTRtxEp)
set_target_properties(trt_ep_perf_test PROPERTIES
        CXX_STANDARD 20
        CXX_EXTENSIONS OFF
        RUNTIME_OUTPUT_DIRECTORY "${TRT_RTX_BIN_DIR}")
target_include_directories(trt_ep_perf_test PRIVATE
        "${CMAKE_SOURCE_DIR}/src")
target_link_libraries(trt_ep_perf_test PRIVATE
        onnxruntime::onnxruntime
        CUDA::cudart)

install(TARGETS trt_ep_perf_test
        RUNTIME DESTINATION bin
        COMPONENT benchmark
        LIBRARY DESTINATION bin
        COMPONENT benchmark
        ARCHIVE DESTINATION lib
        COMPONENT benchmark)
if (TRT_RTX_ORT_RUNTIME_FILES)
    install(FILES ${TRT_RTX_ORT_RUNTIME_FILES} DESTINATION bin COMPONENT benchmark)
endif ()

if (WIN32)
    target_link_libraries(trt_ep_perf_test PRIVATE CUDA::cuda_driver dxgi)
else()
    target_link_libraries(trt_ep_perf_test PRIVATE CUDA::nvml)
endif ()

if (TRT_RTX_EP_ENABLE_NVTX3)
    target_link_libraries(trt_ep_perf_test PRIVATE CUDA::nvtx3)
    target_compile_definitions(trt_ep_perf_test PRIVATE TRT_RTX_EP_ENABLE_NVTX3=1)
else()
    target_compile_definitions(trt_ep_perf_test PRIVATE TRT_RTX_EP_ENABLE_NVTX3=0)
endif ()

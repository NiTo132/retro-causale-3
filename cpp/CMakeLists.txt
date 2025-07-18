cmake_minimum_required(VERSION 3.12)
project(quantum_selector)

# Configuration C++
set(CMAKE_CXX_STANDARD 20)  # C++20 pour std::execution
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimisations de compilation
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Flags d'optimisation aggressive
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -mtune=native")

# Support SIMD et instructions modernes
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mfma -msse4.2")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -flto -ffast-math")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX2")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /GL /fp:fast")
endif()

# Recherche des dépendances
find_package(pybind11 REQUIRED)

# OpenMP pour parallélisation
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP trouvé: ${OpenMP_CXX_VERSION}")
else()
    message(FATAL_ERROR "OpenMP requis mais non trouvé")
endif()

# Configuration du module
pybind11_add_module(quantum_selector quantum_selector.cpp)

# Liaison avec OpenMP
target_link_libraries(quantum_selector PRIVATE OpenMP::OpenMP_CXX)

# Optimisations spécifiques
target_compile_features(quantum_selector PRIVATE cxx_std_20)

# Flags spécifiques par compilateur
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(quantum_selector PRIVATE 
        -fopenmp-simd 
        -fvectorize 
        -funroll-loops
        -fprefetch-loop-arrays
    )
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(quantum_selector PRIVATE 
        -fopenmp-simd 
        -vectorize 
        -unroll-loops
    )
endif()

# Définitions pour optimisations conditionnelles
target_compile_definitions(quantum_selector PRIVATE 
    PYBIND11_DETAILED_ERROR_MESSAGES
    QUANTUM_SELECTOR_OPTIMIZED=1
)

# Configuration debug
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(quantum_selector PRIVATE QUANTUM_DEBUG=1)
    target_compile_options(quantum_selector PRIVATE -g -O0)
endif()

# Tests de performance optionnels
option(BUILD_BENCHMARKS "Build performance benchmarks" OFF)
if(BUILD_BENCHMARKS)
    find_package(benchmark REQUIRED)
    add_executable(quantum_benchmark benchmark.cpp)
    target_link_libraries(quantum_benchmark benchmark::benchmark)
endif()

# Informations de build
message(STATUS "Configuration Quantum Selector:")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  OpenMP: ${OpenMP_CXX_VERSION}")
message(STATUS "  Flags: ${CMAKE_CXX_FLAGS_RELEASE}")

# Installation
install(TARGETS quantum_selector DESTINATION .)
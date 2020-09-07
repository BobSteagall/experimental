cmake_minimum_required(VERSION 3.15)
project(simd)

include(CTest)

##- Include and library paths for Intel compiler run-time library.
##
if(DEFINED $ENV{INTEL_RTL_INC_PATH})
    set(RTL_INC_PATH $ENV{INTEL_RTL_INC_PATH})
else()
    set(RTL_INC_PATH /usr/local/intel/mkl/include)
endif()

if(DEFINED $ENV{INTEL_RTL_LIB_PATH})
    set(RTL_LIB_PATH $ENV{INTEL_RTL_LIB_PATH})
else()
    set(RTL_LIB_PATH /usr/local/intel/lib/intel64)
endif()

##- Include and library paths for MKL
##
if(DEFINED $ENV{INTEL_MKL_INC_PATH})
    set(MKL_INC_PATH $ENV{INTEL_MKL_INC_PATH})
else()
    set(MKL_INC_PATH /usr/local/intel/mkl/include)
endif()

if(DEFINED $ENV{INTEL_MKL_LIB_PATH})
    set(MKL_LIB_PATH $ENV{INTEL_MKL_LIB_PATH})
else()
    set(MKL_LIB_PATH /usr/local/intel/mkl/lib/intel64)
endif()

##- Headers and source files
##
set(Includes
    include/simd.hpp
    include/utils.hpp
    include/test.hpp
)

set(Sources
    src/main.cpp
    src/simd.cpp
    src/utils.cpp
    src/test.cpp
)

##- Header and library search paths
##
set(IncludeDirs
    include
    src
    ${MKL_INC_PATH}
    ${RTL_INC_PATH}
)

set(LibraryDirs
    ${MKL_LIB_PATH}
    ${RTL_LIB_PATH}
)

##- Compilation flags and libs
##
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)

set(SIMD_CXX_FLAGS "-fPIC -mavx512f -march=skylake-avx512 -Wall -Wextra -pedantic -fmax-errors=10")

set(CMAKE_CXX_FLAGS_DEBUG          "-g ${SIMD_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-g -O2 ${SIMD_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE        "-O3 ${SIMD_CXX_FLAGS}")

find_library(MKL  mkl_rt PATHS ${LibraryDirs})
find_library(SVML svml   PATHS ${LibraryDirs})

##- Define the target
##
add_executable(simd_test ${Includes} ${Sources})
target_include_directories(simd_test PUBLIC ${IncludeDirs})
target_link_libraries(simd_test ${MKL} pthread)


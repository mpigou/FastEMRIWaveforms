# Define the compilation of the Cython part of this project

# ---- Handle LAPACK/LAPACKE access ----
if(FEW_LAPACK_FETCH STREQUAL "AUTO")
  # Try to find both LAPACK and LAPACKE
  find_package(lapack)
  if(lapack_FOUND)
    find_package(lapacke)
  endif()
  if(lapack_FOUND AND lapacke_FOUND)
    set(_FEW_ACTUALLY_FETCH_LAPACK OFF)
    message(
      STATUS "Found LAPACK ${lapack_VERSION} and LAPACKE ${lapacke_VERSION}.")
  else()
    set(_FEW_ACTUALLY_FETCH_LAPACK ON)
  endif()

elseif(FEW_LAPACK_FETCH STREQUAL "ON")
  set(_FEW_ACTUALLY_FETCH_LAPACK ON)

elseif(FEW_LAPACK_FETCH STREQUAL "OFF")
  find_package(lapack REQUIRED)
  find_package(lapacke REQUIRED)
  set(_FEW_ACTUALLY_FETCH_LAPACK OFF)

  message(
    STATUS "Found LAPACK ${lapack_VERSION} and LAPACKE ${lapacke_VERSION}.")

else()
  message(
    FATAL_ERROR
      "Unexpected value for FEW_FETCH_LAPACK (=${FEW_FETCH_LAPACK}, expected [ON|OFF|AUTO])."
  )

endif()

# Add message to detail how AUTO is handled
if(FEW_FETCH_LAPACK STREQUAL "AUTO")
  # Test whether LAPACK(E) is accessible in environment
  if(NOT _FEW_ACTUALLY_FETCH_LAPACK)
    message(
      STATUS
        "FEW_FETCH_LAPACK was set to AUTO and will be turned OFF as LAPACK(E) is already available."
    )
  else()
    message(
      STATUS
        "FEW_FETCH_LAPACK was set to AUTO and will be turned ON as LAPACK(E) is not available."
    )
  endif()
endif()

# ---- Install LAPACK using FetchContent ----
if(_FEW_ACTUALLY_FETCH_LAPACK)
  include(CPM)
  enable_language(Fortran)
  CPMAddPackage(
    NAME lapack
    GITHUB_REPOSITORY Reference-LAPACK/lapack
    GIT_TAG 04b044e020a3560ccfa9988c8a80a1fb7083fc2e # v3.12.0
    OPTIONS "LAPACKE"
            "ON"
            "CMAKE_POSITION_INDEPENDENT_CODE"
            "ON"
            "CMAKE_UNITY_BUILD"
            "ON"
            "CMAKE_UNITY_BUILD_BATCH_SIZE"
            64)
endif()

set(_COMMON_ARGS INCLUDE PRIVATE ${Python_NumPy_INCLUDE_DIR} WITH_CPU_VERSION)

few_add_cython_library(
  NAME few_cutils_pymatmul
  PYX_SOURCES pymatmul.pyx
  CU_SOURCES matmul.cu
  HEADERS cuda_complex.hpp global.h matmul.hh ${_COMMON_ARGS})

few_add_cython_library(
  NAME few_cutils_pyinterp
  PYX_SOURCES pyinterp.pyx
  CU_SOURCES interpolate.cu
  LINK PUBLIC ${FEW_LAPACK_LIBS}
  HEADERS cuda_complex.hpp global.h interpolate.hh ${_COMMON_ARGS})

few_add_cython_library(
  NAME few_cutils_pyAAK
  PYX_SOURCES gpuAAKWrap.pyx
  CU_SOURCES gpuAAK.cu
  HEADERS cuda_complex.hpp global.h gpuAAK.hh ${_COMMON_ARGS})

few_add_cython_library(
  NAME few_cutils_pyAmpInterp2D
  PYX_SOURCES pyampinterp2D.pyx
  CU_SOURCES AmpInterp2D.cu
  HEADERS cuda_complex.hpp global.h AmpInterp2D.hh ${_COMMON_ARGS})

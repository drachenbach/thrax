cmake_minimum_required(VERSION 3.0)
project(thrax)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fopenmp")

# add a NativeRelease build type
SET( CMAKE_CXX_FLAGS_NATIVERELEASE "-O3 -DNDEBUG -march=native -ftree-vectorize" CACHE STRING
        "Flags used by the C++ compiler during native builds."
        FORCE )
SET( CMAKE_C_FLAGS_NATIVERELEASE "-O3 -DNDEBUG -march=native -ftree-vectorize" CACHE STRING
        "Flags used by the C compiler during native builds."
        FORCE )
SET( CMAKE_EXE_LINKER_FLAGS_NATIVERELEASE "" CACHE STRING
        "Flags used for linking binaries during native builds."
        FORCE )
SET( CMAKE_SHARED_LINKER_FLAGS_NATIVERELEASE "" CACHE STRING
        "Flags used by the shared libraries linker during native builds."
        FORCE )
MARK_AS_ADVANCED(
        CMAKE_CXX_FLAGS_NATIVERELEASE
        CMAKE_C_FLAGS_NATIVERELEASE
        CMAKE_EXE_LINKER_FLAGS_NATIVERELEASE
        CMAKE_SHARED_LINKER_FLAGS_NATIVERELEASE )

if(CMAKE_CONFIGURATION_TYPES)
    list(APPEND CMAKE_CONFIGURATION_TYPES NativeRelease)
    list(REMOVE_DUPLICATES CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_CONFIGURATION_TYPES "${CMAKE_CONFIGURATION_TYPES}" CACHE STRING
            "Add the native configuration"
            FORCE)
endif()

## default build type is native
IF(NOT CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE NativeRelease CACHE STRING
            "Choose the type of build. Options are: None Debug NativeRelease Release RelWithDebInfo MinSizeRel."
            FORCE)
ENDIF(NOT CMAKE_BUILD_TYPE)
MESSAGE(STATUS "Build type: " ${CMAKE_BUILD_TYPE})

## look for MKL
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/extern/cmake/")
find_package(MKL REQUIRED)
if(MKL_FOUND)
    message(STATUS "MKL found " ${MKL_INCLUDE_DIRS})
    include_directories(${MKL_INCLUDE_DIRS})
    # link_directories(${MKL_LIBRARIES})
    set(EXTRA_LIBS ${EXTRA_LIBS} ${MKL_LIBRARIES})
endif()

## look for TBB
#find_package(TBB REQUIRED)
#if(TBB_FOUND)
#    message(STATUS "TBB found: " ${TBB_INCLUDE_DIRS})
#    include_directories(${TBB_INCLUDE_DIRS})
#    # link_libraries(${TBB_LIBRARIES})
#    set(EXTRA_LIBS ${EXTRA_LIBS} ${TBB_LIBRARIES})
#endif(TBB_FOUND)

## look for boost
find_package(Boost COMPONENTS program_options system filesystem log timer)
if(NOT ${Boost_FOUND})
    SET(BOOST_ROOT ~/local) # default
    SET(Boost_NO_SYSTEM_PATHS ON) # force to use own build
    find_package(Boost COMPONENTS program_options system filesystem log timer)
endif(NOT ${Boost_FOUND})
if(Boost_FOUND)
    message(STATUS "Boost found")
    include_directories(${Boost_INCLUDE_DIR})
    # link_libraries(${Boost_LIBRARIES})
    set(EXTRA_LIBS ${EXTRA_LIBS} ${Boost_LIBRARIES})
endif(Boost_FOUND)
## compile and link flags for boost log
set(BOOST_LOG_COMPILE_FLAGS "-DBOOST_LOG_DYN_LINK")
set(BOOST_LOG_LINK_FLAGS "-DBOOST_LOG_DYN_LINK")

## add compile flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${BOOST_LOG_COMPILE_FLAGS}")
## add link flags
SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${BOOST_LOG_LINK_FLAGS}")

# look for blas
find_package(BLAS REQUIRED)
if(BLAS_FOUND)
    message(STATUS "BLAS found")
    include_directories(${BLAS_INCLUDE_DIR})
    # link_libraries(${BLAS_LIBRARIES})
    set(EXTRA_LIBS ${EXTRA_LIBS} ${BLAS_LIBRARIES})
endif(BLAS_FOUND)


## set include directory
include_directories(include)
include_directories(/usr/local/lib/eigen3)

## define thrax library
set(LIB_HEADERS
        include/thrax/struct/Data.h
        include/thrax/util/FileUtil.h
        include/thrax/struct/Triple.h
        include/thrax/model/BaseModel.h
        include/thrax/model/EmbeddingParameterSet.h
        include/thrax/model/TransE.h
        include/thrax/sampler/Sampler.h
        include/thrax/initializer/Initializer.h
        include/thrax/util/RandomUtil.h
        include/thrax/parameterUpdater/ParameterUpdater.h
        include/thrax/struct/DataContainer.h
        include/thrax/initializer/XavierInitializer.h
        include/thrax/initializer/ScalarInitializer.h
        include/thrax/struct/Gradient.h
        include/thrax/evaluation/Evaluation.h
        include/thrax/parameterUpdater/SGDParameterUpdater.h
        include/thrax/sampler/LCWASampler.h
        include/thrax/parameterUpdater/AdaGradParameterUpdater.h
        include/thrax/model/DISTMULT.h
        include/thrax/sampler/CorruptionSampler.h
        include/thrax/util/Typedefs.h
        include/thrax/model/ModelFactory.h
        include/thrax/util/GradientChecker.h
        include/thrax/optimizer/Optimizer.h
        include/thrax/util/MathUtil.h
        include/thrax/model/RESCAL.h
        include/thrax/parameterUpdater/AdaDeltaParameterUpdater.h
        include/thrax/parameterUpdater/RMSPropParameterUpdater.h
        include/thrax/parameterUpdater/ParameterUpdaterFactory.h
        include/thrax/sampler/SamplerFactory.h
        include/thrax/model/ComplEx.h
        include/thrax/lossFunction/LossFunction.h
        include/thrax/lossFunction/PairwiseLossFunction.h
        include/thrax/lossFunction/LossFunctionFactory.h
        include/thrax/lossFunction/LogisticLossFunction.h
        include/thrax/lossFunction/SoftmaxLossFunction.h
        include/thrax/initializer/InitializerFactory.h
        include/thrax/initializer/PreTrainedInitializer.h
        include/thrax/model/TrivialEnsemble.h
        include/thrax/model/AbstractModel.h
        include/thrax/model/BaseEnsembleModel.h)

## add library to project
add_library(thrax SHARED ${LIB_HEADERS})
set_target_properties(thrax PROPERTIES LINKER_LANGUAGE CXX)

## link libraries
link_libraries(thrax ${EXTRA_LIBS})

set(SOURCE_FILES src/main.cpp)
add_executable(thrax.bin ${SOURCE_FILES})

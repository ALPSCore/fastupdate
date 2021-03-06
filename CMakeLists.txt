### fastupdate cmake configuration

cmake_minimum_required (VERSION 2.8.12)

# use only CXX compiler
project (FastUpdate CXX)

# find additional cmake module in cmake dir
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

# Print build type
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# Where to install?
message(STATUS "Will be installed in ${CMAKE_INSTALL_PREFIX}")
message(STATUS "Redifine CMAKE_INSTALL_PREFIX if you wish to change install location")

### Dependencies
find_package(Boost 1.55.0 REQUIRED)
find_package(Eigen3 REQUIRED)

#include directories
set(FASTUPDATE_LIBRARY_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIR} ${Boost_INCLUDE_DIRS} "fastupdate/include" "test") #rest taken care of by libraries dependencies

#unset(LINK_ALL)
#list(APPEND LINK_ALL ${ALPSCore_LIBRARIES})
    
#add_library(fastupdate fastupdate/src/fastupdate.cpp)
#target_link_libraries(gf ${ALPSCore_LIBRARIES} ${MPI_CXX_LIBRARIES})
#list(APPEND LINK_ALL gf)

# Build and enable tests
#testing setup
option(Testing "Enable testing" ON)
include(EnableGtests) #defined in ./cmake
#include_directories("test")
message(${FASTUPDATE_LIBRARY_INCLUDE_DIRS})
include_directories(${FASTUPDATE_LIBRARY_INCLUDE_DIRS})
#list(APPEND LINK_ALL hyb)

#testing source files
set(unittest_src unittest)

enable_testing(test)

foreach(test ${unittest_src})
    add_gtest(${test} test)
endforeach(test)

install(DIRECTORY "fastupdate/include" DESTINATION "." FILES_MATCHING PATTERN "*.hpp")

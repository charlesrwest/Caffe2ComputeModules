cmake_minimum_required (VERSION 2.8.3)
PROJECT(sineWave)

#Add git submodule
#git submodule add https://github.com/philsquared/Catch.git

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/modules/")

#Get c++11
ADD_DEFINITIONS(-std=c++11)

find_package(Eigen3 REQUIRED)
find_package(Caffe2 REQUIRED)
find_package(Protobuf REQUIRED)


#Tell compiler where to find required libraries
link_directories(/usr/lib/x86_64-linux-gnu/)

include_directories(./src/library/ ./src/executables/unitTests ./src/executables/trainMNISTExample ./Catch/single_include ${Caffe2_INCLUDE_DIR} ${EIGEN3_INCLUDE_DIR})

find_package(Threads)

#Get lists of the source files
FILE(GLOB SOURCEFILES ./src/library/*.cpp ./src/library/*.c)

FILE(GLOB UNIT_TESTS_EXECUTABLE_SOURCE ./src/executables/unitTests/*.cpp ./src/executables/unitTests/*.c)

FILE(GLOB TRAIN_SINE_WAVE_EXAMPLE ./src/executables/trainSineWave/*.cpp ./src/executables/trainSineWave/*.c)

#Set the binaries to be placed in the ./bin/ directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "bin/")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "lib/")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "lib/")


#Create a libraries
add_library(sineWaveLibrary SHARED  ${SOURCEFILES})


#Tell CMake what executables to make
ADD_EXECUTABLE(unitTests ${UNIT_TESTS_EXECUTABLE_SOURCE}) 

ADD_EXECUTABLE(trainSineWave ${TRAIN_SINE_WAVE_EXAMPLE}) 

target_link_libraries(sineWaveLibrary dl ${CMAKE_THREAD_LIBS_INIT} ${Caffe2_LIBS} ${PROTOBUF_LIBRARY} glog)

#link libraries to executable
target_link_libraries(unitTests sineWaveLibrary)

target_link_libraries(trainSineWave sineWaveLibrary)

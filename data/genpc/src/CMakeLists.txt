cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(genpc)

find_package(PCL 1.2 REQUIRED)
find_package(OpenCV 3.2 REQUIRED )
find_package(Boost REQUIRED)
find_package(VTK REQUIRED)

if(NOT VTK_FOUND)
    message("VTK library Not found")
endif()

include_directories(${PCL_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
include(${VTK_USE_FILE})

add_library(obj_scan SHARED obj_scan.cpp)
add_library(surface_construct SHARED surface_construct.cpp)

add_executable (genpc main.cpp)
target_link_libraries (genpc obj_scan surface_construct ${VTK_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBS})

install(TARGETS genpc RUNTIME DESTINATION bin)

cmake_minimum_required(VERSION 3.10)
project(Test1)

find_package(OpenCV REQUIRED)

set(CMAKE_CXX_STANDARD 17)

include_directories("C:\\Program Files (x86)\\Eigen3\\include")

add_executable(test1.cpp)
target_link_libraries(Test1 ${OpenCV_LIBRARIES})

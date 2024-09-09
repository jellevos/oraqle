cmake_minimum_required(VERSION 3.10.2 FATAL_ERROR)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(helib)
add_executable(main main.cpp)
target_link_libraries(main helib)

cmake_minimum_required(VERSION 3.14)

# Project name and version
project(OpenFHEProject VERSION 1.0 LANGUAGES CXX)

# Use C++17 or newer
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find OpenFHE
find_package(OpenFHE REQUIRED)

# Add your executable
add_executable(main main.cpp)

# Link OpenFHE to your executable
target_link_libraries(main PRIVATE OpenFHE::pke OpenFHE::binfhe OpenFHE::core)

# Include OpenFHE headers
target_include_directories(main PRIVATE ${OpenFHE_INCLUDE_DIRS})

cmake_minimum_required(VERSION 2.8)
set(CMAKE_CXX_STANDARD 11)
#set(CMAKE_BUILD_TYPE RelWithDebInfo)
set(CMAKE_BUILD_TYPE Debug)

project(boruvkasupix)
find_package(OpenCV 3)
add_executable(${PROJECT_NAME} "src/main.cpp" "src/boruvka_superpixel.cpp")
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})


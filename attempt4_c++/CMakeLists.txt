cmake_minimum_required(VERSION 2.8)
project(ECE_417_598_Project)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(DisplayImage DisplayImage.cpp)
add_executable(CornerDetection CornerDetection.cpp)
add_executable(ProjectMain ProjectMain.cpp)

target_link_libraries(DisplayImage ${OpenCV_LIBS})
target_link_libraries(CornerDetection ${OpenCV_LIBS})
target_link_libraries(ProjectMain Eigen3::Eigen ${OpenCV_LIBS})


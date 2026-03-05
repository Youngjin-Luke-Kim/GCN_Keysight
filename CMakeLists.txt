cmake_minimum_required(VERSION 3.18)
project(gcn_homework)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Path to LibTorch
# /mnt/d/sccad/2026_spring/01_publications/iccad-26/openeye-1.0/.venv/lib/python3.12/site-packages/torch/share/cmake
find_package(Torch REQUIRED)
add_executable(gcn main.cpp)
target_link_libraries(gcn "${TORCH_LIBRARIES}")

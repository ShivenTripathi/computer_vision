cmake_minimum_required(VERSION 3.10)
project(Computer_Vision)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(d32 d32.cpp)
target_link_libraries(d32 ${OpenCV_LIBS})

add_executable(drawing drawing.cpp)
target_link_libraries(drawing ${OpenCV_LIBS})

add_executable(morph morph.cpp)
target_link_libraries(morph ${OpenCV_LIBS})

add_executable(shape shape_detection.cpp)
target_link_libraries(shape ${OpenCV_LIBS})

add_executable(gate_detection gate_detection.cpp)
target_link_libraries(gate_detection ${OpenCV_LIBS})

add_executable(file_sys file_sys.cpp)
target_link_libraries(file_sys ${OpenCV_LIBS})
target_link_libraries(file_sys stdc++fs)


add_executable(petcat_face_rec petcat_face_rec.cpp)
target_link_libraries(petcat_face_rec ${OpenCV_LIBS})
target_link_libraries(petcat_face_rec stdc++fs)


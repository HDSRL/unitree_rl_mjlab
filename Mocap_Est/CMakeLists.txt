cmake_minimum_required(VERSION 3.10)

project(MocapEstimationBridge LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Threads REQUIRED)

set(VRPN_ROOT "" CACHE PATH "Root directory of a local VRPN install")
if(NOT VRPN_ROOT AND DEFINED ENV{VRPN_ROOT})
    set(VRPN_ROOT "$ENV{VRPN_ROOT}")
endif()

set(VRPN_HINTS
    ${VRPN_ROOT}
    ${CMAKE_CURRENT_LIST_DIR}/../../Libraries/VRPN_workspace/install
    ${CMAKE_CURRENT_LIST_DIR}/../../Libraries/VRPN_ws/install
    ${CMAKE_CURRENT_LIST_DIR}/../../../raisim_legged/Libraries/VRPN_ws/install
)

find_path(VRPN_INCLUDE_DIR
    NAMES vrpn_Tracker.h
    HINTS ${VRPN_HINTS}
    PATHS /usr/local/include /usr/include
    PATH_SUFFIXES include
)

find_library(VRPN_LIBRARY
    NAMES vrpn
    HINTS ${VRPN_HINTS}
    PATHS /usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu
    PATH_SUFFIXES lib lib64
)

find_library(QUAT_LIBRARY
    NAMES quat
    HINTS ${VRPN_HINTS}
    PATHS /usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu
    PATH_SUFFIXES lib lib64
)

set(MISSING_DEPENDENCIES)

if(NOT Threads_FOUND)
    list(APPEND MISSING_DEPENDENCIES "Threads")
endif()

if(NOT VRPN_LIBRARY)
    list(APPEND MISSING_DEPENDENCIES "libvrpn (set VRPN_LIBRARY or VRPN_ROOT)")
endif()

if(NOT VRPN_INCLUDE_DIR)
    list(APPEND MISSING_DEPENDENCIES "vrpn_Tracker.h (set VRPN_INCLUDE_DIR or VRPN_ROOT)")
endif()

if(NOT QUAT_LIBRARY)
    list(APPEND MISSING_DEPENDENCIES "libquat (set QUAT_LIBRARY or VRPN_ROOT)")
endif()

if(MISSING_DEPENDENCIES)
    string(REPLACE ";" "\n  - " MISSING_DEPENDENCIES_LIST "${MISSING_DEPENDENCIES}")
    message(FATAL_ERROR "Missing required dependencies:\n  - ${MISSING_DEPENDENCIES_LIST}")
endif()

message(STATUS "Dependency check passed")
message(STATUS "  Threads: found")
message(STATUS "  VRPN include: ${VRPN_INCLUDE_DIR}")
message(STATUS "  VRPN library: ${VRPN_LIBRARY}")
message(STATUS "  QUAT library: ${QUAT_LIBRARY}")

add_executable(broadcast_node
    Broadcast_node.cpp
)

add_executable(multi_mocap_bridge
    MultiBroadcast_node.cpp
)

add_executable(monitor_node
    multi_mocap_monitor.cpp
)

target_include_directories(broadcast_node PRIVATE
    ${VRPN_INCLUDE_DIR}
)

target_include_directories(multi_mocap_bridge PRIVATE
    ${VRPN_INCLUDE_DIR}
)

target_link_libraries(broadcast_node PRIVATE
    ${VRPN_LIBRARY}
    ${QUAT_LIBRARY}
    Threads::Threads
)

target_link_libraries(multi_mocap_bridge PRIVATE
    ${VRPN_LIBRARY}
    ${QUAT_LIBRARY}
    Threads::Threads
)

target_link_libraries(monitor_node PRIVATE
    Threads::Threads
)

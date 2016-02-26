# Copyright (c) 2016, Intel Corporation
# NeuralIA

set (TARGET_NAME "examples")
# Sources for test files

file (GLOB API_HEADERS
       ${CMAKE_SOURCE_DIR}/api/*.h
      )
file (GLOB API_SRC
       ${CMAKE_SOURCE_DIR}/src/*.h
       ${CMAKE_SOURCE_DIR}/src/*.cpp
      )
file  (GLOB EXAMPLES_SRC
       ${CMAKE_SOURCE_DIR}/examples/*.h
       ${CMAKE_SOURCE_DIR}/examples/*.cpp
      )
# Create named folders for the sources within the .vcproj
# Empty name lists them directly under the .vcproj
source_group("api\\headers" FILES ${API_HEADERS})
source_group("api\\src" FILES ${API_SRC})
source_group("src" FILES ${EXAMPLES_SRC})

# Properties->C/C++->General->Additional Include Directories
include_directories (${CMAKE_SOURCE_DIR})

# Create library file from sources.
add_executable(${TARGET_NAME}
               ${API_SRC}
               ${API_HEADERS}
               ${EXAMPLES_SRC})

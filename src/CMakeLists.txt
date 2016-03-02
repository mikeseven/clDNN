# Copyright (c) 2016, Intel Corporation
# NeuralIA

set (TARGET_NAME "neuralia")
# Sources for test files

file (GLOB API_HEADERS
       ${CMAKE_SOURCE_DIR}/api/*.h
      )
file (GLOB LIBRARY_SRC
       ${CMAKE_SOURCE_DIR}/src/*.h
       ${CMAKE_SOURCE_DIR}/src/*.cpp
      )

# Create named folders for the sources within the .vcproj
# Empty name lists them directly under the .vcproj

source_group("api" FILES ${API_HEADERS})
source_group("src" FILES ${LIBRARY_SRC})

# Properties->C/C++->General->Additional Include Directories
include_directories (${CMAKE_SOURCE_DIR}
                     ${CMAKE_SOURCE_DIR}/src
                    )

# Create library file from sources.
add_library(${TARGET_NAME} STATIC
            ${LIBRARY_SRC}
            ${API_HEADERS})

#cmake ver req
cmake_minimum_required(VERSION 2.8)

if(UNIX)
	add_subdirectory(precompile)
endif()
	
set(project_name SSSS)

#project name
project(${project_name})

set(CMAKE_BUILD_TYPE Debug)

#so that it talks to us
set(CMAKE_VERBOSE_MAKEFILE ON)

if(UNIX)
	add_definitions( "-Wall -Wno-long-long -ansi -pedantic -std=c++0x" )
endif()

if(${CMAKE_BUILD_TYPE} STREQUAL Debug)
  add_definitions("-DDebug")
  message("Debug mode")
endif()

if(${CMAKE_BUILD_TYPE} STREQUAL Release)
  add_definitions("-DRelease")
  message("Release mode")
endif()

#header files source
include_directories(${CMAKE_SOURCE_DIR}/src/)
link_directories(${CMAKE_SOURCE_DIR}/src/)

include_directories(${CMAKE_SOURCE_DIR}/external/include/)
link_directories(${CMAKE_SOURCE_DIR}/external/lib/)

include_directories(${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

if(WIN32)
	set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../)
	set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../)
	set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../)
endif()

if(UNIX)
	include_directories(
	/usr/include/gtk-2.0
	/usr/lib/x86_64-linux-gnu/gtk-2.0/include
	/usr/include/atk-1.0
	/usr/include/cairo
	/usr/include/gdk-pixbuf-2.0
	/usr/include/pango-1.0
	/usr/include/gio-unix-2.0/
	/usr/include/glib-2.0
	/usr/lib/x86_64-linux-gnu/glib-2.0/include
	/usr/include/pixman-1
	/usr/include/freetype2
	/usr/include/libpng12
	)
endif()

if(UNIX)
	set(${project_name}_external_libs sfml-window sfml-system sfml-audio sfml-graphics GL GLEW freetype assimp)
endif()

if(WIN32)
	set(${project_name}_external_libs optimized sfml-window debug sfml-window-d optimized sfml-system debug sfml-system-d 
	optimized sfml-audio debug sfml-audio-d optimized sfml-graphics debug sfml-graphics-d OpenGL32 GLEW32 
  #debug freetype2410_D optimized freetype2410 
  debug assimpd optimized assimp
  berkelium)
endif()
	
#adding the project's exe
add_executable(${project_name} ssss browser)

if(UNIX)
	add_precompiled_header(${project_name} #add precompiled headers
	${CMAKE_SOURCE_DIR}/external/include/mymath/mm_common.h
	${CMAKE_SOURCE_DIR}/external/include/mymath/mm_vec2_impl.h
	${CMAKE_SOURCE_DIR}/external/include/mymath/mm_vec3_impl.h
	${CMAKE_SOURCE_DIR}/external/include/mymath/mm_vec4_impl.h
	${CMAKE_SOURCE_DIR}/external/include/mymath/includes/vec2_swizzle_declarations.h
	${CMAKE_SOURCE_DIR}/external/include/mymath/includes/vec3_swizzle_declarations.h
	${CMAKE_SOURCE_DIR}/external/include/mymath/includes/vec4_swizzle_declarations.h
	${CMAKE_SOURCE_DIR}/external/include/mymath/mm_vec_swizzle_out.h
	${CMAKE_SOURCE_DIR}/external/include/mymath/mm_vec_func.h
	${CMAKE_SOURCE_DIR}/external/include/mymath/mm_mat2_impl.h
	${CMAKE_SOURCE_DIR}/external/include/mymath/mm_mat3_impl.h
	${CMAKE_SOURCE_DIR}/external/include/mymath/mm_mat4_impl.h
	${CMAKE_SOURCE_DIR}/external/include/mymath/mm_mat_func.h
	${CMAKE_SOURCE_DIR}/external/include/mymath/mm_util.h
	${CMAKE_SOURCE_DIR}/external/include/mymath/mm_frame.h
	${CMAKE_SOURCE_DIR}/external/include/mymath/mm_camera.h
	${CMAKE_SOURCE_DIR}/external/include/mymath/mm_matrix_stack.h
	${CMAKE_SOURCE_DIR}/external/include/mymath/mm_pipeline.h
	${CMAKE_SOURCE_DIR}/external/include/mymath/mymat.h)
endif()

target_link_libraries(${project_name} ${${project_name}_external_libs})
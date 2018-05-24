#
#	GLSL boilerplate CMake file
#
#	Designed for VS2013 and other generic
#
#	maintained by mwkm
#

cmake_minimum_required( VERSION 2.8 )
cmake_policy( SET CMP0015 NEW )	# Relative path to SOURCE_DIR

#
#		---- Change Project Name here ----
#
project( idm CXX C )

#	CMake configuration
set( CMAKE_USE_RELATIVE_PATHS ON )
set( CMAKE_SUPPRESS_REGENERATION 1 )
set( CMAKE_BUILD_TYPE Release )


set(CMAKE_BINARY_DIR ".")
#set_target_properties( libpbrt PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "lib/")

#	include directories
#
include_directories (
	"."
	"./ext/glad/include"
	"./ext/eigen"
	"./ext/glfw/include"
	"./ext/nanovg"
	"./ext/nanogui/include"
)

link_directories (
	"./ext/nanogui/lib"
)

add_definitions ("-DNANOGUI_SHARED" "-DGLAD_GLAPI_EXPORT" "-D_CRT_SECURE_NO_WARNINGS")

#	pbrt executable
add_executable( idm
	idm.cpp
	bmp.cpp
	bmp.h
)

target_link_libraries( idm
	nanogui.lib
	opengl32.lib
)

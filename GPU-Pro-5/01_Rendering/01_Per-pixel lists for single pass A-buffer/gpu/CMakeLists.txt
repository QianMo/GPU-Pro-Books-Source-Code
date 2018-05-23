CMAKE_MINIMUM_REQUIRED(VERSION 2.6)
PROJECT(abuffer)

set(LibSL_DIR $ENV{LIBSL_DIR} )
find_package(LibSL)

INCLUDE_DIRECTORIES( 
	${LIBSL_INCLUDE_DIR}
)

LINK_DIRECTORIES(
	$ENV{LIBSL_DIR}/lib
)

SET(ASPPLL_SOURCES
	asppll/dll.cpp
	asppll/asppll_build.vp
	asppll/asppll_build.fp
	asppll/asppll_build.h
	asppll/asppll_render.vp
	asppll/asppll_render.fp
	asppll/asppll_render.h
	asppll/asppll_clear.vp
	asppll/asppll_clear.fp
	asppll/asppll_clear.h
	asppll/implementations.fp
)

# Always sorted per-pixel linked-lists, paged alloc, max64

ADD_LIBRARY(prelin-paged SHARED ${ASPPLL_SOURCES})
target_link_libraries(prelin-paged
	${LIBSL_LIBRARIES} 
	${LIBSL_GL4_LIBRARIES}
)
INSTALL(TARGETS prelin-paged RUNTIME DESTINATION ${CMAKE_SOURCE_DIR}/bin)

# Always sorted per-pixel linked-lists, naive alloc, max64

ADD_LIBRARY(prelin-naive SHARED ${ASPPLL_SOURCES})
target_link_libraries(prelin-naive
	${LIBSL_LIBRARIES} 
	${LIBSL_GL4_LIBRARIES}
)
INSTALL(TARGETS prelin-naive RUNTIME DESTINATION ${CMAKE_SOURCE_DIR}/bin)
set_target_properties(prelin-naive PROPERTIES COMPILE_DEFINITIONS "ALLOC_NAIVE=1")

# Always sorted per-pixel linked-lists, paged alloc, cas32

ADD_LIBRARY(prelin-cas32-paged SHARED ${ASPPLL_SOURCES})
target_link_libraries(prelin-cas32-paged
	${LIBSL_LIBRARIES} 
	${LIBSL_GL4_LIBRARIES}
)
INSTALL(TARGETS prelin-cas32-paged RUNTIME DESTINATION ${CMAKE_SOURCE_DIR}/bin)
set_target_properties(prelin-cas32-paged PROPERTIES COMPILE_DEFINITIONS "ASPPLL_CAS32=1")

# Always sorted per-pixel linked-lists, naive alloc, cas32

ADD_LIBRARY(prelin-cas32-naive SHARED ${ASPPLL_SOURCES})
target_link_libraries(prelin-cas32-naive
	${LIBSL_LIBRARIES} 
	${LIBSL_GL4_LIBRARIES}
)
INSTALL(TARGETS prelin-cas32-naive RUNTIME DESTINATION ${CMAKE_SOURCE_DIR}/bin)
set_target_properties(prelin-cas32-naive PROPERTIES COMPILE_DEFINITIONS "ALLOC_NAIVE=1;ASPPLL_CAS32=1")

# Linked-list, bubble sort, paged alloc

ADD_LIBRARY(postlin-paged SHARED ${ASPPLL_SOURCES})
target_link_libraries(postlin-paged
	${LIBSL_LIBRARIES} 
	${LIBSL_GL4_LIBRARIES}
)
INSTALL(TARGETS postlin-paged RUNTIME DESTINATION ${CMAKE_SOURCE_DIR}/bin)
set_target_properties(postlin-paged PROPERTIES COMPILE_DEFINITIONS "BUBBLE_SORT=1")

# Linked-list, bubble sort, naive alloc

ADD_LIBRARY(postlin-naive SHARED ${ASPPLL_SOURCES})
target_link_libraries(postlin-naive
	${LIBSL_LIBRARIES} 
	${LIBSL_GL4_LIBRARIES}
)
INSTALL(TARGETS postlin-naive RUNTIME DESTINATION ${CMAKE_SOURCE_DIR}/bin)
set_target_properties(postlin-naive PROPERTIES COMPILE_DEFINITIONS "BUBBLE_SORT=1;ALLOC_NAIVE=1")

# Open addressing lists, bubble sort

ADD_LIBRARY(postopen SHARED ${ASPPLL_SOURCES})
target_link_libraries(postopen
	${LIBSL_LIBRARIES} 
	${LIBSL_GL4_LIBRARIES}
)
INSTALL(TARGETS postopen RUNTIME DESTINATION ${CMAKE_SOURCE_DIR}/bin)
set_target_properties(postopen PROPERTIES COMPILE_DEFINITIONS "BUBBLE_SORT=1;HABUFFER=1")

# Open addressing lists, always sorted (HA-buffer)

ADD_LIBRARY(preopen SHARED ${ASPPLL_SOURCES})
target_link_libraries(preopen
	${LIBSL_LIBRARIES} 
	${LIBSL_GL4_LIBRARIES}
)
INSTALL(TARGETS preopen RUNTIME DESTINATION ${CMAKE_SOURCE_DIR}/bin)
set_target_properties(preopen PROPERTIES COMPILE_DEFINITIONS "HABUFFER=1")

# Open addressing lists, always sorted (HA-buffer), with early-culling

ADD_LIBRARY(preopen-ec SHARED ${ASPPLL_SOURCES})
target_link_libraries(preopen-ec
	${LIBSL_LIBRARIES} 
	${LIBSL_GL4_LIBRARIES}
)
INSTALL(TARGETS preopen-ec RUNTIME DESTINATION ${CMAKE_SOURCE_DIR}/bin)
set_target_properties(preopen-ec PROPERTIES COMPILE_DEFINITIONS "HABUFFER=1;EARLYCULLING=1")

##################

ADD_EXECUTABLE(benchmark benchmark.cpp abuffer.cpp sh_count.h sh_count.vp sh_count.fp)
TARGET_LINK_LIBRARIES(benchmark
	${LIBSL_LIBRARIES} 
	${LIBSL_GL4_LIBRARIES}
)
INSTALL(TARGETS benchmark RUNTIME DESTINATION ${CMAKE_SOURCE_DIR}/bin)

ADD_EXECUTABLE(seethrough seethrough.cpp abuffer.cpp sh_count.h sh_count.vp sh_count.fp)
TARGET_LINK_LIBRARIES(seethrough
	${LIBSL_LIBRARIES} 
	${LIBSL_GL4_LIBRARIES}
)
INSTALL(TARGETS seethrough RUNTIME DESTINATION ${CMAKE_SOURCE_DIR}/bin)

AUTO_BIND_SHADERS( 
	asppll/asppll_build
	asppll/asppll_render
	asppll/asppll_clear
	sh_count
)

##################

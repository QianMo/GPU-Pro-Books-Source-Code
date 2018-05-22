# Makefile for Win32

!include <win32.mak>

TOP  = ../../..
SRCS = smooth.c

!include "$(TOP)/glutwin32.mak"

trackball.c: ..\..\examples\trackball.c
	-del /f trackball.c
	copy ..\..\examples\trackball.c trackball.c

trackball.h: ..\..\examples\trackball.h
	-del /f trackball.h
	copy ..\..\examples\trackball.h trackball.h

# dependencies
smooth.exe	: tb.obj trackball.obj glm.obj gltx.obj
smooth.obj	: glm.h smooth.c tb.h
glm.obj         : glm.c glm.h
tb.obj		: tb.c tb.h trackball.h
trackball.obj	: trackball.c trackball.h

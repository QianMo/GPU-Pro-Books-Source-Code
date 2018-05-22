# Makefile for Win32

!include <win32.mak>

TOP  = ../../..
SRCS = bounce.c

!include "$(TOP)/glutwin32.mak"

trackball.c:
	@del trackball.c
	copy ..\..\examples\trackball.c trackball.c

trackball.h:
	@del trackball.h
	copy ..\..\examples\trackball.h trackball.h

tb.c:
	@del tb.c
	copy ..\smooth\tb.c tb.c

tb.h:
	@del tb.h
	copy ..\smooth\tb.h tb.h

clobber::
	@del tb.c tb.h trackball.c trackball.h

# dependencies
bounce.exe	: glui.obj tb.obj trackball.obj
bounce.obj	: bounce.c tb.h glui.h
glui.obj	: glui.c glui.h
tb.obj		: tb.c tb.h trackball.h

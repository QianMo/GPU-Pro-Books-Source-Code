# Makefile for Win32

!include <win32.mak>

TOP  = ../../..
SRCS = gliq.c

!include "$(TOP)/glutwin32.mak"

trackball.c:
	@del trackball.c
	copy ..\..\examples\trackball.c trackball.c

trackball.h:
	@del trackball.h
	copy ..\..\examples\trackball.h trackball.h

clobber::
	@del trackball.c
	@del trackball.h

# dependencies
gliq.exe	: tb.obj trackball.obj game.obj score.obj pick.obj board.obj
gliq.obj	: gliq.c tb.h gliq.h
tb.obj		: tb.c tb.h trackball.h
trackball.obj	: trackball.c trackball.h
game.obj	: game.c gliq.h
score.obj	: score.c gliq.h
pick.obj	: pick.c gliq.h
board.obj	: board.c gliq.h
gliq.h		: trackball.h

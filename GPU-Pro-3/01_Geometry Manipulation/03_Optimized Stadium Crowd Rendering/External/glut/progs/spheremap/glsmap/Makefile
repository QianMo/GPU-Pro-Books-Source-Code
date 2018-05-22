# Makefile for Win32

!include <win32.mak>

TOP  = ../../..
SRCS = rtsmap.c fakeraytrace.c

EXTRALIBS = $(GLSMAP)
DEPLIBS = $(GLSMAP)

!include "$(TOP)/glutwin32.mak"

# dependencies
rtsmap.obj: $(TOP)/include/GL/glsmap.h
fakeraytrace.obj: $(TOP)/include/GL/glsmap.h

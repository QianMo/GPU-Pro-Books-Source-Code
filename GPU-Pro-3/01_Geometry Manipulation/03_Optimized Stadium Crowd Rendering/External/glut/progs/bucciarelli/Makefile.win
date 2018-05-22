# Makefile for Win32
# Copyright (c) Mark J. Kilgard, 1997.

!include <win32.mak>

TOP  = ../..
SRCS = tunnel.c teapot.c fire.c terrain.c paltex.c gltest.c ray.c glbpaltx.c
EXTRALIBS = winmm.lib

!include "$(TOP)/glutwin32.mak"

# dependencies
tunnel.exe : image.obj sources.obj
teapot.exe : image.obj dteapot.obj shadow.obj
fire.exe : image.obj

# Makefile for Win32

!include <win32.mak>

TOP  = ../../..
SRCS = glflare.c

!include "$(TOP)/glutwin32.mak"

# dependencies
glflare.exe	: loadlum.obj
glflare.obj	: glflare.c vec3d.c loadlum.h

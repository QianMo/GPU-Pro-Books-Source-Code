# Makefile for Win32

!include <win32.mak>

TOP  = ../../..
SRCS = bluepony.c

!include "$(TOP)/glutwin32.mak"

# dependencies
bluepony.exe	: readtex.obj
bluepony.obj	: bluepony.c readtex.h
readtex.obj	: readtex.c readtex.h

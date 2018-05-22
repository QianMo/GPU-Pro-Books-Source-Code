# Makefile for Win32

!include <win32.mak>

TOP  = ../../..
SRCS = walker.c

!include "$(TOP)/glutwin32.mak"

# dependencies
walker.exe	: walkviewer.obj models.obj
walker.obj	: walker.h win32_dirent.h walker.c
walkviewer.obj	: walkviewer.h walkviewer.c

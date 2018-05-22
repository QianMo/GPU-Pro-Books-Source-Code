# Makefile for Win32

!include <win32.mak>

TOP  = ../../..
SRCS = Editor.c

!include "$(TOP)/glutwin32.mak"
CFLAGS = $(CFLAGS) -DYACME_DBG

# dependencies
Editor.exe	: invertmat.obj
Editor.c	: mallocbis.h RGBA.h

# Makefile for Win32

!include <win32.mak>

TOP  = ../../..
SRCS = make_face.c

!include "$(TOP)/glutwin32.mak"

# dependencies
make_face.exe	: display.obj fileio.obj main.obj muscle.obj 
make_face.obj	: memory.h 
head.obj	: head.h

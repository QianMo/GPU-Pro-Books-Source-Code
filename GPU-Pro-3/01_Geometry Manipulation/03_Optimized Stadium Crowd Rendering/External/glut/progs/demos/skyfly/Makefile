# Makefile for Win32
# Copyright (c) Mark J. Kilgard, 1997.

!include <win32.mak>

TOP  = ../../../
SRCS = skyfly.c

!include "$(TOP)/glutwin32.mak"

# dependencies
skyfly.exe	: database.obj fly.obj gm_main.obj image.obj perfdraw.obj random.obj
skyfly.obj	: skyfly.h

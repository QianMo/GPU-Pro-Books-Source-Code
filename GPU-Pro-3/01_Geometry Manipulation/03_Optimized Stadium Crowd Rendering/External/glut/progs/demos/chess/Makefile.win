# Makefile for Win32

!include <win32.mak>

TOP  = ../../..
SRCS = chess.c

!include "$(TOP)/glutwin32.mak"

# dependencies
chess.exe	: animate.obj main.obj pathplan.obj texture.obj
chess.obj	: chess.h

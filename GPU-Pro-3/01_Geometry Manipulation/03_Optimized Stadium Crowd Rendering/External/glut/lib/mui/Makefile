# Makefile for Win32

!include <win32.mak>

TOP = ../..

SRCS = gizmo.c glutmui.c displaylist.c uicolor.c miscui.c mui.c vslider.c textlist.c button.c pulldown.c hslider.c
OBJS = $(SRCS:.c=.obj)

all	: mui.lib

!include "$(TOP)/glutwin32.mak"

mui.lib	: $(OBJS)
	lib -out:mui.lib $(OBJS)

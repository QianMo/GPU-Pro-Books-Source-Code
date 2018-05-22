# Makefile for Win32

!include <win32.mak>

TOP = ../..

EXTRACFLAGS = -DOPENGL_10 -DAUTO_TEXTURE -D__GUTIL_DOUBLE

SRCS = ex_angle.c ex_cut_round.c ex_raw.c extrude.c intersect.c rot_prince.c rotate.c round_cap.c segment.c texgen.c urotate.c view.c
OBJS = $(SRCS:.c=.obj)

all	: gle.lib

!include "$(TOP)/glutwin32.mak"

gle.lib	: $(OBJS)
	lib -out:gle.lib $(OBJS)

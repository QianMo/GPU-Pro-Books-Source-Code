#!smake
#
# Copyright (c) Mark J. Kilgard, 1995, 1997.
#

TOP = ../..
include $(TOP)/glutdefs
include $(ROOT)/usr/include/make/commondefs

TARGETS = fscene sphere fbitfont example

LN = ln -s
MV = mv
RM = -rm -rf

LCOPTS = -I$(TOP)/include -fullwarn
LWOFF = ,813,852,827,826
LF77INCS = -I$(TOP)/include

LLDOPTS = -Wl,-ignore_unresolved
LLDLIBS = $(FGLUT) -lfGLU -lfGL $(GLUT) -lGLU -lGL -lXmu -lXi -lXext -lX11 -lm
LDIRT = *.fc *.s *.wo *~ *.pure

default: $(TARGETS)

include $(COMMONRULES)

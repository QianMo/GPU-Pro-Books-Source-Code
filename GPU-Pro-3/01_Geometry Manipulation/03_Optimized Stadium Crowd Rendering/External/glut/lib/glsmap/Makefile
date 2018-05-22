# Makefile for Win32

!include <win32.mak>

TOP = ../..

SRCS = smap_buildsmap.c smap_context.c smap_create.c smap_destroy.c smap_drawmesh.c smap_flag.c smap_makemesh.c smap_nearfar.c smap_rvec2st.c smap_set.c smap_setfunc.c smap_setvec.c smap_texdim.c smap_texobj.c
OBJS = $(SRCS:.c=.obj)

all	: glsmap.lib

!include "$(TOP)/glutwin32.mak"

glsmap.lib : $(OBJS)
	lib -out:glsmap.lib $(OBJS)

# Makefile for Win32

!include <win32.mak>

TOP  = ../..
SRCS = abgr.c bitfont.c blender.c cube.c dials.c dials2.c dinoball.c dinoshade.c dinospin.c evaltest.c fogtst.c fontdemo.c glpuzzle.c glutdino.c glutplane.c halomagic.c highlight.c lightlab.c molehill.c movelight.c oclip.c ohidden.c olight.c olympic.c origami.c oversphere.c sb2db.c scene.c screendoor.c scube.c simple.c sphere.c sphere2.c spots.c stars.c stenciltst.c stereo.c stroke.c subwin.c surfgrid.c texenv.c trippy.c triselect.c zoomdino.c reflectdino.c editgrid.c circlefit.c dinodraw.c resolution.c
# mjkwarp.c splatlogo.c

!include "$(TOP)/glutwin32.mak"

# dependencies
dinodraw.exe	\
dinospin.exe	\
glpuzzle.exe	: trackball.obj
mjkwarp.exe	: mjkimage.obj

splatlogo.exe	: splatlogo.obj logo.obj
	cl -o $@ splatlogo.obj logo.obj $(LIBS)

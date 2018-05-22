# Makefile for Win32

!include <win32.mak>

TOP  = ../..
SRCS = beam.c candlestick.c cone.c cylinder.c helix.c helix2.c helix3.c helix4.c helixtex.c horn.c joincone.c joinoffset.c martini.c screw.c taper.c texas.c twoid.c twistoid.c

EXTRALIBS = $(GLE)
DEPLIBS = $(GLE)

!include "$(TOP)/glutwin32.mak"

joincone.obj: cone.obj
	@del joincone.obj
	copy cone.obj joincone.obj

trackball.c:
	@del trackball.c
	copy ..\examples\trackball.c trackball.c

trackball.h:
	@del trackball.h
	copy ..\examples\trackball.h trackball.h

clobber::
	@del trackball.c
	@del trackball.h

# dependencies
beam.exe	: mainsimple.obj
cylinder.exe	: mainsimple.obj
cone.exe	: mainsimple.obj
helix.exe	: mainsimple.obj
helix2.exe	: mainsimple.obj
helix3.exe	: mainsimple.obj
helix4.exe	: mainsimple.obj
helixtex.exe	: texture.obj maintex.obj
horn.exe	: mainsimple.obj
joincone.exe	: mainjoin.obj
joinoffset.exe	: mainjoin.obj
martini.exe	: trackball.obj
screw.exe	: texture.obj maintex.obj
taper.exe	: mainsimple.obj
texas.exe	: mainjoin.obj
twistoid.exe	: texture.obj maintex.obj
trackball.obj	: trackball.h
martini.obj	: trackball.h

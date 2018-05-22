# Makefile for Win32

!include <win32.mak>

TOP  = ../..
SRCS = af_depthcue.c af_teapots.c accumaa.c boundary.c comp.c convolve.c csg.c decal.c dissolve.c envmap.c envphong.c field.c genmipmap.c haloed.c hello2rts.c hiddenline.c imgproc.c mipmap_lines.c motionblur.c multilight.c occlude.c pointburst.c projshadow.c projtex.c rasonly.c sgiflag.c shadowfun.c shadowmap.c shadowvol.c silhouette.c softshadow.c tess.c textext.c textile.c textrim.c tvertex.c vox.c warp.c zcomposite.c redblue_stereo.c texwinalign.c
# videoresize.c

DATA_LINKS = 00.rgb 02.rgb 04.rgb a.rgb mandrill.rgb 01.rgb 03.rgb 05.rgb b.rgb tree.rgb
# vox.bin.gz not included in DATA_LINKS list since requires 3D textures
LDIRT = $(DATA_LINKS)

!include "$(TOP)/glutwin32.mak"

# dependencies
comp.exe	\
envmap.exe	\
genmipmap.exe	\
imgproc.exe	\
projtex.exe	\
textile.exe	\
textrim.exe	\
warp.exe	: texture.obj
tess.exe	: sphere.obj
textext.exe	: textmap.obj texture.obj
af_depthcue.exe	\
af_teapots.exe	: addfog.obj
hello2rts.exe	: rts.obj
mipmap_lines.exe: izoom.obj texture.obj

rts.obj : rtshadow.h
hello2rts.obj : rtshadow.h

links:
	cd ..\..\data
	for %f in ($(DATA_LINKS)) do copy %f ..\progs\advanced
	cd ..\progs\advanced


#--------------------------------------------------------------------------
# Name         : content.mak
# Title        : Makefile to build content files
# Author       : Auto-generated
# Created      : 22/07/2009
#
# Copyright    : 2007 by Imagination Technologies.  All rights reserved.
#              : No part of this software, either material or conceptual 
#              : may be copied or distributed, transmitted, transcribed,
#              : stored in a retrieval system or translated into any 
#              : human or computer language in any form by any means,
#              : electronic, mechanical, manual or other-wise, or 
#              : disclosed to third parties without the express written
#              : permission of VideoLogic Limited, Unit 8, HomePark
#              : Industrial Estate, King's Langley, Hertfordshire,
#              : WD4 8LZ, U.K.
#
# Description  : Makefile to build content files for demos in the PowerVR SDK
#
# Platform     :
#
# $Revision: 1.5 $
#--------------------------------------------------------------------------

#############################################################################
## Variables
#############################################################################

PVRTEXTOOL 	= ..\..\..\Utilities\PVRTexTool\PVRTexToolCL\Win32\PVRTexTool.exe
FILEWRAP 	= ..\..\..\Utilities\Filewrap\Win32\Filewrap.exe
PVRUNISCO 	= ..\..\..\Utilities\PVRUniSCo\OGLES\Win32\PVRUniSCo.exe

MEDIAPATH = ../Media
CONTENTDIR = Content

#############################################################################
## Instructions
#############################################################################

TEXTURES = \
	Mask.pvr \
	TableCover.pvr \
	Torus.pvr

BIN_SHADERS = \
	FragShader.fsc \
	VertShader.vsc \
	ShadowFragShader.fsc \
	ShadowVertShader.vsc

RESOURCES = \
	$(CONTENTDIR)/Scene.cpp \
	$(CONTENTDIR)/Mask.cpp \
	$(CONTENTDIR)/TableCover.cpp \
	$(CONTENTDIR)/Torus.cpp \
	$(CONTENTDIR)/FragShader.cpp \
	$(CONTENTDIR)/VertShader.cpp \
	$(CONTENTDIR)/ShadowFragShader.cpp \
	$(CONTENTDIR)/ShadowVertShader.cpp

all: resources
	
help:
	@echo Valid targets are:
	@echo resources, textures, binary_shaders, clean
	@echo PVRTEXTOOL, FILEWRAP and PVRUNISCO can be used to override the default paths to these utilities.

clean:
	-rm $(RESOURCES)
	-rm $(BIN_SHADERS)
	-rm $(TEXTURES)

resources: 		$(CONTENTDIR) $(RESOURCES)
textures: 		$(TEXTURES)
binary_shaders:	$(BIN_SHADERS)

$(CONTENTDIR):
	-mkdir $@

Mask.pvr: $(MEDIAPATH)/Mask.bmp
	$(PVRTEXTOOL) -m -fOGLPVRTC4 -i$(MEDIAPATH)/Mask.bmp -o$@

TableCover.pvr: $(MEDIAPATH)/TableCover.bmp
	$(PVRTEXTOOL) -m -fOGLPVRTC4 -i$(MEDIAPATH)/TableCover.bmp -o$@

Torus.pvr: $(MEDIAPATH)/Torus.bmp
	$(PVRTEXTOOL) -m -fOGLPVRTC4 -i$(MEDIAPATH)/Torus.bmp -o$@

$(CONTENTDIR)/Scene.cpp: Scene.pod
	$(FILEWRAP)  -o $@ Scene.pod

$(CONTENTDIR)/Mask.cpp: Mask.pvr
	$(FILEWRAP)  -o $@ Mask.pvr

$(CONTENTDIR)/TableCover.cpp: TableCover.pvr
	$(FILEWRAP)  -o $@ TableCover.pvr

$(CONTENTDIR)/Torus.cpp: Torus.pvr
	$(FILEWRAP)  -o $@ Torus.pvr

$(CONTENTDIR)/FragShader.cpp: FragShader.fsh FragShader.fsc
	$(FILEWRAP)  -s  -o $@ FragShader.fsh
	$(FILEWRAP)  -oa $@ FragShader.fsc

$(CONTENTDIR)/VertShader.cpp: VertShader.vsh VertShader.vsc
	$(FILEWRAP)  -s  -o $@ VertShader.vsh
	$(FILEWRAP)  -oa $@ VertShader.vsc

$(CONTENTDIR)/ShadowFragShader.cpp: ShadowFragShader.fsh ShadowFragShader.fsc
	$(FILEWRAP)  -s  -o $@ ShadowFragShader.fsh
	$(FILEWRAP)  -oa $@ ShadowFragShader.fsc

$(CONTENTDIR)/ShadowVertShader.cpp: ShadowVertShader.vsh ShadowVertShader.vsc
	$(FILEWRAP)  -s  -o $@ ShadowVertShader.vsh
	$(FILEWRAP)  -oa $@ ShadowVertShader.vsc

FragShader.fsc: FragShader.fsh
	$(PVRUNISCO) FragShader.fsh $@  -f 

VertShader.vsc: VertShader.vsh
	$(PVRUNISCO) VertShader.vsh $@  -v 

ShadowFragShader.fsc: ShadowFragShader.fsh
	$(PVRUNISCO) ShadowFragShader.fsh $@  -f 

ShadowVertShader.vsc: ShadowVertShader.vsh
	$(PVRUNISCO) ShadowVertShader.vsh $@  -v 

############################################################################
# End of file (content.mak)
############################################################################

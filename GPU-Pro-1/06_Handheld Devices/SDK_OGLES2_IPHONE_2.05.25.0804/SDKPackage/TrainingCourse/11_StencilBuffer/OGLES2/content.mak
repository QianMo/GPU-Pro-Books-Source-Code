#--------------------------------------------------------------------------
# Name         : content.mak
# Title        : Makefile to build content files
# Author       : Auto-generated
# Created      : 12/08/2009
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
# $Revision: 1.4 $
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
	Lattice.pvr \
	Stone.pvr \
	Tile.pvr

BIN_SHADERS = \
	FragShader.fsc \
	VertShader.vsc

RESOURCES = \
	$(CONTENTDIR)/Lattice.cpp \
	$(CONTENTDIR)/Stone.cpp \
	$(CONTENTDIR)/Tile.cpp \
	$(CONTENTDIR)/FragShader.cpp \
	$(CONTENTDIR)/VertShader.cpp \
	$(CONTENTDIR)/Cylinder.cpp \
	$(CONTENTDIR)/Sphere.cpp

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

Lattice.pvr: $(MEDIAPATH)/cylinder.bmp $(MEDIAPATH)/cylindera.bmp
	$(PVRTEXTOOL) -nt -fOGL4444 -i$(MEDIAPATH)/cylinder.bmp -a$(MEDIAPATH)/cylindera.bmp -o$@

Stone.pvr: $(MEDIAPATH)/stone.bmp
	$(PVRTEXTOOL) -m -fOGLPVRTC4 -i$(MEDIAPATH)/stone.bmp -o$@

Tile.pvr: $(MEDIAPATH)/tile.bmp
	$(PVRTEXTOOL) -m -fOGLPVRTC4 -i$(MEDIAPATH)/tile.bmp -o$@

$(CONTENTDIR)/Lattice.cpp: Lattice.pvr
	$(FILEWRAP)  -o $@ Lattice.pvr

$(CONTENTDIR)/Stone.cpp: Stone.pvr
	$(FILEWRAP)  -o $@ Stone.pvr

$(CONTENTDIR)/Tile.cpp: Tile.pvr
	$(FILEWRAP)  -o $@ Tile.pvr

$(CONTENTDIR)/FragShader.cpp: FragShader.fsh FragShader.fsc
	$(FILEWRAP)  -s  -o $@ FragShader.fsh
	$(FILEWRAP)  -oa $@ FragShader.fsc

$(CONTENTDIR)/VertShader.cpp: VertShader.vsh VertShader.vsc
	$(FILEWRAP)  -s  -o $@ VertShader.vsh
	$(FILEWRAP)  -oa $@ VertShader.vsc

$(CONTENTDIR)/Cylinder.cpp: Cylinder.pod
	$(FILEWRAP)  -o $@ Cylinder.pod

$(CONTENTDIR)/Sphere.cpp: Sphere.pod
	$(FILEWRAP)  -o $@ Sphere.pod

FragShader.fsc: FragShader.fsh
	$(PVRUNISCO) FragShader.fsh $@  -f 

VertShader.vsc: VertShader.vsh
	$(PVRUNISCO) VertShader.vsh $@  -v 

############################################################################
# End of file (content.mak)
############################################################################

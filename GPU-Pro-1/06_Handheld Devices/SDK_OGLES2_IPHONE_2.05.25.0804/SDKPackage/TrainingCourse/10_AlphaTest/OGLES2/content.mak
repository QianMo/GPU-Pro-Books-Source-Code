#--------------------------------------------------------------------------
# Name         : content.mak
# Title        : Makefile to build content files
# Author       : Auto-generated
# Created      : 31/01/2008
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
	Wallwire.pvr

BIN_SHADERS = \
	TexFragShader.fsc \
	DiscardFragShader.fsc \
	VertShader.vsc

RESOURCES = \
	$(CONTENTDIR)/Wallwire.cpp \
	$(CONTENTDIR)/TexFragShader.cpp \
	$(CONTENTDIR)/DiscardFragShader.cpp \
	$(CONTENTDIR)/VertShader.cpp

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

Wallwire.pvr: $(MEDIAPATH)/wallwire.bmp $(MEDIAPATH)/twallwire.bmp
	$(PVRTEXTOOL) -nt -fOGL8888 -i$(MEDIAPATH)/wallwire.bmp -a$(MEDIAPATH)/twallwire.bmp -o$@

$(CONTENTDIR)/Wallwire.cpp: Wallwire.pvr
	$(FILEWRAP)  -o $@ Wallwire.pvr

$(CONTENTDIR)/TexFragShader.cpp: TexFragShader.fsh TexFragShader.fsc
	$(FILEWRAP)  -s  -o $@ TexFragShader.fsh
	$(FILEWRAP)  -oa $@ TexFragShader.fsc

$(CONTENTDIR)/DiscardFragShader.cpp: DiscardFragShader.fsh DiscardFragShader.fsc
	$(FILEWRAP)  -s  -o $@ DiscardFragShader.fsh
	$(FILEWRAP)  -oa $@ DiscardFragShader.fsc

$(CONTENTDIR)/VertShader.cpp: VertShader.vsh VertShader.vsc
	$(FILEWRAP)  -s  -o $@ VertShader.vsh
	$(FILEWRAP)  -oa $@ VertShader.vsc

TexFragShader.fsc: TexFragShader.fsh
	$(PVRUNISCO) TexFragShader.fsh $@  -f 

DiscardFragShader.fsc: DiscardFragShader.fsh
	$(PVRUNISCO) DiscardFragShader.fsh $@  -f 

VertShader.vsc: VertShader.vsh
	$(PVRUNISCO) VertShader.vsh $@  -v 

############################################################################
# End of file (content.mak)
############################################################################

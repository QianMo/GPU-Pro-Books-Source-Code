#--------------------------------------------------------------------------
# Name         : content.mak
# Title        : Makefile to build content files
# Author       : Auto-generated
# Created      : 21/11/2007
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
# $Revision: 1.2 $
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
	Basetex.pvr \
	Reflection.pvr \
	Cubemap.pvr \
	AnisoMap.pvr

BIN_SHADERS =

RESOURCES = \
	$(CONTENTDIR)/Basetex.cpp \
	$(CONTENTDIR)/Reflection.cpp \
	$(CONTENTDIR)/Cubemap.cpp \
	$(CONTENTDIR)/AnisoMap.cpp \
	$(CONTENTDIR)/anisotropic_lighting.cpp \
	$(CONTENTDIR)/directional_lighting.cpp \
	$(CONTENTDIR)/envmap.cpp \
	$(CONTENTDIR)/fasttnl.cpp \
	$(CONTENTDIR)/lattice.cpp \
	$(CONTENTDIR)/phong_lighting.cpp \
	$(CONTENTDIR)/point_lighting.cpp \
	$(CONTENTDIR)/reflections.cpp \
	$(CONTENTDIR)/simple.cpp \
	$(CONTENTDIR)/spot_lighting.cpp \
	$(CONTENTDIR)/toon.cpp \
	$(CONTENTDIR)/vertex_sine.cpp \
	$(CONTENTDIR)/wood.cpp

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

Basetex.pvr: $(MEDIAPATH)/Base.bmp
	$(PVRTEXTOOL) -m -nt -fOGL565 -i$(MEDIAPATH)/Base.bmp -o$@

Reflection.pvr: $(MEDIAPATH)/Reflection.bmp
	$(PVRTEXTOOL) -m -nt -fOGL565 -i$(MEDIAPATH)/Reflection.bmp -o$@

Cubemap.pvr: $(MEDIAPATH)/cubemap1.bmp $(MEDIAPATH)/cubemap2.bmp $(MEDIAPATH)/cubemap3.bmp $(MEDIAPATH)/cubemap4.bmp $(MEDIAPATH)/cubemap5.bmp $(MEDIAPATH)/cubemap6.bmp
	$(PVRTEXTOOL) -s -m -fOGLPVRTC4 -i$(MEDIAPATH)/cubemap1.bmp -o$@

AnisoMap.pvr: $(MEDIAPATH)/anisotropicmap.bmp
	$(PVRTEXTOOL) -m -nt -fOGL565 -i$(MEDIAPATH)/anisotropicmap.bmp -o$@

$(CONTENTDIR)/Basetex.cpp: Basetex.pvr
	$(FILEWRAP)  -o $@ Basetex.pvr

$(CONTENTDIR)/Reflection.cpp: Reflection.pvr
	$(FILEWRAP)  -o $@ Reflection.pvr

$(CONTENTDIR)/Cubemap.cpp: Cubemap.pvr
	$(FILEWRAP)  -o $@ Cubemap.pvr

$(CONTENTDIR)/AnisoMap.cpp: AnisoMap.pvr
	$(FILEWRAP)  -o $@ AnisoMap.pvr

$(CONTENTDIR)/anisotropic_lighting.cpp: anisotropic_lighting.pfx
	$(FILEWRAP)  -s  -o $@ anisotropic_lighting.pfx

$(CONTENTDIR)/directional_lighting.cpp: directional_lighting.pfx
	$(FILEWRAP)  -s  -o $@ directional_lighting.pfx

$(CONTENTDIR)/envmap.cpp: envmap.pfx
	$(FILEWRAP)  -s  -o $@ envmap.pfx

$(CONTENTDIR)/fasttnl.cpp: fasttnl.pfx
	$(FILEWRAP)  -s  -o $@ fasttnl.pfx

$(CONTENTDIR)/lattice.cpp: lattice.pfx
	$(FILEWRAP)  -s  -o $@ lattice.pfx

$(CONTENTDIR)/phong_lighting.cpp: phong_lighting.pfx
	$(FILEWRAP)  -s  -o $@ phong_lighting.pfx

$(CONTENTDIR)/point_lighting.cpp: point_lighting.pfx
	$(FILEWRAP)  -s  -o $@ point_lighting.pfx

$(CONTENTDIR)/reflections.cpp: reflections.pfx
	$(FILEWRAP)  -s  -o $@ reflections.pfx

$(CONTENTDIR)/simple.cpp: simple.pfx
	$(FILEWRAP)  -s  -o $@ simple.pfx

$(CONTENTDIR)/spot_lighting.cpp: spot_lighting.pfx
	$(FILEWRAP)  -s  -o $@ spot_lighting.pfx

$(CONTENTDIR)/toon.cpp: toon.pfx
	$(FILEWRAP)  -s  -o $@ toon.pfx

$(CONTENTDIR)/vertex_sine.cpp: vertex_sine.pfx
	$(FILEWRAP)  -s  -o $@ vertex_sine.pfx

$(CONTENTDIR)/wood.cpp: wood.pfx
	$(FILEWRAP)  -s  -o $@ wood.pfx

############################################################################
# End of file (content.mak)
############################################################################

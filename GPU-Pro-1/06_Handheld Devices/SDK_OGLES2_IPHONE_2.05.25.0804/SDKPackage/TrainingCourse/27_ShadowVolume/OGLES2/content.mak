#--------------------------------------------------------------------------
# Name         : content.mak
# Title        : Makefile to build content files
# Author       : Auto-generated
# Created      : 20/08/2007
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
# $Revision: 1.3 $
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
	Background.pvr \
	Rust.pvr

BIN_SHADERS = \
	BaseFragShader.fsc \
	BaseVertShader.vsc \
	ConstFragShader.fsc \
	ShadowVolVertShader.vsc \
	FullscreenVertShader.vsc

RESOURCES = \
	$(CONTENTDIR)/Background.cpp \
	$(CONTENTDIR)/Rust.cpp \
	$(CONTENTDIR)/BaseFragShader.cpp \
	$(CONTENTDIR)/BaseVertShader.cpp \
	$(CONTENTDIR)/ConstFragShader.cpp \
	$(CONTENTDIR)/ShadowVolVertShader.cpp \
	$(CONTENTDIR)/FullscreenVertShader.cpp \
	$(CONTENTDIR)/scene.cpp

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

Background.pvr: $(MEDIAPATH)/background.bmp
	$(PVRTEXTOOL) -m -nt -fOGLPVRTC4 -i$(MEDIAPATH)/background.bmp -o$@

Rust.pvr: $(MEDIAPATH)/rust.bmp
	$(PVRTEXTOOL) -m -nt -fOGLPVRTC4 -i$(MEDIAPATH)/rust.bmp -o$@

$(CONTENTDIR)/Background.cpp: Background.pvr
	$(FILEWRAP)  -o $@ Background.pvr

$(CONTENTDIR)/Rust.cpp: Rust.pvr
	$(FILEWRAP)  -o $@ Rust.pvr

$(CONTENTDIR)/BaseFragShader.cpp: BaseFragShader.fsh BaseFragShader.fsc
	$(FILEWRAP)  -s  -o $@ BaseFragShader.fsh
	$(FILEWRAP)  -oa $@ BaseFragShader.fsc

$(CONTENTDIR)/BaseVertShader.cpp: BaseVertShader.vsh BaseVertShader.vsc
	$(FILEWRAP)  -s  -o $@ BaseVertShader.vsh
	$(FILEWRAP)  -oa $@ BaseVertShader.vsc

$(CONTENTDIR)/ConstFragShader.cpp: ConstFragShader.fsh ConstFragShader.fsc
	$(FILEWRAP)  -s  -o $@ ConstFragShader.fsh
	$(FILEWRAP)  -oa $@ ConstFragShader.fsc

$(CONTENTDIR)/ShadowVolVertShader.cpp: ShadowVolVertShader.vsh ShadowVolVertShader.vsc
	$(FILEWRAP)  -s  -o $@ ShadowVolVertShader.vsh
	$(FILEWRAP)  -oa $@ ShadowVolVertShader.vsc

$(CONTENTDIR)/FullscreenVertShader.cpp: FullscreenVertShader.vsh FullscreenVertShader.vsc
	$(FILEWRAP)  -s  -o $@ FullscreenVertShader.vsh
	$(FILEWRAP)  -oa $@ FullscreenVertShader.vsc

$(CONTENTDIR)/scene.cpp: scene.pod
	$(FILEWRAP)  -o $@ scene.pod

BaseFragShader.fsc: BaseFragShader.fsh
	$(PVRUNISCO) BaseFragShader.fsh $@  -f 

BaseVertShader.vsc: BaseVertShader.vsh
	$(PVRUNISCO) BaseVertShader.vsh $@  -v 

ConstFragShader.fsc: ConstFragShader.fsh
	$(PVRUNISCO) ConstFragShader.fsh $@  -f 

ShadowVolVertShader.vsc: ShadowVolVertShader.vsh
	$(PVRUNISCO) ShadowVolVertShader.vsh $@  -v 

FullscreenVertShader.vsc: FullscreenVertShader.vsh
	$(PVRUNISCO) FullscreenVertShader.vsh $@  -v 

############################################################################
# End of file (content.mak)
############################################################################

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
	Basetex.pvr

BIN_SHADERS = \
	FastFragShader.fsc \
	FastVertShader.vsc \
	SlowFragShader.fsc \
	SlowVertShader.vsc

RESOURCES = \
	$(CONTENTDIR)/Basetex.cpp \
	$(CONTENTDIR)/FastFragShader.cpp \
	$(CONTENTDIR)/FastVertShader.cpp \
	$(CONTENTDIR)/SlowFragShader.cpp \
	$(CONTENTDIR)/SlowVertShader.cpp \
	$(CONTENTDIR)/Mask.cpp

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

Basetex.pvr: $(MEDIAPATH)/tex_base.bmp
	$(PVRTEXTOOL) -m -fOGLPVRTC4 -i$(MEDIAPATH)/tex_base.bmp -o$@

$(CONTENTDIR)/Basetex.cpp: Basetex.pvr
	$(FILEWRAP)  -o $@ Basetex.pvr

$(CONTENTDIR)/FastFragShader.cpp: FastFragShader.fsh FastFragShader.fsc
	$(FILEWRAP)  -s  -o $@ FastFragShader.fsh
	$(FILEWRAP)  -oa $@ FastFragShader.fsc

$(CONTENTDIR)/FastVertShader.cpp: FastVertShader.vsh FastVertShader.vsc
	$(FILEWRAP)  -s  -o $@ FastVertShader.vsh
	$(FILEWRAP)  -oa $@ FastVertShader.vsc

$(CONTENTDIR)/SlowFragShader.cpp: SlowFragShader.fsh SlowFragShader.fsc
	$(FILEWRAP)  -s  -o $@ SlowFragShader.fsh
	$(FILEWRAP)  -oa $@ SlowFragShader.fsc

$(CONTENTDIR)/SlowVertShader.cpp: SlowVertShader.vsh SlowVertShader.vsc
	$(FILEWRAP)  -s  -o $@ SlowVertShader.vsh
	$(FILEWRAP)  -oa $@ SlowVertShader.vsc

$(CONTENTDIR)/Mask.cpp: Mask.pod
	$(FILEWRAP)  -o $@ Mask.pod

FastFragShader.fsc: FastFragShader.fsh
	$(PVRUNISCO) FastFragShader.fsh $@  -f 

FastVertShader.vsc: FastVertShader.vsh
	$(PVRUNISCO) FastVertShader.vsh $@  -v 

SlowFragShader.fsc: SlowFragShader.fsh
	$(PVRUNISCO) SlowFragShader.fsh $@  -f 

SlowVertShader.vsc: SlowVertShader.vsh
	$(PVRUNISCO) SlowVertShader.vsh $@  -v 

############################################################################
# End of file (content.mak)
############################################################################

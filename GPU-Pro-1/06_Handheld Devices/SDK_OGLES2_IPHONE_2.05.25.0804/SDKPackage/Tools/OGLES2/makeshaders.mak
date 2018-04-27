#--------------------------------------------------------------------------
# Name         : makeshaders.mak
# Title        : Makefile to build resource files
# Author       : PowerVR
# Created      : 13/07/2007
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
# Description  : Makefile for shaders in PowerVR SDK Tools
#
# Platform     : GNU make / MS Nmake
#
# $Revision: 1.2 $
#--------------------------------------------------------------------------

#############################################################################
## Variables
#############################################################################

FILEWRAP 	= ..\..\Utilities\Filewrap\Win32\Filewrap.exe
PVRUNISCO 	= ..\..\Utilities\PVRUniSCo\OGLES\Win32\PVRUniSCo.exe

#############################################################################
## Instructions
#############################################################################

MB_SRC_SHADERS = \
	BackgroundFragShader.fsh \
	BackgroundVertShader.vsh

MB_BIN_SHADERS = \
	BackgroundFragShader.fsc \
	BackgroundVertShader.vsc

P3D_SRC_SHADERS = \
	Print3DFragShader.fsh \
	Print3DVertShader.vsh

P3D_BIN_SHADERS = \
	Print3DFragShader.fsc \
	Print3DVertShader.vsc

RESOURCES = \
	PVRTBackgroundShaders.h \
	PVRTPrint3DShaders.h

all: $(RESOURCES)
	
help:
	@echo Valid targets are:
	@echo all, binary_shaders, clean
	@echo FILEWRAP and PVRUNISCO can be used to override the default paths to these utilities.

clean:
	-rm $(RESOURCES) $(MB_BIN_SHADERS) $(P3D_BIN_SHADERS)

binary_shaders:	$(MB_BIN_SHADERS) $(P3D_BIN_SHADERS)

PVRTPrint3DShaders.h: $(P3D_SRC_SHADERS) $(P3D_BIN_SHADERS)
	$(FILEWRAP) -h -s -o $@ $(P3D_SRC_SHADERS)
	$(FILEWRAP) -h -oa $@ $(P3D_BIN_SHADERS)

PVRTBackgroundShaders.h: $(MB_SRC_SHADERS) $(MB_BIN_SHADERS)
	$(FILEWRAP) -h -s -o $@ $(MB_SRC_SHADERS)
	$(FILEWRAP) -h -oa $@ $(MB_BIN_SHADERS)
	
BackgroundFragShader.fsc: BackgroundFragShader.fsh
	$(PVRUNISCO) BackgroundFragShader.fsh $@ -f

BackgroundVertShader.vsc: BackgroundVertShader.vsh
	$(PVRUNISCO) BackgroundVertShader.vsh $@ -v

Print3DFragShader.fsc: Print3DFragShader.fsh
	$(PVRUNISCO) Print3DFragShader.fsh $@ -f

Print3DVertShader.vsc: Print3DVertShader.vsh
	$(PVRUNISCO) Print3DVertShader.vsh $@ -v
	
############################################################################
# End of file (makeshaders.mak)
############################################################################
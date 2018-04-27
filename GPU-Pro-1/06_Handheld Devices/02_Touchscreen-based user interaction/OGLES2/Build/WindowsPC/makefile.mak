#--------------------------------------------------------------------------
# Name         : makefile.mak
# Title        : PowerVR SDK demo makefile
# Author       : Auto-generated
# Created      : 21/11/2008
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
# Description  : Makefile for demos in the PowerVR SDK
#
# Platform     : WindowsPC/NMake
#
# $Revision: 1.7 $
#--------------------------------------------------------------------------

#############################################################################
## Variables
#############################################################################

PROJECTDIR = TrainingCourse
API = OGLES2
PROJECTNAME = IntroducingPODTouch

OBJFILES = \
	$(OUTDIR)\tex_base.obj \
	$(OUTDIR)\tex_arm.obj \
	$(OUTDIR)\FragShader.obj \
	$(OUTDIR)\VertShader.obj \
	$(OUTDIR)\Scene.obj \
	$(OUTDIR)\MDKCamera.obj \
	$(OUTDIR)\MDKMath.obj \
	$(OUTDIR)\MDKPrecisionTimer.obj \
	$(OUTDIR)\MDKTouch.obj \
	

#############################################################################
## Business
#############################################################################

!INCLUDE ..\..\..\..\..\Builds\WindowsVC2003\makedemo.mak

############################################################################
# End of file (makefile.mak)
############################################################################

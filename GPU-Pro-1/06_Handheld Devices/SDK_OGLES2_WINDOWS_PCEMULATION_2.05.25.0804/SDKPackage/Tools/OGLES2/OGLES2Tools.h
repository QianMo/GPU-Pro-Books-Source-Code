/******************************************************************************

 @File         OGLES2Tools.h

 @Title        OGLES2Tools

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     ANSI compatible

 @Description  Header file of OGLES2Tools.lib.

******************************************************************************/
#ifndef _OGLES2TOOLS_H_
#define _OGLES2TOOLS_H_

/*****************************************************************************/
/*! @mainpage OGLES2Tools
******************************************************************************
@section _tools_sec1 Overview
*****************************

OGLES2Tools is a collection of source code to help developers with some common
tasks which are frequently used in 3D programming.
OGLES2Tools supplies code for mathematical operations, matrix handling,
loading 3D models and to optimise geometry.
The API specific tools contain code for displaying text and loading textures.

@section _tools_sec2 Content
*****************************
This is a description of the files which compose OGLES2Tools. Not all the files might have been released for
your platform so check the file list to know what is available.

\b PVRTBoneBatch: Group vertices per bones to allow skinning when the maximum number of bones is limited.

\b PVRTDecompress: Descompress PVRTC texture format.

\b PVRTFixedPoint: Fast fixed point mathematical functions.

\b PVRTMatrix: Vector and Matrix functions.

\b PVRTVector: Vector and Matrix functions that are gradually replacing PVRTMatrix.

\b PVRTQuaternion: Quaternion functions.

\b PVRTResourceFile: The tools code for loading files included using FileWrap.

\b PVRTMisc: Skybox, line plane intersection code, etc...

\b PVRTModelPOD: Load geometry and animation from a POD file.

\b PVRTTrans: Transformation and projection functions.

\b PVRTTriStrip: Geometry optimization using strips.

\b PVRTVertex.cpp: Vertex order optimisation for 3D acceleration.

\b PVRTPrint3D: Display text/logos on the screen.

\b PVRTTexture: Load textures from resources, BMP or PVR files.

\b PVRTBackground: Create a textured background.

\b PVRTError: Error codes and tools output debug.

\b PVRTShadowVol: Tools code for creating shadow volumes.

\b PVRTString: A string class.

\b PVRTShader: Code to load vertex, pixel and geometry shaders.

\b PVRTPFXParser: Code to parse our PFX file format.
*/

#ifndef BUILD_OGLES2
	#define BUILD_OGLES2
#endif

#include "PVRTContext.h"
#include "../PVRTGlobal.h"
#include "../PVRTVector.h"
#include "../PVRTString.h"
#include "../PVRTFixedPoint.h"
#include "../PVRTMatrix.h"
#include "../PVRTQuaternion.h"
#include "../PVRTTrans.h"
#include "../PVRTVertex.h"
#include "../PVRTMisc.h"
#include "../PVRTBackground.h"
#include "PVRTgles2Ext.h"
#include "../PVRTPrint3D.h"
#include "../PVRTBoneBatch.h"
#include "../PVRTModelPOD.h"
#include "../PVRTTexture.h"
#include "PVRTTextureAPI.h"
#include "../PVRTTriStrip.h"
#include "PVRTShader.h"
#include "../PVRTPFXParser.h"
#include "PVRTPFXParserAPI.h"
#include "../PVRTShadowVol.h"
#include "../PVRTResourceFile.h"
#include "../PVRTError.h"

#endif /* _OGLES2TOOLS_H_ */

/*****************************************************************************
 End of file (OGLES2Tools.h)
*****************************************************************************/

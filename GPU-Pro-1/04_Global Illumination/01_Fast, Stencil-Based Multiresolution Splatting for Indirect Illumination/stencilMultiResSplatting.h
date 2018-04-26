/***************************************************************************/
/* causticDemo.h                                                           */
/* -----------------------                                                 */
/*                                                                         */
/* The header file for using the basic scene loader in OpenGL programs.    */
/*     This scene loader will parse the exact same scene files used in my  */
/*     basic ray tracing framework, allowing direct comparisons using the  */
/*     exact same scene functions.                                         */
/*                                                                         */
/* Chris Wyman (02/01/2008)                                                */
/***************************************************************************/

#ifndef __SCENELOADER_H__
#define __SCENELOADER_H__

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "Utils/GLee.h"
#include <GL/glut.h>
#include <GL/glu.h>

#include "DataTypes/Array1D.h"
#include "DataTypes/Color.h"
#include "DataTypes/MathDefs.h"
#include "DataTypes/Matrix4x4.h"
#include "DataTypes/Point.h"
#include "DataTypes/Vector.h"
#include "DataTypes/glTexture.h"

#include "Materials/Material.h"
#include "Materials/GLMaterial.h"

#include "Objects/Group.h"
#include "Objects/Object.h"

#include "Utils/drawTextToGLWindow.h"
#include "Utils/frameRate.h"
#include "Utils/ProgramPathLists.h"
#include "Utils/searchPathList.h"
#include "Utils/TextParsing.h"
#include "Utils/Trackball.h"
#include "Utils/ImageIO/imageIO.h"
#include "Utils/framebufferObject.h"
#include "Utils/frameGrab.h"
#include "Utils/VideoIO/MovieMaker.h"
#include "Utils/VideoIO/videoReader.h"
#include "Utils/Random.h"

#include "Scene/Camera.h"
#include "Scene/glLight.h"
#include "Scene/Scene.h"

// GLUT callback functions
void ReshapeCallback ( int w, int h );
void IdleCallback ( void );
void MouseMotionCallback ( int x, int y );
void MouseButtonCallback ( int b, int st, int x, int y );
void KeyboardCallback ( unsigned char key, int x, int y );
void SpecialKeyboardCallback ( int key, int x, int y );


void DisplayLayerOfTextureArray( GLuint texArrayID, int layerID );
void SetupHelpScreen( FrameBuffer *helpFB );
void DisplayHelpScreen( FrameBuffer *helpFB );


#ifndef BUFFER_OFFSET 
#define BUFFER_OFFSET(i) ((char *)NULL + (i))
#endif

#endif




// Source code for the demo accompanying "GPU Pro" article 
// "Polygonal-Functional Hybrids for Computer Animation and Games"
// Contacts: deniskravtsov@gmail.com

// This file is derived from the NVIDIA CUDA SDK example 'marchingCubes'
// You will need  NVIDIA CUDA SDK release 2.1 and CUDPP 1.0a (be sure to
// have $(CUDA_INC_PATH) and $(NVSDKCUDA_ROOT) env variables set correctly)
// The easiest way would be to copy the directory with the source code to
// your "$(NVSDKCUDA_ROOT)/projects" directory, otherwise you will have
// to set CUDA custom build step manually

// Uses stbi-1.18 - public domain JPEG/PNG reader

/*
* Copyright 1993-2007 NVIDIA Corporation.  All rights reserved.
*
* NOTICE TO USER:
*
* This source code is subject to NVIDIA ownership rights under U.S. and
* international Copyright laws.  Users and possessors of this source code
* are hereby granted a nonexclusive, royalty-free license to use this code
* in individual and commercial software.
*
* NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
* CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
* IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
* REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
* MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
* IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
* OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
* OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
* OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE
* OR PERFORMANCE OF THIS SOURCE CODE.
*
* U.S. Government End Users.   This source code is a "commercial item" as
* that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of
* "commercial computer  software"  and "commercial computer software
* documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995)
* and is provided to the U.S. Government only as a commercial end item.
* Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
* 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
* source code with only those rights set forth herein.
*
* Any use of this source code in individual and commercial software must
* include, in the user documentation and internal comments to the code,
* the above Disclaimer and U.S. Government End Users Notice.
*/

/*

 All main CUDA computations are invoked in FREP_EVALUATOR::evaluate(...)


1. For each cell \label{algo1:write_cell}	
   1.1   Write out the number of vertices it contains 
	1.2	Write out the flag indicating whether it contains any geometry                 

2. Find the number of non-empty cells
3. Create a group of all non-empty cells using the flags information from previous steps
4. Generate the table of vertex buffer offsets for non-empty cells 
5. For each non-empty cell 
   5.1   Find the number of vertices it outputs
   5.2   Generate vertices of the triangles being output from the cell
	5.3   Generate normal for each vertex being output 
   5.4   Save vertices and normals using offset generated at step 4

More info and explanations are provided in the book
 
 */

#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <GL/glew.h>
#include <assert.h>

#include <vector_types.h>
#include <vector_functions.h>
#include <cuda_runtime_api.h>
#include <cutil_inline.h>

#include "defines.h"

#include "FRep/evaluator.h"
#include "GL/renderer.h"

#include "animTimer.h"

#include "functions.h"
#include "loading/frepLoader.h"

#include "loading/skinLoader.h"

#include	"skin/skin.h"
#include "skin/animationData.h"

#include "menu/optionsMenu.h"
#include "mouse/mouse.h"

// invokes all CUDA code and extract the isosurface
FREP_EVALUATOR frepEvaluator;

// a few global variables changed by the user
float rotation    [] =  {0.0f, 30.0f, 0.0f};
float translation [] =  {0.0f, -14.f, -45.0f};

const char* APP_NAME = "CUDA FRep";

// some resources used in the demo:
const char* sceneNames[] = {
   "sinking",      
   "walking",   
   "bathing-2",
   "mirror",   
   "desert",
   "hybrid-attach",
   "tentacle",
   "hand-mirror",         
   "monster",  
};

const char* shaderNames[] = {
   "blinn",
   "cubemap",
   "triplanar",
   "procedural",
};

const char* textureNames[] = {
   "organic-2.bmp",
   "asphalt.bmp",
   "blue.bmp",
   "grass.bmp",
   "military.bmp",   
   "organic-1.bmp",
   "sand.bmp",
   "web.bmp",
   "girlVest.jpg"
};

const char* cubemapNames[] = {
   "terrain.png",
   "autumn.png",
   "waterscape.png"
};

const char* texture3DNames[] = {
   "noise.vol",
};

const char* resolutionNames[] = {
   "16 X 16 X 16",
   "32 X 32 X 32",
   "64 X 64 X 64",
   "128 X 128 X 128",
};

struct SCENE_DATA {   

	FREP_DESCRIPTION  frepModel;
	SKIN_DATA         skinData;   
	ANIMATION_DATA		animationData;
};

SCENE_DATA scenes[NUM_ELEMS(sceneNames)];

// all the loading is rather slow, but should be simple + human readable files
bool loadData(LIST_DESC scenesInfo, SCENE_DATA* scenes, int scenesSize)
{  
   if (scenesInfo.itemsSize != scenesSize) {
      assert(0);
      return false;
   }

   for (int i = 0; i < scenesInfo.itemsSize; i++) {

      std::string fileName = std::string("scenes/") + scenesInfo.itemNames[i];      

      printf("%*s - ", 20, fileName.c_str());

      SCENE_DATA& scene = scenes[i];
      
      SKIN_LOADER::RESULT loadingResult = SKIN_LOADER::loadData(fileName + ".txt", &scene.skinData, &scene.animationData);      
      if (loadingResult == SKIN_LOADER::WRONG_PARAMS || loadingResult == SKIN_LOADER::ERROR_LOADING) {
         assert(0);
         return false;
      }

      if (!FREP_LOADER::loadData(fileName + ".ani", &scene.frepModel)) {      
         assert(0);
         return false;
      }    

      printf("%d / %d\n", i + 1, scenesInfo.itemsSize);

   }   

   return true;
}

void displayCallback()
{
	MENU::ITEMS_RANGE& sceneItem = getMenu().getItem(MENU::GROUP_SCENE);

   SCENE_DATA& scene = scenes[ sceneItem ];

   if (sceneItem.isChanged()) {
      ANIMATION_TIMER::requestFrame(ANIMATION_TIMER::RESET_FRAME);
		sceneItem.resetChanged();		
   }

   static int currentFrame = 0;
   currentFrame = getMenu().getValue(MENU::ITEM_TOGGLE_ANIMATION) ? ANIMATION_TIMER::requestFrame() : currentFrame;   

   // fill vertices and normals   
   frepEvaluator.evaluate( scene.frepModel, getMenu().getItem(MENU::GROUP_RESOLUTION), currentFrame );  

   getRenderer().preRender(translation, rotation);

   // mesh 
   if ( getMenu().getValue(MENU::ITEM_TOGGLE_MESH) ) {

      // get last texture
      int currentTexture = getMenu().getItem(MENU::GROUP_TEXTURE).getSize() - 1;

      getRenderer().updateCurrentTechnique(RENDERER::TECHNIQUE_SKINNING, currentTexture);

      getRenderer().renderSkin(&scene.skinData, &scene.animationData, currentFrame);
   }

   // FRep
   if ( getMenu().getValue(MENU::ITEM_TOGGLE_FREP) ) {

      bool  isCubemapOn       =  getMenu().getValue(MENU::ITEM_CUBEMAP_SHADING);      
      bool  isTriplanarOn     =  getMenu().getValue(MENU::ITEM_TRIPLANAR_SHADING);
      bool  isProceduralOn    =  getMenu().getValue(MENU::ITEM_PROCEDURAL_SHADING);
      
      int   technique, texture;

      if (isCubemapOn) { 
         technique   =  RENDERER::TECHNIQUE_CUBEMAP;
         texture     =  getMenu().getItem(MENU::GROUP_CUBEMAP);
      } else if (isTriplanarOn) {

         technique   =  RENDERER::TECHNIQUE_TRIPLANAR;
         texture     =  getMenu().getItem(MENU::GROUP_TEXTURE);

      } else if (isProceduralOn) {

         technique   =  RENDERER::TECHNIQUE_PROCEDURAL;
         texture     =  0; // only 1 volume texture with noise

      } else {         
         assert(0);
         return;
      }
      
      bool  isWireframeOn = getMenu().getValue(MENU::ITEM_TOGGLE_WIREFRAME);

      getRenderer().updateCurrentTechnique((RENDERER::TECHNIQUE_TYPE)technique, texture, isWireframeOn);

      getRenderer().renderGeometry(frepEvaluator.getGeometryDesc(), scene.animationData.getAttachmentMatrix(currentFrame));      
   }   
  
   glutSwapBuffers();
}

void keyboardCallback(unsigned char key, int /*x*/, int /*y*/)
{
   static float moveDelta = 0.5f;

   switch(key) {
   case(27) :      
      frepEvaluator.clean();
      cudaThreadExit();
      exit(0);    
   case '`':        
      getMenu().toggleValue(MENU::ITEM_TOGGLE_WIREFRAME);
      break;
   case ' ':        
      getMenu().toggleValue(MENU::ITEM_TOGGLE_ANIMATION);
      break;     
   case 'w':
      translation[2] += moveDelta;
      break;
   case 's':
      translation[2] -= moveDelta;
      break;
   case 'a':
      translation[0] += moveDelta;
      break;
   case 'd':
      translation[0] -= moveDelta;
      break;  
   }

   int numCells = scenes[ getMenu().getItem(MENU::GROUP_SCENE) ].frepModel.polygonizationParams.numCells;

   glutPostRedisplay();
}

void idleCallback()
{
  glutPostRedisplay();
}

void reshapeCallback(int w, int h)
{
   glMatrixMode(GL_PROJECTION);
   glLoadIdentity();   
   gluPerspective(60.0, (float) w / (float) h, 0.2, 400.);

   glMatrixMode(GL_MODELVIEW);
   glViewport(0, 0, w, h);
}

const float FPS_ESTIMATION_PERIOD = 3000.f; // ms

void timerCallback(int value) 
{
	char message[STR_LENGTH_MAX];	

	sprintf(message, "%s FPS: %.1f", APP_NAME, getRenderer().getFramesRendered() / (FPS_ESTIMATION_PERIOD / 1000.f));
	glutSetWindowTitle(message);

	glutTimerFunc(FPS_ESTIMATION_PERIOD, timerCallback, 1);

   getRenderer().resetFrames();
}

int main(int argc, char** argv)
{  
   if ( !getRenderer().init(800, 600, APP_NAME, &argc, argv) ) {
      return -1;
   } 
   

   getRenderer().initShaders( MAKE_LIST(shaderNames), "shaders");   
   getRenderer().initTextures( RENDERER::TEXTURE_2D, MAKE_LIST(textureNames), "textures" );
   getRenderer().initTextures( RENDERER::TEXTURE_CUBEMAP, MAKE_LIST(cubemapNames), "textures" );
   getRenderer().initTextures( RENDERER::TEXTURE_3D, MAKE_LIST(texture3DNames), "textures" );
    

   printf("WSAD - move\nMouse - rotate\nMouse right button - menu\n\nLoading examples...\n");

   // register callbacks
   glutDisplayFunc(displayCallback);
   glutKeyboardFunc(keyboardCallback);   
   glutIdleFunc(idleCallback);
   glutReshapeFunc(reshapeCallback);

   loadData( MAKE_LIST(sceneNames), scenes, NUM_ELEMS(scenes) );

   // init menus
   getMenu().init(	MAKE_LIST(sceneNames),
                     MAKE_LIST(textureNames),
                     MAKE_LIST(cubemapNames),
                     MAKE_LIST(resolutionNames)
                  );

   getMouse().init(rotation, translation);

   frepEvaluator.init();

   glutTimerFunc(FPS_ESTIMATION_PERIOD, timerCallback, 1);
 
   // start rendering mainloop
   glutMainLoop();

   frepEvaluator.clean();
   cudaThreadExit();

   cutilExit(argc, argv);
}

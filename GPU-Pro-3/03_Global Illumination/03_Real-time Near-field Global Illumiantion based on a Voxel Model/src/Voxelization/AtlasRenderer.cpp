///////////////////////////////////////////////////////////////////////
// Real-time Near-Field Global Illumination based on a Voxel Model
// GPU Pro 3
///////////////////////////////////////////////////////////////////////
// Authors : Sinje Thiedemann, Niklas Henrich, 
//           Thorsten Grosch, Stefan Mueller
// Created:  August 2011
///////////////////////////////////////////////////////////////////////

#include "AtlasRenderer.h"

#include "Scene/ObjectSequence.h"
#include "Scene/ObjModel.h"
#include "Scene/Scene.h"

#include "Utils/EmptyTexture.h"
#include "Utils/GLError.h"
#include "Utils/ShaderProgram.h" 
#include "Utils/TexturePool.h"

AtlasRenderer::AtlasRenderer()
{
   createFBO();
   createShader();

   initialAtlasRendering(); 
}

void AtlasRenderer::createFBO()
{
   // Frame Buffer Object for rendering to Texture Atlas

   for(unsigned int e = 0; e < SCENE->getSceneElements().size(); e++)
   {
      createAtlas(e, true);
      mPixelDisplayLists.push_back(0);
   }

   V(glGenFramebuffersEXT(1, &mFBOAtlas));
   V(glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, mFBOAtlas));

   // attach first element's atlas to color attachment points
   glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, mAtlases.front().front(), 0);                                                                                                                                                   

	glDrawBuffer(GL_COLOR_ATTACHMENT0_EXT);

   std::cout << "  <>   [FBO Status] Atlas: " << checkFramebufferStatus()<< endl;

}

void AtlasRenderer::createShader()
{
   pAtlasPosition = new ShaderProgram("src/shader/AtlasPosition.vert", "src/shader/AtlasPosition.frag");

   glUseProgram(0);
}


void AtlasRenderer::createAtlas(int elementIndex, bool addNew)
{
   const ObjectSequence* const elem = SCENE->getSceneElements().at(elementIndex);
   int atlasWidth = elem->getAtlasWidth();
   int atlasHeight = elem->getAtlasHeight();

   vector<GLuint> elemAtlas;
   vector<GLuint> preElemAtlas;

   for(unsigned int inst = 0; inst < elem->getNumInstances(); inst++)
   {
      // position atlas
      elemAtlas.push_back(EmptyTexture::create2D(atlasWidth, atlasHeight, GL_RGBA16F_ARB, GL_RGBA, GL_FLOAT));
   }

   if(addNew)
      mAtlases.push_back(elemAtlas);
   else
      mAtlases.at(elementIndex) = elemAtlas;
}   


void AtlasRenderer::createPixelDisplayList(int elementIndex)
{
   int atlasHeight = SCENE->getSceneElements().at(elementIndex)->getAtlasHeight();
   int atlasWidth  = SCENE->getSceneElements().at(elementIndex)->getAtlasWidth();

   GLfloat* positionBuffer = new GLfloat[atlasWidth * atlasHeight * 3];

   glActiveTexture(GL_TEXTURE0);
   glBindTexture(GL_TEXTURE_2D, mAtlases.at(elementIndex).front());
   glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_FLOAT, positionBuffer);

   cout << "Creating Pixel Display List for scene element " << SCENE->getSceneElements().at(elementIndex)->getName() << endl;
   int counter = 0;

   // create pixel display list
   if(glIsList(mPixelDisplayLists.at(elementIndex)))
      glDeleteLists(mPixelDisplayLists.at(elementIndex), 1);
   mPixelDisplayLists.at(elementIndex) = glGenLists(1);
   glNewList(mPixelDisplayLists.at(elementIndex), GL_COMPILE);

   glPointSize(1.0);  // = 1.0: exactly one voxel per vertex
                      // > 1.0: more than 1 voxel per inserted point (might close holes with too small atlas)

   glBegin(GL_POINTS);
   for(int y = 0; y < atlasHeight; y++)
   {
      for(int x = 0; x < atlasWidth; x++)
      {
         float z = positionBuffer[(3 * y * atlasWidth + 3 * x) + 2];
         if(z < 99.9) // valid
         {
            glVertex2i(x, y);
            counter++;
         }
      }
   }
   glEnd();
   glEndList();

   delete[] positionBuffer;

   cout << "Done. Reduced from " << (atlasWidth* atlasHeight) << " points to " << counter << " points." << endl;

}

void AtlasRenderer::changeAtlasResolutionRelative(int elementIndex, unsigned int deltaX, unsigned int deltaY)
{
   const ObjectSequence* const seq = SCENE->getSceneElements().at(elementIndex);
   changeAtlasResolution(elementIndex, seq->getAtlasWidth()+deltaX, seq->getAtlasHeight()+deltaY);
}

void AtlasRenderer::changeAtlasResolution(int elementIndex, unsigned int newWidth, unsigned int newHeight)
{
   ObjectSequence* const elem = SCENE->getSceneElements().at(elementIndex);

   if(elem->getAtlasWidth() != newWidth || elem->getAtlasHeight() != newHeight )
   {
		elem->setAtlasWidth(newWidth);
		elem->setAtlasHeight(newHeight);

      cout << "New Atlas Resolution for Element " << elem->getName() << " [" << elementIndex << "]: " << newWidth << " x " << newHeight << endl;
   }
	else
   {
      // nothing to do
		return;
   }

   // detach current texture from framebuffer
   V(glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, mFBOAtlas));
   V(glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, 0, 0));

   for(unsigned int i = 0; i < elem->getNumInstances(); i++)
   {
       glDeleteTextures(1, &(mAtlases.at(elementIndex).at(i)));
   }

   mAtlases.at(elementIndex).clear();

	createAtlas(elementIndex, false);

   // Render atlas
   V(glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, mFBOAtlas));
   glDrawBuffer(GL_COLOR_ATTACHMENT0_EXT);

   pAtlasPosition->useProgram();

   glDisable(GL_DEPTH_TEST);
   glDisable(GL_CULL_FACE);

   renderTextureAtlas(elementIndex);

   
   // Read atlas texture to identify valid atlas pixels
   createPixelDisplayList(elementIndex);
}

void AtlasRenderer::initialAtlasRendering()
{
   V(glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, mFBOAtlas));
   glDrawBuffer(GL_COLOR_ATTACHMENT0_EXT);

   pAtlasPosition->useProgram();

   glDisable(GL_DEPTH_TEST);
   glDisable(GL_CULL_FACE);

   glMatrixMode(GL_MODELVIEW);

   for(unsigned int elementIndex = 0; elementIndex < SCENE->getSceneElements().size(); elementIndex++)
   {
      // first render the texture atlas with world positions
      renderTextureAtlas(elementIndex);

      // then create the corresponding pixel display list
      createPixelDisplayList(elementIndex);
   }
}


void AtlasRenderer::updateAllTextureAtlases()
{
   // setup
   glDisable(GL_DEPTH_TEST);
   glDisable(GL_CULL_FACE);
   V(glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, mFBOAtlas));
   glClearColor(0, 0, 100, 0); // 100 = undefined

   pAtlasPosition->useProgram();

   glDrawBuffer(GL_COLOR_ATTACHMENT0_EXT);

   for(unsigned int elementIndex = 0; elementIndex < SCENE->getSceneElements().size(); elementIndex++)
   {
      const bool staticElement = SCENE->getSceneElements().at(elementIndex)->isStatic();
      if(staticElement)
      {
         // nothing needs to be done
         continue; 
      }
      else
      {
         // render positions of the dynamic object 
         renderTextureAtlas(elementIndex);
      }
   }
}


// render texture atlas into texture bound by mFBOAtlas
void AtlasRenderer::renderTextureAtlas(int elementIndex)
{
   const ObjectSequence* const elem = SCENE->getSceneElements().at(elementIndex);
   const vector<GLuint> elemAtlas = mAtlases.at(elementIndex);

   glPushAttrib(GL_VIEWPORT_BIT);
   glViewport(0, 0, elem->getAtlasWidth(), elem->getAtlasHeight()); // same viewport for all instances
   glClearColor(0, 0, 100, 0); // 100 = undefined
   
   // render atlas

   for(unsigned int inst = 0; inst < elem->getNumInstances(); inst++)
   {
      glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, elemAtlas.at(inst), 0);                                                                           
      glClear(GL_COLOR_BUFFER_BIT);

      glPushMatrix();
      glLoadIdentity(); // world space
      SCENE->drawElementWithMode(elementIndex, inst, GLM_TEXTURE_ATLAS);
      glPopMatrix();
   }

   // reset viewport
   glPopAttrib();
}


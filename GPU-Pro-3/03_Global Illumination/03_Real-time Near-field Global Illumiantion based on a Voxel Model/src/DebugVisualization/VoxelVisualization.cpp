///////////////////////////////////////////////////////////////////////
// Real-time Near-Field Global Illumination based on a Voxel Model
// GPU Pro 3
///////////////////////////////////////////////////////////////////////
// Authors : Sinje Thiedemann, Niklas Henrich, 
//           Thorsten Grosch, Stefan Mueller
// Created:  August 2011
///////////////////////////////////////////////////////////////////////

#include "VoxelVisualization.h"

#include "Lighting/IndirectLight.h"
#include "Lighting/SpotMapRenderer.h"
#include "Voxelization/PreVoxelizationData.h"
#include "Voxelization/MipmapRenderer.h"

#include "Scene/Camera.h"
#include "Scene/Scene.h"
#include "Scene/SpotLight.h"

#include "Utils/CoutMethods.h"
#include "Utils/EmptyTexture.h"
#include "Utils/FullScreenQuad.h"
#include "Utils/GLError.h"
#include "Utils/ShaderPool.h"
#include "Utils/ShaderProgram.h"
#include "Utils/TexturePool.h"

#include "GL/glut.h"

GLuint* VoxelVisualization::binarySlicemapData = 0;
GLfloat* VoxelVisualization::volumeData = 0;
const int VoxelVisualization::range = 1000;

// vertex coords array
GLfloat vertices[] = {1,1,1,  0,1,1,  0,0,1,  1,0,1,        // v0-v1-v2-v3
                      1,1,1,  1,0,1,  1,0,0,  1,1,0,        // v0-v3-v4-v5
                      1,1,1,  1,1,0,  0,1,0,  0,1,1,        // v0-v5-v6-v1
                      0,1,1,  0,1,0,  0,0,0,  0,0,1,    // v1-v6-v7-v2
                      0,0,0,  1,0,0,  1,0,1,  0,0,1,    // v7-v4-v3-v2
                      1,0,0,  0,0,0,  0,1,0,  1,1,0};   // v4-v7-v6-v5

// normal array
GLfloat normals[] = {0,0,1,  0,0,1,  0,0,1,  0,0,1,             // v0-v1-v2-v3
                     1,0,0,  1,0,0,  1,0,0, 1,0,0,              // v0-v3-v4-v5
                     0,1,0,  0,1,0,  0,1,0, 0,1,0,              // v0-v5-v6-v1
                     -1,0,0,  -1,0,0, -1,0,0,  -1,0,0,          // v1-v6-v7-v2
                     0,-1,0,  0,-1,0,  0,-1,0,  0,-1,0,         // v7-v4-v3-v2
                     0,0,-1,  0,0,-1,  0,0,-1,  0,0,-1};        // v4-v7-v6-v5

 //color array
//GLfloat colors[] = {1,1,1,  1,1,0,  1,0,0,  1,0,1,              // v0-v1-v2-v3
//                    1,1,1,  1,0,1,  0,0,1,  0,1,1,              // v0-v3-v4-v5
//                    1,1,1,  0,1,1,  0,1,0,  1,1,0,              // v0-v5-v6-v1
//                    1,1,0,  0,1,0,  0,0,0,  1,0,0,              // v1-v6-v7-v2
//                    0,0,0,  0,0,1,  1,0,1,  1,0,0,              // v7-v4-v3-v2
//                    0,0,1,  0,0,0,  0,1,0,  0,1,1};             // v4-v7-v6-v5

// grey
GLfloat colors[] = {1,1,1,  0.9f,0.9f,0.9f,  0.7f,0.7f,0.7f,  0.8f,0.8f,0.8f,              // v0-v1-v2-v3
                    1,1,1,  0.8f,0.8f,0.8f,  0.7f,0.7f,0.7f,  0.8f,0.8f,0.8f,              // v0-v3-v4-v5
                    1,1,1,  0.9f,0.9f,0.9f,  0.7f,0.7f,0.7f,  0.9f,0.9f,0.9f,              // v0-v5-v6-v1
                    0.9f,0.9f,0.9f,  0.7f,0.7f,0.7f,  0.15,0.15,0.15,  0.7f,0.7f,0.7f,              // v1-v6-v7-v2
                    0.15,0.15,0.15,  0.7f,0.7f,0.7f,  0.8f,0.8f,0.8f,  0.7f,0.7f,0.7f,              // v7-v4-v3-v2
                    0.7f,0.7f,0.7f,  0.15,0.15,0.15,  0.7f,0.7f,0.7f,  0.9f,0.9f,0.9f};             // v4-v7-v6-v5

float VoxelVisualization::voxelAlpha = 1.0;

VoxelVisualization::VoxelVisualization(PreVoxelizationData* preVoxData)
: mPreVoxData(preVoxData)
{
   createFBO();
   createShader();

   mSliceData = 0;
   mCurrentSliceSize = 0;
   mFirstSliceRun = true;

   //New quadric object
   mQuadric = gluNewQuadric(); 
}

VoxelVisualization::~VoxelVisualization()
{
   delete[] mSliceData;
}


void VoxelVisualization::createFBO()
{
   // create textures
   mTexPositionRayStart = EmptyTexture::create2D(SCENE->getWindowWidth(), SCENE->getWindowHeight(), GL_RGBA16F_ARB, GL_RGBA, GL_FLOAT);	
	mTexPositionRayEnd   = EmptyTexture::create2D(SCENE->getWindowWidth(), SCENE->getWindowHeight(), GL_RGBA16F_ARB, GL_RGBA, GL_FLOAT);

	// Create Frame Buffer Object
	V(glGenFramebuffersEXT(1, &fboCubePositions));
	V(glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fboCubePositions));

	// bind textures to color attachment points
	glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, mTexPositionRayStart, 0);                                                                           

	// attach render buffer object to FBO for depth test
	glGenRenderbuffersEXT(1, &mDepthRenderBuffer);
	glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, mDepthRenderBuffer);
	glRenderbufferStorageEXT(GL_RENDERBUFFER_EXT, GL_DEPTH_COMPONENT, SCENE->getWindowWidth(), SCENE->getWindowHeight());
	glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, GL_RENDERBUFFER_EXT, mDepthRenderBuffer);

   glDrawBuffer(GL_COLOR_ATTACHMENT0_EXT);

   cout << "  <>   [FBO Status] Ray Start-End Positions: " << checkFramebufferStatus() << endl;

   vector<float> empty;
   for(int i = 0; i < SCENE->getWindowWidth() * SCENE->getWindowHeight(); i++)
   {
      empty.push_back(0);
      empty.push_back(0.4f);
      empty.push_back(0);
      empty.push_back(0);
   }

   mTexRaycastingResult = EmptyTexture::create2D(SCENE->getWindowWidth(), SCENE->getWindowHeight(), GL_RGBA16F_ARB, GL_RGBA, GL_FLOAT, GL_NEAREST, GL_NEAREST, &empty[0]);	

   V(glGenFramebuffersEXT(1, &fboRaycastingResult));
	V(glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fboRaycastingResult));
	glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, mTexRaycastingResult, 0);                                                                           

   glDrawBuffer(GL_COLOR_ATTACHMENT0_EXT);
   cout << "  <>   [FBO Status] VoxelVis RayCasting Result: " << checkFramebufferStatus() << endl;

   mTexBitRay = EmptyTexture::create2D(1, 1, GL_RGBA32UI_EXT, GL_RGBA_INTEGER_EXT, GL_UNSIGNED_INT);
	V(glGenFramebuffersEXT(1, &fboBitRay));
	V(glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fboBitRay));
	glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, mTexBitRay, 0);                                                                           
   glDrawBuffer(GL_COLOR_ATTACHMENT0_EXT);
  

	V(glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0));

   glGenTextures(1, &tboTranslate);
	glGenBuffers(1, &vboTranslate);

}

void VoxelVisualization::createShader()
{
   pCreatePositions = new ShaderProgram("src/shader/RayCastingCreatePositionTextures.vert",
      "src/shader/RayCastingCreatePositionTextures.frag");

   pRayCastingSlicemap  
      = new ShaderProgram("src/shader/RayCasting.vert", "src/shader/RayCastingSlicemap.frag");

   pRayCastingSlicemapMipmap  
      = new ShaderProgram("src/shader/RayCasting.vert", "src/shader/RayCastingSlicemapMipmap.frag");
  
   pInstancedCubes 
      = new ShaderProgram("src/shader/InstancedCubes.vert", "src/shader/InstancedCubes.frag");

   pBitRay
      = new ShaderProgram("src/shader/Quad.vert", "src/shader/BitRay.frag");

   glUseProgram(0);
}


void VoxelVisualization::renderPositionsTextures(const Camera* voxelCamera, bool alwaysWithNearPlane)
{
   Frustum f = voxelCamera->getFrustum();

   // first step: create positions texture of a unit cube
	V(glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fboCubePositions));
   pCreatePositions->useProgram();
   
   glEnable(GL_DEPTH_TEST);

   // unit cube front faces
	glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, mTexPositionRayStart, 0);                                                                           
 	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);	
   glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

   // allow camera being inside the volume's bounding geometry
   const bool renderNearPlane = alwaysWithNearPlane;
   if(renderNearPlane)
   {
      glUniform1i(pCreatePositions->getUniformLocation("writeColor"), 1);

      glDisable(GL_LIGHTING);
      glDisable(GL_COLOR_MATERIAL);
      glColor4f(0, 0, 0, 1);
      FullScreenQuad::drawComplete();

      glUniform1i(pCreatePositions->getUniformLocation("writeColor"), 0);

   }

   // transform volume cube to original voxelization region
   glPushMatrix();
      glMultMatrixf(&voxelCamera->getInverseViewMatrix()[0][0]); 

      if( !renderNearPlane )
      {
         //glColor3f(0,1,0);
         drawBox(f.left, f.bottom, -f.zFar, f.right, f.top, -f.zNear);
      }
      // unit cube back faces
      glEnable(GL_CULL_FACE);
      glCullFace(GL_FRONT);
	   glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, mTexPositionRayEnd, 0);                                                                           
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

      drawBox(f.left, f.bottom, -f.zFar, f.right, f.top, -f.zNear);

   glPopMatrix();

   glCullFace(GL_BACK);
   glDisable(GL_CULL_FACE);
   glDisable(GL_DEPTH_TEST);
}

void VoxelVisualization::rayCastBinaryVoxelTexture(GLuint binaryVoxelTexture,
                                         const Camera* voxelCamera)
{

   renderPositionsTextures(voxelCamera, true);

   // second step: volume ray casting
	V(glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0));

   /* DEBUG OUTPUT OF RAY TEXTURE 
   glActiveTexture(GL_TEXTURE0);
   glBindTexture(GL_TEXTURE_2D, mTexPositionRayStart);

   ShaderPool::getQuad()->useProgram();
   FullScreenQuad::drawComplete();
   */

   pRayCastingSlicemap->useProgram();
   glActiveTexture(GL_TEXTURE0);
   glBindTexture(GL_TEXTURE_2D, mTexPositionRayStart);
	V(glUniform1i(pRayCastingSlicemap->getUniformLocation("texPositionRayStart"), 0)); 

   glActiveTexture(GL_TEXTURE1);
   glBindTexture(GL_TEXTURE_2D, mTexPositionRayEnd);
	V(glUniform1i(pRayCastingSlicemap->getUniformLocation("texPositionRayEnd"), 1)); 

   glActiveTexture(GL_TEXTURE2);
   glBindTexture(GL_TEXTURE_2D, binaryVoxelTexture);
	V(glUniform1i(pRayCastingSlicemap->getUniformLocation("voxelTexture"), 2)); 

	V(glActiveTexture(GL_TEXTURE3));
   V(glBindTexture(GL_TEXTURE_1D, TexturePool::getTexture("bitmaskOR")));
	V(glUniform1i(pRayCastingSlicemap->getUniformLocation("bitmask"), 3)); 

	V(glUniform1i(pRayCastingSlicemap->getUniformLocation("writeEyePos"), 0)); 

   // pass matrices
   glUniformMatrix4fv(pRayCastingSlicemap->getUniformLocation("viewMatrixVoxelCam"), 1, GL_FALSE, &voxelCamera->getViewMatrix()[0][0]);
   glUniformMatrix4fv(pRayCastingSlicemap->getUniformLocation("projMatrixVoxelCam"), 1, GL_FALSE, voxelCamera->getProjectionMatrix());
   glUniformMatrix4fv(pRayCastingSlicemap->getUniformLocation("inverseViewMatrixUserCam"), 1, GL_FALSE, &SCENE->getCamera()->getInverseViewMatrix()[0][0]);


   V(glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0));


   FullScreenQuad::drawComplete();
}

void VoxelVisualization::rayCastMipmappedBinaryVoxelTexture(GLuint mipmappedBinaryVoxelTexture,
                                               unsigned int resolution, int level,
                                               const Camera* voxelCamera)
{
   renderPositionsTextures(voxelCamera, false);

   // second step: volume ray casting
	V(glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0));

   /* DEBUG OUTPUT OF RAY TEXTURE 
   glActiveTexture(GL_TEXTURE0);
   glBindTexture(GL_TEXTURE_2D, mTexPositionRayStart);

   ShaderPool::getQuad()->useProgram();
   FullScreenQuad::drawComplete();
   */

   pRayCastingSlicemapMipmap->useProgram();
   glActiveTexture(GL_TEXTURE0);
   glBindTexture(GL_TEXTURE_2D, mTexPositionRayStart);
	V(glUniform1i(pRayCastingSlicemapMipmap->getUniformLocation("texPositionRayStart"), 0)); 

   glActiveTexture(GL_TEXTURE1);
   glBindTexture(GL_TEXTURE_2D, mTexPositionRayEnd);
	V(glUniform1i(pRayCastingSlicemapMipmap->getUniformLocation("texPositionRayEnd"), 1)); 

   glActiveTexture(GL_TEXTURE2);
   glBindTexture(GL_TEXTURE_2D, mipmappedBinaryVoxelTexture);
	V(glUniform1i(pRayCastingSlicemapMipmap->getUniformLocation("voxelTexture"), 2)); 

	V(glActiveTexture(GL_TEXTURE3));
   V(glBindTexture(GL_TEXTURE_1D, TexturePool::getTexture("bitmaskOR")));
	V(glUniform1i(pRayCastingSlicemapMipmap->getUniformLocation("bitmask"), 3)); 

	V(glUniform1i(pRayCastingSlicemapMipmap->getUniformLocation("level"), level)); 

   // pass matrices
   glUniformMatrix4fv(pRayCastingSlicemapMipmap->getUniformLocation("viewMatrixVoxelCam"), 1, GL_FALSE, &voxelCamera->getViewMatrix()[0][0]);
   glUniformMatrix4fv(pRayCastingSlicemapMipmap->getUniformLocation("projMatrixVoxelCam"), 1, GL_FALSE, voxelCamera->getProjectionMatrix());
   glUniformMatrix4fv(pRayCastingSlicemapMipmap->getUniformLocation("inverseViewMatrixUserCam"), 1, GL_FALSE, &SCENE->getCamera()->getInverseViewMatrix()[0][0]);


   FullScreenQuad::drawComplete();

}

void VoxelVisualization::drawBox(const GLfloat box[6])
{
   drawBox(box[0], box[1], box[2], box[3], box[4], box[5]);
}

void VoxelVisualization::drawBox(float minX, float minY, float minZ,
                                 float maxX, float maxY, float maxZ,
                                 bool withContour)
{

   if(withContour)
   {
      GLfloat color[4];
      glGetFloatv(GL_CURRENT_COLOR, color);
      glColor4f(color[0]*0.5, color[1]*0.5, color[2]*0.5, 1);

      glLineWidth(1.5);
      glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
      glBegin(GL_QUADS);
      glVertex3f(minX,minY,minZ); glVertex3f(minX,maxY,minZ); glVertex3f(maxX,maxY,minZ); glVertex3f(maxX,minY,minZ); // front face
      glVertex3f(minX,minY,maxZ); glVertex3f(maxX,minY,maxZ); glVertex3f(maxX,maxY,maxZ); glVertex3f(minX,maxY,maxZ); // back face
      glVertex3f(minX,minY,minZ); glVertex3f(minX,minY,maxZ); glVertex3f(minX,maxY,maxZ); glVertex3f(minX,maxY,minZ); // left face
      glVertex3f(maxX,minY,minZ); glVertex3f(maxX,maxY,minZ); glVertex3f(maxX,maxY,maxZ); glVertex3f(maxX,minY,maxZ); // right face
      glVertex3f(minX,minY,minZ); glVertex3f(maxX,minY,minZ); glVertex3f(maxX,minY,maxZ); glVertex3f(minX,minY,maxZ); // bottom face  
      glVertex3f(minX,maxY,minZ); glVertex3f(minX,maxY,maxZ); glVertex3f(maxX,maxY,maxZ); glVertex3f(maxX,maxY,minZ); // top face
      glEnd();

      glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
      glColor4fv(color);
   }

   glBegin(GL_QUADS);
   glNormal3f(0,0,-1); glVertex3f(minX,minY,minZ); glVertex3f(minX,maxY,minZ); glVertex3f(maxX,maxY,minZ); glVertex3f(maxX,minY,minZ); // front face
   glNormal3f(0,0,+1); glVertex3f(minX,minY,maxZ); glVertex3f(maxX,minY,maxZ); glVertex3f(maxX,maxY,maxZ); glVertex3f(minX,maxY,maxZ); // back face
   glNormal3f(-1,0,0); glVertex3f(minX,minY,minZ); glVertex3f(minX,minY,maxZ); glVertex3f(minX,maxY,maxZ); glVertex3f(minX,maxY,minZ); // left face
   glNormal3f(+1,0,0); glVertex3f(maxX,minY,minZ); glVertex3f(maxX,maxY,minZ); glVertex3f(maxX,maxY,maxZ); glVertex3f(maxX,minY,maxZ); // right face
   glNormal3f(0,-1,0); glVertex3f(minX,minY,minZ); glVertex3f(maxX,minY,minZ); glVertex3f(maxX,minY,maxZ); glVertex3f(minX,minY,maxZ); // bottom face  
   glNormal3f(0,+1,0); glVertex3f(minX,maxY,minZ); glVertex3f(minX,maxY,maxZ); glVertex3f(maxX,maxY,maxZ); glVertex3f(maxX,maxY,minZ); // top face
   glEnd();


}


void VoxelVisualization::drawVoxelsAsCubesInstanced(GLuint voxelTexture,
                                                    unsigned int resX, unsigned int resY,
                                                    const Camera* const cam)
{ 
   if(mSliceData == 0)
   {
      mSliceData = new GLuint[resX * resY * 4];
      mCurrentSliceSize = resX * resY * 4;
   }
   else if(mCurrentSliceSize < int(resX * resY * 4))
   {
      delete[] mSliceData;
      mSliceData = new GLuint[resX * resY * 4];
      mCurrentSliceSize = resX * resY * 4;
   }

   glm::vec3 eye = cam->getEye();
   Frustum f = cam->getFrustum();

   float voxelWidth  = f.width  / resX;
   float voxelHeight = f.height / resY;
   float voxelLength = f.zRange / 128;



   if(mFirstSliceRun)
   {
      V(glActiveTexture(GL_TEXTURE0));
      V(glBindTexture(GL_TEXTURE_2D, voxelTexture));
      V(glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA_INTEGER_EXT, GL_UNSIGNED_INT, mSliceData));
      glBindTexture(GL_TEXTURE_2D, 0);

      int bitsPerChannel = 32;
      GLuint mask = GLuint(pow(2.0, double(bitsPerChannel-1)));

      GLuint val;
      mNumVoxels = 0;
      vector<float> translate;

      // COLLECT DATA

      for (unsigned int y = 0; y < resY; y++)
      {
         for(unsigned int x = 0; x < resX; x++)
         {
            for (int channel = 0; channel < 4; channel++)
            {
               int pos = 4 * resX * y + 4 * x + channel;
               val = mSliceData[pos];

               mask = GLuint(pow(2.0, double(bitsPerChannel-1)));
               for (int b = 0; b < bitsPerChannel; b++)
               {
                  int z = channel * bitsPerChannel + b;
                  if (val & mask)
                  {
                     mNumVoxels++;
                     translate.push_back(f.left + x * voxelWidth);
                     translate.push_back(f.bottom + y * voxelHeight);
                     translate.push_back(-f.zNear - (z + 1) * voxelLength);
                     translate.push_back(0);
                  }
                  mask = mask >> 1;
               }

            }//end for channel
         }// end for x 

      }// end for y

      unsigned int data_size = mNumVoxels*4*sizeof(float);
      V(glBindBuffer(GL_ARRAY_BUFFER, vboTranslate));
      V(glBufferData(GL_ARRAY_BUFFER, data_size, &translate[0], GL_STATIC_DRAW));
      V(glBindBuffer(GL_ARRAY_BUFFER, 0));
   }
   pInstancedCubes->useProgram();

   V(glUniform1i(pInstancedCubes->getUniformLocation("tboTranslate"), 0));
   glUniform3f(pInstancedCubes->getUniformLocation("scale"), voxelWidth, voxelHeight, voxelLength);


	glActiveTexture(GL_TEXTURE0);
	V(glBindTexture(GL_TEXTURE_BUFFER_EXT, tboTranslate));
	V(glTexBufferEXT(GL_TEXTURE_BUFFER_EXT, GL_RGBA32F_ARB, vboTranslate));


   // RENDER 
   glDisable(GL_LIGHTING);
   V(glEnable(GL_COLOR_MATERIAL));
   //V(glEnableClientState(GL_NORMAL_ARRAY));
   V(glEnableClientState(GL_COLOR_ARRAY));
   V(glEnableClientState(GL_VERTEX_ARRAY));
   //V(glNormalPointer(GL_FLOAT, 0, normals));
   V(glColorPointer(3, GL_FLOAT, 0, colors));
   V(glVertexPointer(3, GL_FLOAT, 0, vertices));


   glEnable(GL_CULL_FACE);
   glLineWidth(2.0);
   glPushMatrix();
   glMultMatrixf(&cam->getInverseViewMatrix()[0][0]);
   V(glDrawArraysInstancedARB(GL_QUADS, 0, 24, mNumVoxels));
   glDisableClientState(GL_COLOR_ARRAY);
   glColor4f(0, 0, 0, 1);
   V(glDrawArraysInstancedARB(GL_LINES, 0, 24, mNumVoxels));
   glPopMatrix();
   glDisable(GL_CULL_FACE);

   glDisableClientState(GL_VERTEX_ARRAY);  // disable vertex arrays
   //glDisableClientState(GL_NORMAL_ARRAY);


   cout << "Drawing " << mNumVoxels << " voxels. ";
   double density = mNumVoxels / (resX * resY * 128.0);
   cout << "Data density: " << density * 100 << " % " << "\r"<< endl;

   //delete[] mSliceData;

}

glm::uvec4 VoxelVisualization::getBitRay(float z1, float z2)
{
   pBitRay->useProgram();
   glUniform1f(pBitRay->getUniformLocation("bitmaskXORRays"), 0);
   glUniform1f(pBitRay->getUniformLocation("z1"), z1);
   glUniform1f(pBitRay->getUniformLocation("z2"), z2);
   glUniform1i(pBitRay->getUniformLocation("getXORRay"), 1);

   glActiveTexture(GL_TEXTURE0);
   glBindTexture(GL_TEXTURE_2D, TexturePool::getTexture("bitmaskXORRays"));

   glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fboBitRay);
   glPushAttrib(GL_VIEWPORT_BIT);
   glViewport(0, 0, 1, 1);

   FullScreenQuad::drawComplete();

   glPopAttrib(); // viewport

   glBindTexture(GL_TEXTURE_2D, mTexBitRay);
   GLuint bitRay[4];
   glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA_INTEGER_EXT, GL_UNSIGNED_INT, bitRay);

   glUseProgram(0);
   glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);

   return glm::uvec4(bitRay[0], bitRay[1], bitRay[2], bitRay[3]);  
}

// "Uniforms"
glm::vec3 origin;
glm::vec3 dir;
glm::vec3 posTNear;
glm::vec2 voxelCameraFrustumOffset;
glm::vec2 frustumWH;
glm::mat4 viewProjToUnitMatrixVoxelCam;
int _maxMipMapLevel;
glm::uvec4 lastBitRay;
glm::uvec4 lastIntersectionBits;
glm::ivec2 lastPixelCoord;
glm::vec2 last_box_min;
glm::vec2 last_box_max;

//http://blog.piloni.net/?p=114
/* Bounding box intersection routine [Slabs]
*  uses globally defined ray origin and direction
*  @param vec3 box_min  Box’s minimum extent coordinates
*  @param vec3 box_max  Box’s maximum extent coordinates
*  @return true if an intersection was found, false otherwise
*   minmax = most far intersection with the box
*   maxmin = nearest intersection with the box
*/
bool IntersectBox(glm::vec3 box_min, glm::vec3 box_max, float& minmax)
{
   glm::vec3 tmin = (box_min - origin) / dir; 
	glm::vec3 tmax = (box_max - origin) / dir; 

   glm::vec3 real_min = glm::min(tmin,tmax);
	glm::vec3 real_max = glm::max(tmin,tmax);

   minmax = glm::min(1.0f, glm::min( glm::min(real_max.x, real_max.y), real_max.z));
	float maxmin = max( max(real_min.x, real_min.y), real_min.z); // tNear not needed

	return (minmax >= maxmin);
}
float IntersectBoxOnlyTFar(glm::vec3 box_min, glm::vec3 box_max)
{
   glm::vec3 tmin = (box_min - origin) / dir; 
	glm::vec3 tmax = (box_max - origin) / dir; 

   glm::vec3 real_min = glm::min(tmin,tmax);
	glm::vec3 real_max = glm::max(tmin,tmax);

	//minmax = glm::min( glm::min(real_max.x, real_max.y), real_max.z);
   return glm::min(1.0f, glm::min( glm::min(real_max.x, real_max.y), real_max.z));
	//maxmin = max( max(real_min.x, real_min.y), real_min.z); // tNear not needed

	//return (minmax >= maxmin);
}


glm::vec3 worldToUnit(glm::vec3 p)
{
   return glm::vec3(viewProjToUnitMatrixVoxelCam * glm::vec4(p, 1.0));
}

bool VoxelVisualization::intersectBits(glm::uvec4 bitRay, glm::ivec2 texel, int level)
{
   lastPixelCoord = texel;
   int thisLevelRes = (1 << (_maxMipMapLevel - level));

   glm::uvec4 slicemapVal;
   for (int channel = 0; channel < 4; channel++)
   {
      int pos = 4 * thisLevelRes * texel.y + 4 * texel.x + channel;
      slicemapVal[channel] = mSliceData[pos];
   }
   lastIntersectionBits = bitRay & slicemapVal;
   return !(glm::all(glm::equal((lastIntersectionBits), glm::uvec4(0))));
}



bool VoxelVisualization::IntersectHierarchy(int level, float& tFar)
{
	// Calculate pixel coordinates ([0,width]x[0,height]) of the current 
	// position along the ray
   float res = float(1 << (_maxMipMapLevel - level));
   glm::ivec2 pixelCoord = glm::ivec2(glm::vec2(posTNear) * res);

   glm::vec2 voxelWH = glm::vec2(1.0) / res;

	// Compute bounding box (AABB) belonging to this pixel position
   // (Slabs for AABB/Ray Intersection)
   glm::vec2 box_min = glm::vec2(pixelCoord) * voxelWH;
   glm::vec2 box_max = box_min + voxelWH;
   last_box_min = box_min * frustumWH + voxelCameraFrustumOffset;
   last_box_max = box_max * frustumWH + voxelCameraFrustumOffset;

	// Compute intersection with the bounding box
	// It is assumed that an intersecion occurs
	// It is assumed that the position of posTNear always remains the same
	tFar = IntersectBoxOnlyTFar(
      glm::vec3(box_min.x, box_min.y, 0.0), 
      glm::vec3(box_max.x, box_max.y, 1.0));

	// Now test if some of the bits intersect
	float z2 = tFar*dir.z + origin.z ;

	// Fetch bit-mask for ray and intersect with current pixel
   lastBitRay = getBitRay(posTNear.z, z2);

	return intersectBits( lastBitRay , pixelCoord, level);	
}

double VoxelVisualization::log2(double d) {return log(d)/log(2.0) ;}	

void VoxelVisualization::drawVoxelsAsCubesInstancedMipmapTestVis(
   GLuint voxelTexture, unsigned int level0_resX, unsigned int level0_resY,
   int maxMipMapLevel,
   const Camera* const voxelCamera, glm::ivec2 userClickPos,
   const IndirectLight* const indirectLight)
{ 
   glUseProgram(0);

   //
   // Initialize "uniforms"
   //
   Frustum f = voxelCamera->getFrustum();

   voxelCameraFrustumOffset = glm::vec2(f.left, f.bottom);
   frustumWH = glm::vec2(f.width, f.height);
   _maxMipMapLevel = maxMipMapLevel;
   lastBitRay = glm::uvec4(0);
   lastPixelCoord = glm::ivec2(0);

   viewProjToUnitMatrixVoxelCam = voxelCamera->getViewProjectionToUnitMatrix();

   // User defined start and endpoint.
   // Transform coordinates to unit cube [0, 1]³
   glm::vec3 origin_unit    
      = glm::vec3(viewProjToUnitMatrixVoxelCam
      * glm::vec4(SETTINGS->getStartPoint()[0], SETTINGS->getStartPoint()[1], SETTINGS->getStartPoint()[2], 1.0f))
      * 0.5f + glm::vec3(0.5f);
   glm::vec3 end_unit    
      = glm::vec3(viewProjToUnitMatrixVoxelCam
      * glm::vec4(SETTINGS->getEndPoint()[0], SETTINGS->getEndPoint()[1], SETTINGS->getEndPoint()[2], 1.0f))
      * 0.5f + glm::vec3(0.5f);

   glm::vec3 dir_unit    = end_unit - origin_unit;

   float vX = f.width  / level0_resX;
   float vY = f.height / level0_resY;
   float vZ = f.zRange / 128;
   float voxelDiagonal = sqrt(vX * vX + vY * vY + vZ * vZ);

   origin = origin_unit;
   dir = dir_unit;

   // Adjust direction a bit to prevent division by zero
   dir.x = (abs(dir.x) < 0.0000001) ? 0.0000001 : dir.x;
   dir.y = (abs(dir.y) < 0.0000001) ? 0.0000001 : dir.y;
   dir.z = (abs(dir.z) < 0.0000001) ? 0.0000001 : dir.z;

   // Compute offset to advance current position on ray 
   // such that in the next traversal step the neighboring
   // voxel stack is tested
   float offset = 0.25f / (1 << maxMipMapLevel) / glm::length(dir);
   //std::cout << " offset " << offset << " with length(dir) " << glm::length(dir) << " and voxelsize: " << (1.0f / (1 << maxMipMapLevel)) << std::endl;

   posTNear = origin;		

   const int MAXSTEPS = 128;
   
   // Choose starting level
   int level = min(maxMipMapLevel, Settings::Instance()->getMipmapLevel()); // maxMipMapLevel / 2;  

   bool intersectionFound = false;			

   int i;
   int intersectionFoundCycle = MAXSTEPS-1;
   int thisLevelResX;
   int thisLevelResY;
   glm::uvec4 bitRays[MAXSTEPS];
   glm::uvec4 intersectionBits[MAXSTEPS];
   glm::ivec2 bitRaysPixelCoords[MAXSTEPS];
   glm::vec2 last_box_mins[MAXSTEPS];
   glm::vec2 last_box_maxs[MAXSTEPS];
   glm::vec3 posTNears[MAXSTEPS];
   glm::vec3 posTNearsWithoutOffset[MAXSTEPS];
   bool intersectedHierarchy[MAXSTEPS];
   bool aabbLeftRightSlabHit[MAXSTEPS];
   int testPerformedAtLevels[MAXSTEPS];

   {
      float tNear = 0.0;

      // restrict tFar to voxelization region
      float tFar  = 1.0;	
      if(!IntersectBox(glm::vec3(0.0), glm::vec3(1.0), tFar))
      {
         // Ray did not hit volume, set tFar to 1.0 (full ray)
         tFar = 1.0;
      }
      else
      {
         // Ray hit volume, tFar has been set by IntersectBox
         // cout << tFar<< endl;
      }


      for(i = 0; (i < MAXSTEPS) 
         && (tNear < tFar)
         && (!intersectionFound); i++)			
      {
         // cout << "i: " << i << endl;
         //last_box_mins[i] = last_box_min;
         //last_box_maxs[i] = last_box_max;

         // current level resolution
         thisLevelResX = level0_resX / pow(2.0, level);
         thisLevelResY = level0_resY / pow(2.0, level);

         if(mSliceData == 0)
         {
            mSliceData = new GLuint[thisLevelResX * thisLevelResY * 4 ];
            mCurrentSliceSize = thisLevelResX * thisLevelResY * 4 ;
         }
         else if(mCurrentSliceSize < int(thisLevelResX * thisLevelResY * 4 ))
         {
            delete[] mSliceData;
            mSliceData = new GLuint[thisLevelResX * thisLevelResY * 4 ];
            mCurrentSliceSize = thisLevelResX * thisLevelResY * 4 ;
         }
         V(glActiveTexture(GL_TEXTURE0));
         V(glBindTexture(GL_TEXTURE_2D, voxelTexture));
         V(glGetTexImage(GL_TEXTURE_2D, level, GL_RGBA_INTEGER_EXT, GL_UNSIGNED_INT, mSliceData));
         glBindTexture(GL_TEXTURE_2D, 0);


         float newTFar = 1.0f;						

         if(IntersectHierarchy(level, newTFar))
         {	
            // If we are at mipmap level 0 and an intersection occured,
            // we have found an intersection of the ray with the volume
            intersectionFound = (level == 0);
            if(intersectionFound) intersectionFoundCycle = i;
            testPerformedAtLevels[i] = level;
            bitRays[i] = lastBitRay;
            intersectionBits[i] = lastIntersectionBits;
            bitRaysPixelCoords[i] = lastPixelCoord;
            intersectedHierarchy[i] = true;
            posTNears[i] = posTNear;
            posTNearsWithoutOffset[i] = origin + newTFar * dir;
            glm::vec3 posTNearsWithoutOffset_world;

            posTNearsWithoutOffset_world.x = f.left   + f.width * posTNearsWithoutOffset[i].x;
            //posTNearsWithoutOffset_world.y = f.bottom + f.height* posTNearsWithoutOffset[i].y;
            //posTNearsWithoutOffset_world.z = -f.zNear - f.zRange* posTNearsWithoutOffset[i].z;
            aabbLeftRightSlabHit[i] = ((abs(posTNearsWithoutOffset_world.x-last_box_min.x) < 0.00001) // hit left
               || (abs(posTNearsWithoutOffset_world.x-last_box_max.x)<0.00001) ); // hit right

            // Otherwise we have to move down one level and
            // start testing from there

            // previous cycle: do not modify level
            {
               level --;		
               //cout << "level--" << endl;
            }
            //cout << "IntersectHierarchy: true" << endl;
         }
         else
         {
            //cout << "IntersectHierarchy: false" << endl;
            // If no intersection occurs, we have to advance the
            // position on the ray to test the next element of the hierachy
            tNear = newTFar + offset;
            posTNear = origin + tNear * dir;

            testPerformedAtLevels[i] = level;
            bitRays[i] = lastBitRay;
            bitRaysPixelCoords[i] = lastPixelCoord;
            intersectedHierarchy[i] = false;
            posTNears[i] = posTNear;
            posTNearsWithoutOffset[i] = origin + newTFar * dir;
            glm::vec3 posTNearsWithoutOffset_world;
            posTNearsWithoutOffset_world.x = f.left   + f.width * posTNearsWithoutOffset[i].x;
            //posTNearsWithoutOffset_world.y = f.bottom + f.height* posTNearsWithoutOffset[i].y;
            //posTNearsWithoutOffset_world.z = -f.zNear - f.zRange* posTNearsWithoutOffset[i].z;
            aabbLeftRightSlabHit[i] = ((abs(posTNearsWithoutOffset_world.x-last_box_min.x) < 0.00001) // hit left
               || (abs(posTNearsWithoutOffset_world.x-last_box_max.x)<0.00001) ); // hit right

            // Move one level up
            {
               level ++; 
               //cout << "level++" << endl;
            }
         }

         // HERE: VISUALIZATION OF CURRENT STATE
         if(i == Settings::Instance()->getMipmapTestCycleIndex())
         {
            //cout << "Test at level: " << testPerformedAtLevels[i] << endl;
            //cout << "Current Level: " << level << endl;
            break;
         }

      }	// for

   }
   //if(intersectionFound) 
   //{
   //   cout << endl << endl << "------------------------------" << endl;
   //   cout << "      FOUND INTERSECTION   " << endl;
   //   cout << "------------------------------" << endl;
   //}

   //for(int i = 0; i < Settings::Instance()->getMipmapTestCycleIndex(); i++)
   //{
   //   cout << "testPerformedAtLevels["<<i<<"] " << testPerformedAtLevels[i] << endl;;
   //}


   // HERE: VISUALIZATION OF CURRENT STATE

   // ALWAYS DRAW VOXEL CUBES AT LEVEL 0
   thisLevelResX = level0_resX;
   thisLevelResY = level0_resY;
   float voxelWidth  = f.width  / thisLevelResX;
   float voxelHeight = f.height / thisLevelResY;
   float voxelLength = f.zRange / 128;

   if(mCurrentSliceSize < int(thisLevelResX * thisLevelResY * 4 ))
   {
      delete[] mSliceData;
      mSliceData = new GLuint[thisLevelResX * thisLevelResY * 4 ];
      mCurrentSliceSize = thisLevelResX * thisLevelResY * 4;
   }
   V(glActiveTexture(GL_TEXTURE0));
   V(glBindTexture(GL_TEXTURE_2D, voxelTexture));
   V(glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA_INTEGER_EXT, GL_UNSIGNED_INT, mSliceData));
   glBindTexture(GL_TEXTURE_2D, 0);


   // DRAW VOXEL CUBES
   if(Settings::Instance()->mustDisplayVoxelCubesDuringMipmapTest())
   {
      int bitsPerChannel = 32;
      GLuint mask = GLuint(pow(2.0, double(bitsPerChannel-1)));

      GLuint val;
      mNumVoxels = 0;
      vector<float> translate;

      // COLLECT DATA

      for (int y = 0; y < thisLevelResY; y++)
      {
         for(int x = 0; x < thisLevelResX; x++)
         {
            for (int channel = 0; channel < 4; channel++)
            {
               int pos = 4 * thisLevelResX * y + 4 * x + channel;
               val = mSliceData[pos];

               mask = GLuint(pow(2.0, double(bitsPerChannel-1)));
               for (int b = 0; b < bitsPerChannel; b++)
               {
                  int z = channel * bitsPerChannel + b;
                  if (val & mask)
                  {
                     mNumVoxels++;
                     translate.push_back(f.left + x * voxelWidth);
                     translate.push_back(f.bottom + y * voxelHeight);
                     translate.push_back(-f.zNear - (z + 1) * voxelLength);
                     translate.push_back(0);
                  }
                  mask = mask >> 1;
               }

            }//end for channel
         }// end for x 

      }// end for y

      unsigned int data_size = mNumVoxels*4*sizeof(float);
      V(glBindBuffer(GL_ARRAY_BUFFER, vboTranslate));
      V(glBufferData(GL_ARRAY_BUFFER, data_size, &translate[0], GL_STATIC_DRAW));
      V(glBindBuffer(GL_ARRAY_BUFFER, 0));


      pInstancedCubes->useProgram();

      V(glUniform1i(pInstancedCubes->getUniformLocation("tboTranslate"), 0));
      glUniform3f(pInstancedCubes->getUniformLocation("scale"), voxelWidth, voxelHeight, voxelLength);


      glActiveTexture(GL_TEXTURE0);
      V(glBindTexture(GL_TEXTURE_BUFFER_EXT, tboTranslate));
      V(glTexBufferEXT(GL_TEXTURE_BUFFER_EXT, GL_RGBA32F_ARB, vboTranslate));


      // RENDER 
      glDisable(GL_LIGHTING);
      V(glEnable(GL_COLOR_MATERIAL));
      //V(glEnableClientState(GL_NORMAL_ARRAY));
      V(glEnableClientState(GL_VERTEX_ARRAY));
      //V(glNormalPointer(GL_FLOAT, 0, normals));
      V(glVertexPointer(3, GL_FLOAT, 0, vertices));


      if(voxelAlpha < 1.0)
      {
         glDisable(GL_CULL_FACE);
         glDisable(GL_DEPTH_TEST);
         glEnable(GL_BLEND);
         glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
         glColor4f(0.3f, 0.3f, 0.3f, voxelAlpha);

      }
      else
      {
         glEnable(GL_CULL_FACE);

         V(glEnableClientState(GL_COLOR_ARRAY));
         V(glColorPointer(3, GL_FLOAT, 0, colors));
      }

      glPushMatrix();
      glMultMatrixf(&voxelCamera->getInverseViewMatrix()[0][0]);
      V(glDrawArraysInstancedARB(GL_QUADS, 0, 24, mNumVoxels));
      if(voxelAlpha >= 1.0)
      {
         glDisableClientState(GL_COLOR_ARRAY);
         glDisable(GL_CULL_FACE);
      }
      //glEnable(GL_BLEND);
      //glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
      glLineWidth(3.0);
      //glColor4f(0.6f, 0.6f, 0.6f, 0.8f);
      //V(glDrawArraysInstancedARB(GL_LINES, 0, 24, mNumVoxels));
       glColor4f(0.5, 0.5, 0.5, voxelAlpha);
      V(glDrawArraysInstancedARB(GL_LINES, 0, 24, mNumVoxels));
      glPopMatrix();

      if(voxelAlpha < 1.0)
      {
         glEnable(GL_DEPTH_TEST);
         glDisable(GL_BLEND);
      }


      glDisableClientState(GL_VERTEX_ARRAY);  // disable vertex arrays
      //glDisableClientState(GL_NORMAL_ARRAY);
      //glDisable(GL_BLEND);


      cout << "Drawing " << mNumVoxels << " voxels. ";
      double density = mNumVoxels / (thisLevelResX * thisLevelResY * 128.0);
      cout << "Data density: " << density * 100 << " % " << "\r"<< endl;

      // END VOXEL CUBES
   }

   glEnable(GL_LIGHTING);
   glUseProgram(0);
   glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);


   glEnable(GL_BLEND);
   glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
   
   glPushMatrix();
   glMultMatrixf(&voxelCamera->getInverseViewMatrix()[0][0]);

   // HERE: voxel camera space

   // whole ray

   // start = green
   glColor4f(0.1f, 1.0f, 0.1f, 1.0f); 
   drawSphereAt(0.005f, f.left + f.width*origin.x, f.bottom + f.height * origin.y, -f.zNear - f.zRange*origin.z);
   // end = red
   glColor4f(1.0f, 0.1f, 0.1f, 1.0f); 
   drawSphereAt(0.005f, f.left + f.width*end_unit.x,    f.bottom + f.height * end_unit.y,    -f.zNear - f.zRange*end_unit.z);

   glColor4f(1.0f, 0.5f, 0.1f, 1.0f);
   glm::vec3 origin_cam, end_cam;
   origin_cam.x = f.left + f.width * origin.x;
   origin_cam.y = f.bottom + f.height * origin.y;
   origin_cam.z = -f.zNear - f.zRange * origin.z;
   
   end_cam.x = f.left + f.width * end_unit.x;
   end_cam.y = f.bottom + f.height * end_unit.y;
   end_cam.z = -f.zNear - f.zRange * end_unit.z;
   glm::vec3 dir_world = end_cam - origin_cam;

   drawCylinderFromTo(0.002f, origin_cam.x, origin_cam.y, origin_cam.z,
      end_cam.x,  end_cam.y, end_cam.z );

   //drawCone(0.01f, end_world - 0.1f * dir_world, end_world);

   // AABB of voxel stack
   {
      glColor4f(0.2f, 0.3f, 0.8f, 1.0f);
      glm::vec3 box_min;
      box_min.x = last_box_min.x;
      box_min.y = last_box_min.y;
      box_min.z = -f.zFar;
      glm::vec3 box_max;
      box_max.x = last_box_max.x;
      box_max.y = last_box_max.y;
      box_max.z = -f.zNear;

      drawCylinderBox(0.0015f, box_min, box_max);
   }

   // current ray part
   {
      glColor4f(0.1f, 0.1f, 0.6f, 1.0f);
      glm::vec3 sectP = posTNearsWithoutOffset[Settings::Instance()->getMipmapTestCycleIndex()];
      sectP.x = f.left + f.width * sectP.x;
      sectP.y = f.bottom + f.height * sectP.y;
      sectP.z = -f.zNear - f.zRange * sectP.z;
      //sectP = glm::vec3(voxelCamera->getInverseViewProjectionToUnitMatrix() * glm::vec4(sectP, 1.0));

      glm::vec3 rotAxis;
      if(aabbLeftRightSlabHit[i])
         rotAxis = glm::vec3(0, 1, 0);
      else
         rotAxis = glm::vec3(1, 0, 0);
      // assume: hit bottom / top
      drawTorusAt(0.0015, 0.0045, sectP, rotAxis);


      // the intersection point reconstructed from set voxels
      if(intersectionFound)
      {
         glm::vec3 voxelizingDirection = voxelCamera->getViewDirection();
         // Compute the position of the highest or lowest set bit in the resulting bitmask.
         // Compute highest bit if ray not reverse to voxelization direction
         bool reversed = glm::dot(glm::normalize(dir_world), voxelizingDirection) < 0.0f;

         int bitPosition = 0; 
         int x = 0;
         if(!reversed)
         {
            // get the position of the highest bit set
            int v;
            for(v = 0; v < 4 && x==0; v++) // r g b a
            {
               x = int(intersectionBits[intersectionFoundCycle][v]);
               if(x != 0)
               {
                  int pos32 = int(log2(float(x)));
                  bitPosition = (3-v)*32 + pos32;

               }
            }

         }
         else
         {
            // get the position of the lowest bit set
            int v;
            for(v = 3; v >= 0 && x == 0; v--) // r g b a
            {
               x = int(intersectionBits[intersectionFoundCycle][v]);
               if(x != 0)
               {
                  int pos32 = int(log2(float(x & ~(x-1)))+0.1);
                  bitPosition = (3-v)*32 + pos32;
               }
            }


         }
         glm::vec3 estimatedHitPosition = posTNears[intersectionFoundCycle];
         estimatedHitPosition.z = float(127 - bitPosition) * 0.0078125 +0.00390625; // 1.0/128.0

         glColor4f(1, 0, 1, 1);
         drawSphereAt(0.01f, f.left + f.width*estimatedHitPosition.x, f.bottom + f.height * estimatedHitPosition.y, -f.zNear - f.zRange*estimatedHitPosition.z);

      }
      glColor4f(1.0f, 1.0f, 0.2f, 1.0f);
      drawCylinderFromTo(0.003f, f.left + f.width*posTNear.x, f.bottom + f.height * posTNear.y, -f.zNear - f.zRange*posTNear.z,
         f.left + f.width*end_unit.x, f.bottom + f.height * end_unit.y,  -f.zNear - f.zRange*end_unit.z);

   }
   for(int i = 0; i <= Settings::Instance()->getMipmapTestCycleIndex(); i++)
   {
      if(i == Settings::Instance()->getMipmapTestCycleIndex())
         glColor4f(0.1f, 0.4f, 0.9f, 1.0f);
      else
         glColor4f(0.3f, 0.3f, 0.3f, 1.0f);
      //glColor4f(i == Settings::Instance()->getMipmapTestCycleIndex() ? 1.0f : 0.5, 0.4f, 0.1f, 1.0f);
      drawSphereAt(i == Settings::Instance()->getMipmapTestCycleIndex() ? 0.0045f : 0.0035f,
         f.left + f.width*posTNears[i].x, f.bottom + f.height * posTNears[i].y, -f.zNear - f.zRange*posTNears[i].z);
   }

   for(int i = 0; i <= min(intersectionFoundCycle, Settings::Instance()->getMipmapTestCycleIndex()); i++)
   {

      glColor4f(0.5f, 0.8f, 1.0f, 1.0f);
      // current bit ray
      if(!(glm::all(glm::equal(glm::uvec4(0), bitRays[i]))))
      {
         int atLevel = testPerformedAtLevels[i];
         //cout << "at level: " << atLevel << endl;
         int atResolution = level0_resX / pow(2.0, atLevel);

         float atVoxelWidth  = f.width   / atResolution;
         float atVoxelHeight = f.height  / atResolution;
         float atVoxelLength = f.zRange  / 128;
         glm::ivec2 atPixelCoord = bitRaysPixelCoords[i];
         //cout << "drawing bit ray" << endl;
         //cout << "lastPixelCoord " << lastPixelCoord.x << " " << lastPixelCoord.y << endl;

         // set color
         if(i == Settings::Instance()->getMipmapTestCycleIndex() 
            || i == intersectionFoundCycle)
         {
            // transparent boxes
            if(intersectedHierarchy[i])
               glColor4f(0.8f, 0.2f, 0.2f, 0.4f);
            else
               glColor4f(0.2f, 0.8f, 0.2f, 0.4f);
         }
         else
         {
            // solid
            if(intersectedHierarchy[i])
               glColor4f(0.7f, 0.05f, 0.05f, 1.0f);
            else
               glColor4f(0.05f, 0.7f, 0.05f, 1.0f);
         }

         int lowestZ = 128;
         int highestZ = 0;
         int bitsPerChannel = 32;
         for (int channel = 0; channel < 4; channel++)
         {
            GLuint mask = GLuint(pow(2.0, double(bitsPerChannel-1)));
            GLuint val  = bitRays[i][channel];
            for (int b = 0; b < bitsPerChannel; b++)
            {
               int z = channel * bitsPerChannel + b;
               if ((val & mask))
               {
                  if(z < lowestZ) lowestZ = z;
                  if(z > highestZ) highestZ = z;
                  // draw this voxel
                  if((i == Settings::Instance()->getMipmapTestCycleIndex()
                     || i == intersectionFoundCycle))
                  {
                     glDisable(GL_LIGHTING);
                     
                     glm::vec3 hitOffsets; 
                     if(intersectionBits[i][channel] & mask)
                     {
                        glColor4f(0.8f, 0.2f, 0.2f, 0.4f);
                        hitOffsets = glm::vec3(atVoxelWidth, atVoxelHeight, atVoxelLength) * 0.005f;
                     }
                     else
                     {
                        glColor4f(0.2f, 0.8f, 0.2f, 0.4f);
                     }
                     drawBox(-hitOffsets.x + f.left + atPixelCoord.x * atVoxelWidth,
                        -hitOffsets.y + f.bottom + atPixelCoord.y * atVoxelHeight,
                        -hitOffsets.z + -f.zNear - (z + 1) * atVoxelLength,
                        hitOffsets.x + f.left + (atPixelCoord.x + 1) * atVoxelWidth,
                        hitOffsets.y + f.bottom + (atPixelCoord.y + 1) * atVoxelHeight,
                        hitOffsets.z + -f.zNear - z * atVoxelLength, true);
                  }
               }
               mask = mask >> 1;
            }

         }//end for channel

         if(i < glm::min(intersectionFoundCycle, Settings::Instance()->getMipmapTestCycleIndex()))
         {
            glEnable(GL_LIGHTING);
            glm::vec3 hitBoxMin, hitBoxMax;
            hitBoxMin.x = f.left + atPixelCoord.x * atVoxelWidth;
            hitBoxMin.y = f.bottom + atPixelCoord.y * atVoxelHeight;
            hitBoxMin.z = -f.zNear - (highestZ+1) * atVoxelLength;
            hitBoxMax.x = f.left + (atPixelCoord.x + 1) * atVoxelWidth;
            hitBoxMax.y = f.bottom + (atPixelCoord.y + 1) * atVoxelHeight;
            hitBoxMax.z = -f.zNear - lowestZ * atVoxelLength;

            drawCylinderBox(0.003, hitBoxMin, hitBoxMax);

         }

      }
   }

   glPopMatrix();
   glDisable(GL_BLEND);

}


void VoxelVisualization::drawTorusAt(float innerRadius, float outerRadius, glm::vec3 pos, glm::vec3 rotAxis)
{
   glPushMatrix();
   glTranslatef(pos.x, pos.y, pos.z);
   glRotatef(90, rotAxis.x, rotAxis.y, rotAxis.z);
   glutSolidTorus(innerRadius, outerRadius, 20, 20);
   glPopMatrix();
}

void VoxelVisualization::drawSphereAt(float radius, float posX, float posY, float posZ)
{
   glPushMatrix();
   glTranslatef(posX, posY, posZ);
   gluSphere(mQuadric, radius, 20, 20);
   glPopMatrix();
}
void VoxelVisualization::drawSphereAt(float radius, glm::vec3 pos)
{
   drawSphereAt(radius, pos.x, pos.y, pos.z);
}
void VoxelVisualization::drawCone(float baseRadius, glm::vec3 from, glm::vec3 to)
{
   float length = glm::distance(from, to);

   glm::mat4 m = glm::lookAt(from, to, glm::normalize(glm::vec3(1,1,1)));

   glPushMatrix();
   glMultMatrixf(&(glm::inverse(m))[0][0]);
   glScalef(1, 1, -1);
   gluCylinder(mQuadric, baseRadius, 0.003f, length, 20, 10);
   glPopMatrix();
}

void VoxelVisualization::drawCylinderFromTo(float radius,
                                            float fromX, float fromY, float fromZ, 
                                            float toX, float toY, float toZ)
{
   drawCylinderFromTo(mQuadric, radius, glm::vec3(fromX, fromY, fromZ), glm::vec3(toX, toY, toZ));
}
void VoxelVisualization::drawCylinderFromTo(float radius, glm::vec3 from, glm::vec3 to)
{
   drawCylinderFromTo(radius, from.x, from.y, from.z, to.x, to.y, to.z);
}

void VoxelVisualization::drawCylinderFromTo(GLUquadric* quad, float radius, glm::vec3 from, glm::vec3 to)
{
   float length = glm::distance(from, to);

   glm::mat4 m = glm::lookAt(from, to, glm::normalize(glm::vec3(1,1,1)));

   glPushMatrix();
   glMultMatrixf(&(glm::inverse(m))[0][0]);
   glScalef(1, 1, -1);
   gluCylinder(quad, radius, radius, length, 20, 2);
   glPopMatrix();

}


void VoxelVisualization::drawCylinderBox(float cylinderRadius, glm::vec3 box_min, glm::vec3 box_max)
{
   // draw box edges

   // front to back
   drawCylinderFromTo(cylinderRadius, box_min.x, box_min.y, box_min.z,
                                      box_min.x, box_min.y, box_max.z);
   drawCylinderFromTo(cylinderRadius, box_min.x, box_max.y, box_min.z,
                                      box_min.x, box_max.y, box_max.z);
   drawCylinderFromTo(cylinderRadius, box_max.x, box_min.y, box_min.z,
                                      box_max.x, box_min.y, box_max.z);
   drawCylinderFromTo(cylinderRadius, box_max.x, box_max.y, box_min.z,
                                      box_max.x, box_max.y, box_max.z);

   // left to right
   drawCylinderFromTo(cylinderRadius, box_min.x, box_min.y, box_min.z,
                                      box_max.x, box_min.y, box_min.z);
   drawCylinderFromTo(cylinderRadius, box_min.x, box_max.y, box_min.z,
                                      box_max.x, box_max.y, box_min.z);
   drawCylinderFromTo(cylinderRadius, box_min.x, box_min.y, box_max.z,
                                      box_max.x, box_min.y, box_max.z);
   drawCylinderFromTo(cylinderRadius, box_min.x, box_max.y, box_max.z,
                                      box_max.x, box_max.y, box_max.z);

   // bottom to top
   drawCylinderFromTo(cylinderRadius, box_min.x, box_min.y, box_min.z,
                                      box_min.x, box_max.y, box_min.z);
   drawCylinderFromTo(cylinderRadius, box_max.x, box_min.y, box_min.z,
                                      box_max.x, box_max.y, box_min.z);
   drawCylinderFromTo(cylinderRadius, box_max.x, box_min.y, box_max.z,
                                      box_max.x, box_max.y, box_max.z);
   drawCylinderFromTo(cylinderRadius, box_min.x, box_min.y, box_max.z,
                                      box_min.x, box_max.y, box_max.z);
}


void VoxelVisualization::drawVoxelsAsCubesInstancedMipmapped(
   GLuint voxelTexture,
   unsigned int level0_resX, unsigned int level0_resY, 
   const Camera* const voxelCam)
{
   const int maxMipMapLevel = MipmapRenderer::computeMaxLevel(level0_resX);
   const int currentMipmapLevel = min(maxMipMapLevel, Settings::Instance()->getMipmapLevel());
   const int thisLevelResX = level0_resX / pow(2.0, currentMipmapLevel);
   const int thisLevelResY = level0_resY / pow(2.0, currentMipmapLevel);

   if(mSliceData == 0)
   {
      mSliceData = new GLuint[thisLevelResX * thisLevelResY * sizeof(unsigned int) ];
      mCurrentSliceSize = thisLevelResX * thisLevelResY * sizeof(unsigned int) ;
   }
   else if(mCurrentSliceSize != int(thisLevelResX * thisLevelResY * sizeof(unsigned int) ))
   {
      delete[] mSliceData;
      mSliceData = new GLuint[thisLevelResX * thisLevelResY * sizeof(unsigned int) ];
      mCurrentSliceSize = thisLevelResX * thisLevelResY * sizeof(unsigned int) ;
   }

   Frustum f = voxelCam->getFrustum();

   float voxelWidth  = f.width  / thisLevelResX;
   float voxelHeight = f.height / thisLevelResY;
   float voxelLength = f.zRange / 128;



   //if(firstSliceRun)
   {
      V(glActiveTexture(GL_TEXTURE0));
      V(glBindTexture(GL_TEXTURE_2D, voxelTexture));
      V(glGetTexImage(GL_TEXTURE_2D, currentMipmapLevel, GL_RGBA_INTEGER_EXT, GL_UNSIGNED_INT, mSliceData));
      glBindTexture(GL_TEXTURE_2D, 0);

      int bitsPerChannel = 32;
      GLuint mask = GLuint(pow(2.0, double(bitsPerChannel-1)));

      GLuint val;
      mNumVoxels = 0;
      vector<float> translate;

      // COLLECT DATA

      for (int y = 0; y < thisLevelResY; y++)
      {
         for(int x = 0; x < thisLevelResX; x++)
         {
            for (int channel = 0; channel < 4; channel++)
            {
               int pos = 4 * thisLevelResX * y + 4 * x + channel;
               val = mSliceData[pos];

               mask = GLuint(pow(2.0, double(bitsPerChannel-1)));
               for (int b = 0; b < bitsPerChannel; b++)
               {
                  int z = channel * bitsPerChannel + b;
                  if (val & mask)
                  {
                     mNumVoxels++;
                     translate.push_back(f.left + x * voxelWidth);
                     translate.push_back(f.bottom + y * voxelHeight);
                     translate.push_back(-f.zNear - (z + 1) * voxelLength);
                     translate.push_back(0);
                  }
                  mask = mask >> 1;
               }

            }//end for channel
         }// end for x 

      }// end for y

      unsigned int data_size = mNumVoxels*4*sizeof(float);
      V(glBindBuffer(GL_ARRAY_BUFFER, vboTranslate));
      V(glBufferData(GL_ARRAY_BUFFER, data_size, &translate[0], GL_STATIC_DRAW));
      V(glBindBuffer(GL_ARRAY_BUFFER, 0));
      //firstSliceRun = false;
   }
   pInstancedCubes->useProgram();

   V(glUniform1i(pInstancedCubes->getUniformLocation("tboTranslate"), 0));
   glUniform3f(pInstancedCubes->getUniformLocation("scale"), voxelWidth, voxelHeight, voxelLength);


	glActiveTexture(GL_TEXTURE0);
	V(glBindTexture(GL_TEXTURE_BUFFER_EXT, tboTranslate));
	V(glTexBufferEXT(GL_TEXTURE_BUFFER_EXT, GL_RGBA32F_ARB, vboTranslate));


   // RENDER 
   glDisable(GL_LIGHTING);
   V(glEnable(GL_COLOR_MATERIAL));
   //V(glEnableClientState(GL_NORMAL_ARRAY));
   V(glEnableClientState(GL_COLOR_ARRAY));
   V(glEnableClientState(GL_VERTEX_ARRAY));
   //V(glNormalPointer(GL_FLOAT, 0, normals));
   V(glColorPointer(3, GL_FLOAT, 0, colors));
   V(glVertexPointer(3, GL_FLOAT, 0, vertices));


   glEnable(GL_CULL_FACE);
   glLineWidth(1.0);
   glPushMatrix();
   glMultMatrixf(&voxelCam->getInverseViewMatrix()[0][0]);
   V(glDrawArraysInstancedARB(GL_QUADS, 0, 24, mNumVoxels));
   glDisableClientState(GL_COLOR_ARRAY);
   glColor4f(0.1, 0.1, 0.1, 1);
   V(glDrawArraysInstancedARB(GL_LINES, 0, 24, mNumVoxels));
   glPopMatrix();
   glDisable(GL_CULL_FACE);

   glDisableClientState(GL_VERTEX_ARRAY);  // disable vertex arrays
   //glDisableClientState(GL_NORMAL_ARRAY);


   cout << "Drawing " << mNumVoxels << " voxels. ";
   double density = mNumVoxels / (thisLevelResX * thisLevelResY * 128.0);
   cout << "Data density: " << density * 100 << " % " << "\r"<< endl;

   //delete[] mSliceData;
}


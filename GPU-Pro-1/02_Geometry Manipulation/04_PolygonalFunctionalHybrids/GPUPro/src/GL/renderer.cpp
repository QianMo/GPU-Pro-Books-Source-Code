
#include <time.h>

#include "skin/skin.h"
#include "skin/animationData.h"

#include "renderer.h"

RENDERER& getRenderer()
{
   static RENDERER renderer;
   return renderer;
}

RENDERER::RENDERER() : m_framesRendered(0)
{
   m_eyePosition[0] = m_eyePosition[1] = m_eyePosition[2] = 10.f;

   m_lightPosition[0] = 1.f;
   m_lightPosition[1] = 0.f;
   m_lightPosition[2] = 1.f;

   m_eyePosition[3] = m_lightPosition [3] = 1.f;

   for (int i = 0; i < TECHNIQUE_LAST; i++) {

      TECHNIQUE& technique = m_techniques[i];

      technique.shader        =  0;
      technique.isWireframeOn =  false;
      technique.textureIndex  =  0;
   }

   //m_currentTechniqueType(RENDERER::TECHNIQUE_CUBEMAP)
   updateCurrentTechnique(RENDERER::TECHNIQUE_PROCEDURAL, 0);

}

bool RENDERER::init(int width, int height, const char* windowTitle, int *argc, char** argv)
{
   glutInit(argc, argv);
   
   glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
   glutInitWindowSize(width, height);
   glutCreateWindow(windowTitle);

   if (!checkGL()) {
      return false;
   }

   glClearColor(0.5, 0.5, 0.5, 1.0);

   glEnable(GL_DEPTH_TEST);
   glEnable(GL_NORMALIZE);

   glEnable(GL_TEXTURE_2D);
   glEnable(GL_TEXTURE_CUBE_MAP); 
   glEnable(GL_TEXTURE_3D);


   glutReportErrors();

   return true;
}

bool RENDERER::initShaders(const LIST_DESC& shaderNames, const char* shaderPath)
{
   if (shaderNames.itemsSize > NUM_ELEMS(m_techniques)) {
      assert(0);
      return false;
   }

   static const char* extVS = "vsh";
   static const char* extFS = "fsh";

   char pathVS[STR_LENGTH_MAX], pathFS[STR_LENGTH_MAX];

	GLhandleARB vertexShader = 0, fragmentShader = 0;   

   // create all shaders
   for (int i = 0; i < shaderNames.itemsSize; i++ ) {

      sprintf(pathVS, "%s/%s.%s", shaderPath, shaderNames.itemNames[i], extVS);
      sprintf(pathFS, "%s/%s.%s", shaderPath, shaderNames.itemNames[i], extFS);

      if(!createShaders(pathVS, &vertexShader, pathFS, &fragmentShader, &m_techniques[i].shader)) {
         return false;
      }
   }

   return true;
}
   
bool RENDERER::initTextures(TEXTURE_TYPE type, const LIST_DESC& textureNames, const char* texturePath)
{

   TEXTURE_HANDLES* textures = NULL;
   
   switch (type) {
   case TEXTURE_2D:      
      textures = &m_textures2D;
      break;
   case TEXTURE_CUBEMAP:      
      textures = &m_texturesCube;
      break;
   case TEXTURE_3D:
      textures = &m_textures3D;
      break;
   default:
      assert(0);
      return false;
   }

   textures->resize(textureNames.itemsSize);

   char textureName[STR_LENGTH_MAX];

   // create all textures of this type
   for (int i = 0; i < textureNames.itemsSize; i++ ) {

      sprintf(textureName, "%s/%s", texturePath, textureNames.itemNames[i]);      

      createTexture(type, textureName, &(*textures)[i]);
   }   

   return true;
}

void RENDERER::updateCurrentTechnique(RENDERER::TECHNIQUE_TYPE type, int textureIndex, bool isWireframeOn) 
{ 
   m_currentTechniqueType = type; 

   m_techniques[m_currentTechniqueType].isWireframeOn = isWireframeOn;
   m_techniques[m_currentTechniqueType].textureIndex  = textureIndex;

}

// basic scene setup
void RENDERER::preRender(const float* translation, const float* rotation)
{
   glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
   glMatrixMode(GL_MODELVIEW);
   glLoadIdentity();   
   glTranslatef(translation[0], translation[1], translation[2]);
   glRotatef(rotation[0], 1.0, 0.0, 0.0);
   glRotatef(rotation[1], 0.0, 1.0, 0.0);
   glRotatef(rotation[2], 0.0, 1.0, 1.0);

   m_framesRendered++;
}

bool RENDERER::renderSkin(const SKIN_DATA* skinData, const ANIMATION_DATA* animationData, int currentFrame)
{
   if (!skinData && !animationData) {
      assert(0);
      return false;
   }

   if (animationData->framesNum < 1) {
      // no mesh animation to render for this scene
      return true;
   }

   const TECHNIQUE& currentTechnique = m_techniques[TECHNIQUE_SKINNING];

   GLuint skinProgram = currentTechnique.shader;

   glUseProgramObjectARB ( skinProgram );

   SET_UNIFORM_VECTOR( skinProgram, eyePosition,   m_eyePosition );
   SET_UNIFORM_VECTOR( skinProgram, lightPosition, m_lightPosition );

   static GLuint colorAttrib = glGetAttribLocationARB(skinProgram, "colorCustom");
   static GLuint jointWeightsAttrib = glGetAttribLocationARB(skinProgram, "jointWeights");
   static GLuint jointIndicesAttrib = glGetAttribLocationARB(skinProgram, "jointIndices");
   static GLuint normalsAttrib = glGetAttribLocationARB(skinProgram, "normal");
   static GLuint uvsAttrib = glGetAttribLocationARB(skinProgram, "uvCoords");

   static int boneMatrixLocation = glGetUniformLocationARB(skinProgram, "jointMatrices[0]");   

   SET_UNIFORM_TEXTURE( skinProgram, diffuseTexture, m_textures2D[ currentTechnique.textureIndex ], GL_TEXTURE_2D);

   glUniformMatrix4fvARB(boneMatrixLocation, animationData->matricesPerSample, GL_FALSE, animationData->getMatrixAt(currentFrame, 0).x );

   glEnableClientState(GL_VERTEX_ARRAY);
   glVertexPointer(3, GL_FLOAT, 0, &skinData->meshVertices[0]);

   glEnableClientState(GL_NORMAL_ARRAY);
   glNormalPointer(GL_FLOAT, 0, &skinData->meshNormals[0]);
   
   glEnableVertexAttribArray(colorAttrib);		
   glVertexAttribPointer(colorAttrib, 4, GL_FLOAT, GL_FALSE, 0, &skinData->colors[0].w[0]);

   glEnableVertexAttribArray(jointWeightsAttrib);		   
   glVertexAttribPointer(jointWeightsAttrib, 4, GL_FLOAT, GL_FALSE, 0, &skinData->verticesWeights[0].w[0]);

   glEnableVertexAttribArray(jointIndicesAttrib);		   
   glVertexAttribPointer(jointIndicesAttrib, 4, GL_FLOAT, GL_FALSE, 0, &skinData->jointIndices[0].i[0]);

   glEnableVertexAttribArray(uvsAttrib);
   glVertexAttribPointer(uvsAttrib, 2, GL_FLOAT, GL_FALSE, 0, &skinData->meshUVs[0]);

   int numTriangles = (int)(skinData->meshTriangles.size() * 3) ;
   glDrawElements(GL_TRIANGLES, numTriangles, GL_UNSIGNED_INT, &skinData->meshTriangles[0] );

   glDisableClientState(GL_VERTEX_ARRAY);
   glDisableClientState(GL_NORMAL_ARRAY);

   glDisableVertexAttribArray(colorAttrib);		
   glDisableVertexAttribArray(jointWeightsAttrib);
   glDisableVertexAttribArray(jointIndicesAttrib);
   glDisableVertexAttribArray(uvsAttrib);		

   return true;
}

bool RENDERER::renderGeometry(const GEOMETRY_DESC& geometryDesc, const MATRIX_STORE* parentMatrix)
{
   if (parentMatrix != NULL) {
      // update according to parent transform
      glMultMatrixf(parentMatrix->x);
   }

   const TECHNIQUE& currentTechnique = m_techniques[m_currentTechniqueType];
   glPolygonMode(GL_FRONT_AND_BACK, currentTechnique.isWireframeOn ? GL_LINE : GL_FILL);

   GLhandleARB curProgram =  currentTechnique.shader;

   glUseProgramObjectARB ( curProgram );       

   // not using static init here, as different shaders might have different locations of the variables
   // it would obviously be better to cache the locations
   setUniformVector( curProgram, "eyePosition",   m_eyePosition );
   setUniformVector( curProgram, "lightPosition", m_lightPosition );   
   setUniformFloat( curProgram,  "scale", 0.1f );
   setUniformFloat( curProgram,  "offset", 0.f );
   setUniformFloat(  curProgram, "specularPower", 110.f );      

   switch (m_currentTechniqueType) {
   case TECHNIQUE_CUBEMAP:
      {
         SET_UNIFORM_TEXTURE( curProgram, cubeMap, m_texturesCube[ currentTechnique.textureIndex ], GL_TEXTURE_CUBE_MAP);
         break;
      }
   case TECHNIQUE_PROCEDURAL:
      {
         setUniformFloat( curProgram,  "scale", 0.08f );
         setUniformFloat( curProgram,  "tick", clock() / 1000.f );

         SET_UNIFORM_TEXTURE( curProgram, volumeTexture, m_textures3D[ currentTechnique.textureIndex ], GL_TEXTURE_3D );
         break;
      }
   case TECHNIQUE_TRIPLANAR:
      {
         SET_UNIFORM_TEXTURE( curProgram, planarTexture, m_textures2D[ currentTechnique.textureIndex ], GL_TEXTURE_2D );
         break;
      }
   default:
      assert(0);
   }

   renderBuffers( geometryDesc.positions, geometryDesc.normals, geometryDesc.numElemets);
  
   glUseProgramObjectARB ( 0 );

   glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

   return true;
}

bool RENDERER::createTexture(TEXTURE_TYPE type, const char* name, GLuint* handle)
{
   switch (type) {
   case TEXTURE_2D:
      return createGLTexture(name, handle);
   case TEXTURE_CUBEMAP:
     return createGLCubeMap(name, handle);
   case TEXTURE_3D:
      return createGLTexture3D(name, handle);
      break;   
   }

   assert(0);
   return false;
}
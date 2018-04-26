#ifndef __GL_RENDERER_H__
#define __GL_RENDERER_H__

#include <vector>
#include <string.h>

#include "../defines.h"
#include "glUtils.h"

struct SKIN_DATA;
struct ANIMATION_DATA;

// basic rendering for all parts of the demo (for the sake of simplicity it also loads the data while initialising)
class RENDERER {
public:

   enum TEXTURE_TYPE {
      TEXTURE_2D,
      TEXTURE_CUBEMAP,
      TEXTURE_3D,
   };

   enum TECHNIQUE_TYPE {
      TECHNIQUE_SKINNING,
      TECHNIQUE_CUBEMAP,
      TECHNIQUE_TRIPLANAR,
      TECHNIQUE_PROCEDURAL,

      TECHNIQUE_LAST
   };

   typedef  GLuint                        TEXTURE_HANDLE;
   typedef  std::vector<TEXTURE_HANDLE>   TEXTURE_HANDLES;

   struct TECHNIQUE {
      GLhandleARB shader;
      bool        isWireframeOn;
      int         textureIndex;
   };

   RENDERER();

   bool  init           (int width, int height, const char* windowTitle, int *argc, char** argv);
         
   bool  initShaders    (const LIST_DESC& shaderNames, const char* shaderPath);
   bool  initTextures   (TEXTURE_TYPE type, const LIST_DESC& textureNames, const char* texturePath);

   // basic scene setup
   void  preRender      (const float* translation, const float* rotation);

   bool  renderSkin     (const SKIN_DATA* skinData, const ANIMATION_DATA* animData, int currentFrame);

   bool  renderGeometry (const GEOMETRY_DESC& geometryDesc, const MATRIX_STORE* parentMatrix);

   //bool  renderScene    (const SCENE_DATA& scene, int currentFrame);

   void  updateCurrentTechnique(RENDERER::TECHNIQUE_TYPE type, int textureIndex, bool isWireframeOn = false);

   //bool  updateTechnique(TECHNIQUE_TYPE type, bool isWireframeOn);

   int   getFramesRendered() const { return m_framesRendered; }

   void  resetFrames()  { m_framesRendered = 0; }


private:

   bool createTexture(TEXTURE_TYPE type, const char* name, GLuint* handle);

   // available shaders   
   TECHNIQUE m_techniques[TECHNIQUE_LAST];

   TECHNIQUE_TYPE m_currentTechniqueType;
   
   int   m_framesRendered;

   float	m_eyePosition[4];
   float	m_lightPosition[4];  

   // registered textures 
   TEXTURE_HANDLES  m_textures2D;
   TEXTURE_HANDLES  m_texturesCube;
   TEXTURE_HANDLES  m_textures3D;

   int m_currentTex2D, m_currentTexCube, m_currentTex3D;
};

RENDERER& getRenderer();

#endif //__GL_RENDERER_H__
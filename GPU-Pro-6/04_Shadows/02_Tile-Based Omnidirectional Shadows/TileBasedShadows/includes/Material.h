#ifndef MATERIAL_H
#define MATERIAL_H

#include <OGL_RasterizerState.h>
#include <OGL_DepthStencilState.h>
#include <OGL_BlendState.h>

class OGL_Shader;
class OGL_Texture;

// Material
//
// - Loaded from a simple text-file (".mtl") with 3 blocks:
//   1."Textures"
//     - "ColorTexture"
//     - "NormalTexture" 
//     - "SpecularTexture"
//   2."RenderStates"
//     - "cull" -> culling -> requires 1 additional parameter: cull mode    
//     - "noDepthTest" -> disable depth-testing
//     - "noDepthMask" -> disable depth-mask
//     - "colorBlend" -> color blending -> requires 3 additional parameters: srcColorBlend/ dstColorBlend/ blendColorOp
//     - "alphaBlend" -> alpha blending -> requires 3 additional parameters: srcAlphaBlend/ dstAlphaBlend/ blendAlphaOp
//   3."Shader"
//     - "permutation" -> requires 1 additional parameter: permutation mask of shader
//     - "file" -> requires 1 additional parameter: filename of shader
// - all parameters are optional 
// - order of parameters is indifferent (except in "Shader": permutation must be specified before file)   
class Material
{
public:
  Material():
    colorTexture(NULL),
    normalTexture(NULL),
    specularTexture(NULL),
    rasterizerState(NULL),
    depthStencilState(NULL),
    blendState(NULL),
    shader(NULL)
  {
    name[0] = 0;
  }

  bool Load(const char *fileName);

  const char* GetName() const
  {
    return name;
  }

  OGL_Texture *colorTexture;
  OGL_Texture *normalTexture;
  OGL_Texture *specularTexture;
  OGL_RasterizerState *rasterizerState;
  OGL_DepthStencilState *depthStencilState;
  OGL_BlendState *blendState;
  OGL_Shader *shader;

private:
  // load "Textures"-block
  bool LoadTextures(std::ifstream &file);
  
  // load "RenderStates"-block
  void LoadRenderStates(std::ifstream &file);
  
  // load "Shader"-block
  bool LoadShader(std::ifstream &file);
  
  char name[DEMO_MAX_FILENAME];
  RasterizerDesc rasterDesc;	
  DepthStencilDesc depthStencilDesc;
  BlendDesc blendDesc;
  
};

#endif
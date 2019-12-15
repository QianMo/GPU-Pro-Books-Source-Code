#ifndef MATERIAL_H
#define MATERIAL_H

#include <List.h>
#include <DX12_PipelineState.h>

enum textureIDs
{
  COLOR_TEX_ID=0,
  NORMAL_TEX_ID,
  SPECULAR_TEX_ID,
  NUM_TEX_IDS
};

class DX12_Shader;
class DX12_Texture;

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
//     - "alphaTested" -> alpha testing
//     - "receiveDecals" -> receive decals
//   3."Shader"
//     - "permutation" -> requires 1 additional parameter: permutation mask of shader
//     - "file" -> requires 1 additional parameter: filename of shader
// - all parameters are optional 
// - order of parameters is indifferent (except "Shader": must be last block and permutation must be specified before file)   
class Material
{
public:
  Material():
    shader(nullptr),
    alphaTested(false),
    receiveDecals(false)
  {
    name[0] = 0;
    memset(textures, 0, sizeof(DX12_Texture*) * NUM_TEX_IDS);
    rasterDesc.Reset();
    depthStencilDesc.Reset();
    blendDesc.Reset();
  }

  bool Load(const char *fileName);

  const char* GetName() const
  {
    return name;
  }

  DX12_Texture *textures[NUM_TEX_IDS];
  DX12_Shader *shader;
  RasterizerDesc rasterDesc;
  DepthStencilDesc depthStencilDesc;
  BlendDesc blendDesc;
  bool alphaTested;
  bool receiveDecals;

private:
  // load "Textures"-block
  bool LoadTextures(std::ifstream &file);
  
  // load "RenderStates"-block
  void LoadRenderStates(std::ifstream &file);
  
  // load "Shader"-block
  bool LoadShader(std::ifstream &file);
  
  char name[DEMO_MAX_FILENAME];
  
};

#endif
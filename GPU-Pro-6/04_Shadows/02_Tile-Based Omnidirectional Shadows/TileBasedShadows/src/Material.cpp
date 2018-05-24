#include <stdafx.h>
#include <Demo.h>
#include <Material.h>

#define NUM_RENDER_STATES 27 // number of render-states that can be specified in material file

struct MaterialInfo
{
  const char *name; // info as string
  int mode; // info as int
};

// render-states
static MaterialInfo renderStateList[NUM_RENDER_STATES] =
{ 
  "NONE_CULL",NONE_CULL, "FRONT_CULL",FRONT_CULL, "BACK_CULL",BACK_CULL, 
  "ZERO_BLEND",ZERO_BLEND, "ONE_BLEND",ONE_BLEND, "SRC_COLOR_BLEND",SRC_COLOR_BLEND,
  "INV_SRC_COLOR_BLEND",INV_SRC_COLOR_BLEND, "DST_COLOR_BLEND",DST_COLOR_BLEND,
  "INV_DST_COLOR_BLEND",INV_DST_COLOR_BLEND, "SRC_ALPHA_BLEND",SRC_ALPHA_BLEND,
  "INV_SRC_ALPHA_BLEND",INV_SRC_ALPHA_BLEND, "DST_ALPHA_BLEND",DST_ALPHA_BLEND, 
  "INV_DST_ALPHA_BLEND",INV_DST_ALPHA_BLEND, "CONST_COLOR_BLEND",CONST_COLOR_BLEND,
  "INV_CONST_COLOR_BLEND",INV_CONST_COLOR_BLEND, "CONST_ALPHA_BLEND",CONST_ALPHA_BLEND,
  "INV_CONST_ALPHA_BLEND",INV_CONST_ALPHA_BLEND, "SRC_ALPHA_SAT_BLEND",SRC_ALPHA_SAT_BLEND, 
  "SRC1_COLOR_BLEND",SRC1_COLOR_BLEND, "INV_SRC1_COLOR_BLEND",INV_SRC1_COLOR_BLEND,
  "SRC1_ALPHA_BLEND",SRC1_ALPHA_BLEND, "INV_SRC1_ALPHA_BLEND",INV_SRC1_ALPHA_BLEND,
  "ADD_BLEND_OP",ADD_BLEND_OP, "SUBTRACT_BLEND_OP",SUBTRACT_BLEND_OP,
  "REV_SUBTRACT_BLEND_OP",REV_SUBTRACT_BLEND_OP, "MIN_BLEND_OP",MIN_BLEND_OP,    
  "MAX_BLEND_OP",MAX_BLEND_OP
};

// map string into corresponding render-state
#define STR_TO_STATE(str, stateType, state) \
  for(unsigned int i=0; i<NUM_RENDER_STATES; i++) \
    if(strcmp(renderStateList[i].name, str.c_str()) == 0) \
      state = (stateType)renderStateList[i].mode;

bool Material::LoadTextures(std::ifstream &file)
{ 
  std::string str, token;
  file >> token;
  while(true)
  {
    file >> str;
    if((str == "}") || (file.eof()))
      break;
    else if(str == "ColorTexture")
    {
      file >> str;
      colorTexture = Demo::resourceManager->LoadTexture(str.c_str());
      if(!colorTexture)
        return false;
    }
    else if(str == "NormalTexture")
    {
      file >> str;
      normalTexture = Demo::resourceManager->LoadTexture(str.c_str());
      if(!normalTexture)
        return false;
    }
    else if(str == "SpecularTexture")
    { 
      file >> str;
      specularTexture = Demo::resourceManager->LoadTexture(str.c_str());
      if(!specularTexture)
        return false;
    }
  }
  return true;
}

void Material::LoadRenderStates(std::ifstream &file)
{
  std::string str, token;
  file >> token;
  while(true)
  {
    file >> str;
    if((str == "}") || (file.eof()))
      break;
    else if(str == "cull")
    {
      file >> str;
      STR_TO_STATE(str, cullModes, rasterDesc.cullMode);
    }
    else if(str == "noDepthTest")
      depthStencilDesc.depthTest = false;
    else if(str == "noDepthMask")
      depthStencilDesc.depthMask = false;
    else if(str == "colorBlend")
    {
      blendDesc.blend = true;
      file >> str;
      STR_TO_STATE(str, blendOptions, blendDesc.srcColorBlend);
      file >> str;
      STR_TO_STATE(str, blendOptions, blendDesc.dstColorBlend);
      file >> str;
      STR_TO_STATE(str, blendOps, blendDesc.blendColorOp);
    }
    else if(str == "alphaBlend")
    {
      blendDesc.blend = true;
      file >> str;
      STR_TO_STATE(str, blendOptions, blendDesc.srcAlphaBlend);
      file >> str;
      STR_TO_STATE(str, blendOptions, blendDesc.dstAlphaBlend);
      file >> str;
      STR_TO_STATE(str, blendOps, blendDesc.blendAlphaOp);
    }
  }	
}

bool Material::LoadShader(std::ifstream &file)
{
  std::string str, token;
  unsigned int permutationMask = 0;
  file >> token;
  while(true)
  {
    file >> str;
    if((str == "}") || (file.eof()))
      break;
    else if(str == "permutation")
    {
      file >> permutationMask;
    }
    else if(str == "file")
    {
      file >> str;
      shader = Demo::resourceManager->LoadShader(str.c_str(), permutationMask);
      if(!shader)
        return false;
    } 
  }
  return true;
}

bool Material::Load(const char *fileName)
{
  strcpy(name, fileName);
  char filePath[DEMO_MAX_FILEPATH];
  if(!Demo::fileManager->GetFilePath(fileName, filePath))
    return false;
  std::ifstream file(filePath, std::ios::in);
  if(!file.is_open())
    return false;

  std::string str, token;
  file >> str;
  while(!file.eof())
  {
    if(str == "Textures")
    {
      if(!LoadTextures(file))
      {
        file.close();
        return false;
      }
    }
    else if(str == "RenderStates")
    {
      LoadRenderStates(file);
    }
    else if(str == "Shader")
    {
      if(!LoadShader(file))
      {
        file.close();
        return false;
      }
    }
    file >> str;
  } 
  file.close();

  rasterizerState = Demo::renderer->CreateRasterizerState(rasterDesc);
  if(!rasterizerState)
    return false;

  // Increment for all opaque geometry the stencil buffer. In this way illumination 
  // can be restricted to the area, where actually the scene geometry is located. 
  // On the other hand the sky can be easily rendered to the area, where the stencil 
  // buffer is still left to 0.
  if(!blendDesc.blend)
    depthStencilDesc.stencilTest = true;
  depthStencilState = Demo::renderer->CreateDepthStencilState(depthStencilDesc);
  if(!depthStencilState)
    return false;

  blendState = Demo::renderer->CreateBlendState(blendDesc);
  if(!blendState)
    return false;

  return true;
}


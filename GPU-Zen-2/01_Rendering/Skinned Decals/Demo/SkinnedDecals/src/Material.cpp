#include <stdafx.h>
#include <Demo.h>
#include <Material.h>

#define NUM_RENDER_STATES 25 // number of render-states that can be specified in material file

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
  "INV_DST_ALPHA_BLEND",INV_DST_ALPHA_BLEND, "BLEND_FACTOR_BLEND",BLEND_FACTOR_BLEND,
  "INV_BLEND_FACTOR_BLEND",INV_BLEND_FACTOR_BLEND, "SRC_ALPHA_SAT_BLEND",SRC_ALPHA_SAT_BLEND, 
  "SRC1_COLOR_BLEND",SRC1_COLOR_BLEND, "INV_SRC1_COLOR_BLEND",INV_SRC1_COLOR_BLEND,
  "SRC1_ALPHA_BLEND",SRC1_ALPHA_BLEND, "INV_SRC1_ALPHA_BLEND",INV_SRC1_ALPHA_BLEND,
  "ADD_BLEND_OP",ADD_BLEND_OP, "SUBTRACT_BLEND_OP",SUBTRACT_BLEND_OP,
  "REV_SUBTRACT_BLEND_OP",REV_SUBTRACT_BLEND_OP, "MIN_BLEND_OP",MIN_BLEND_OP,    
  "MAX_BLEND_OP",MAX_BLEND_OP
};

// map string into corresponding render-state
#define STR_TO_STATE(str, stateType, state) \
  for(UINT i=0; i<NUM_RENDER_STATES; i++) \
    if(strcmp(renderStateList[i].name, str.c_str()) == 0) \
      state = (stateType)renderStateList[i].mode;

bool Material::LoadTextures(std::ifstream &file)
{ 
  std::string str, token;
  file >> token;
  while(true)
  {
    int textureID = -1;
    textureFlags flags = STATIC_TEXTURE_FLAG;
    file >> str;
    if((str == "}") || (file.eof()))
      break;
    else if(str == "ColorTexture")
    {
      textureID = COLOR_TEX_ID;
      flags |= SRGB_READ_TEXTURE_FLAG;
    }
    else if(str == "NormalTexture")
    {
      textureID = NORMAL_TEX_ID;
    }
    else if(str == "SpecularTexture")
    { 
      textureID = SPECULAR_TEX_ID;
    }
    if(textureID > -1)
    {
      file >> str;
      textures[textureID] = Demo::resourceManager->LoadTexture(str.c_str(), flags);
      if(!textures[textureID])
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
      blendDesc.blendEnable = true;
      file >> str;
      STR_TO_STATE(str, blendStates, blendDesc.srcColorBlend);
      file >> str;
      STR_TO_STATE(str, blendStates, blendDesc.dstColorBlend);
      file >> str;
      STR_TO_STATE(str, blendOps, blendDesc.blendColorOp);
    }
    else if(str == "alphaBlend")
    {
      blendDesc.blendEnable = true;
      file >> str;
      STR_TO_STATE(str, blendStates, blendDesc.srcAlphaBlend);
      file >> str;
      STR_TO_STATE(str, blendStates, blendDesc.dstAlphaBlend);
      file >> str;
      STR_TO_STATE(str, blendOps, blendDesc.blendAlphaOp);
    }
    else if(str == "alphaTested")
    {
      alphaTested = true;
    }
    else if(str == "receiveDecals")
    {
      receiveDecals = true;
    }
  }	
}

bool Material::LoadShader(std::ifstream &file)
{
  std::string str, token;
  UINT permutationMask = 0;
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

  return true;
}


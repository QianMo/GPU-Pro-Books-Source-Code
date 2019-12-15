#include <stdafx.h>
#include <Demo.h>
#include <ResourceManager.h>

void ResourceManager::Release()
{
  SAFE_DELETE_PLIST(shaders);
  SAFE_DELETE_PLIST(textures);
  SAFE_DELETE_PLIST(materials);
  SAFE_DELETE_PLIST(fonts);
  SAFE_DELETE_PLIST(demoModels);
}

DX12_Shader* ResourceManager::LoadShader(const char *fileName, UINT permutationMask)
{
  assert(fileName != nullptr);
  for(UINT i=0; i<shaders.GetSize(); i++)
  {
    if((strcmp(shaders[i]->GetName(), fileName) == 0) &&
      (shaders[i]->GetPermutationMask() == permutationMask))
    {
      return shaders[i];
    }
  }
  DX12_Shader *shader = new DX12_Shader;
  if(!shader)
    return nullptr;
  if(!shader->Load(fileName, permutationMask))
  {
    SAFE_DELETE(shader);
    LOG_ERROR("Failed to load shader: %s (permutationMask: %i)", fileName, permutationMask);
    return nullptr;
  }
  shaders.AddElement(&shader);
  return shader;
}

DX12_Texture* ResourceManager::LoadTexture(const char *fileName, textureFlags flags)
{
  assert(fileName != nullptr);
  for(UINT i=0; i<textures.GetSize(); i++)
  {
    if(strcmp(textures[i]->GetName(), fileName) == 0)
    {
      if(textures[i]->GetTextureDesc().flags == flags)
        return textures[i];
    }
  }
  DX12_Texture *texture = new DX12_Texture;
  if(!texture)
    return nullptr;
  if(!texture->LoadFromFile(fileName, flags))
  {
    SAFE_DELETE(texture);
    LOG_ERROR("Failed to load texture: %s", fileName);
    return nullptr;
  }
  textures.AddElement(&texture);
  return texture;
}

DX12_Texture* ResourceManager::CreateTexture(const Image **images, UINT numImages, textureFlags flags)
{
  assert(images[0]->GetName() != nullptr);
  DX12_Texture *texture = new DX12_Texture;
  if(!texture)
    return nullptr;
  if(!texture->Create(images, numImages, flags))
  {
    SAFE_DELETE(texture);
    LOG_ERROR("Failed to create texture: %s", images[0]->GetName());
    return nullptr;
  }
  textures.AddElement(&texture);
  return texture;
}

DX12_Texture* ResourceManager::CreateTexture(const TextureDesc &desc, const char *name, const DX12_TextureData *initData)
{
  assert(name != nullptr);
  DX12_Texture *texture = new DX12_Texture;
  if(!texture)
    return nullptr;
  if(!texture->Create(desc, name, initData))
  {
    SAFE_DELETE(texture);
    LOG_ERROR("Failed to create texture: %s", name);
    return nullptr;
  }
  textures.AddElement(&texture);
  return texture;
}

Material* ResourceManager::LoadMaterial(const char *fileName)
{
  assert(fileName != nullptr);
  for(UINT i=0; i<materials.GetSize(); i++)
  {
    if(strcmp(materials[i]->GetName(), fileName) == 0)
      return materials[i];
  }
  Material *material = new Material;
  if(!material)
    return nullptr;
  if(!material->Load(fileName))
  {
    SAFE_DELETE(material);
    LOG_ERROR("Failed to load material: %s", fileName);
    return nullptr;
  }
  materials.AddElement(&material);
  return material;
}

Font* ResourceManager::LoadFont(const char *fileName)
{
  assert(fileName != nullptr);
  for(UINT i=0; i<fonts.GetSize(); i++)
  {
    if(strcmp(fonts[i]->GetName(), fileName) == 0)
      return fonts[i];
  }
  Font *font = new Font;
  if(!font)
    return nullptr;
  if(!font->Load(fileName))
  {
    SAFE_DELETE(font);
    LOG_ERROR("Failed to load font: %s", fileName);
    return nullptr;
  }
  fonts.AddElement(&font);
  return font;
}

DemoModel* ResourceManager::LoadDemoModel(const char *fileName)
{
  DemoModel *demoModel = new DemoModel;
  if(!demoModel)
    return nullptr;
  if(!demoModel->Load(fileName))
  {
    SAFE_DELETE(demoModel);
    LOG_ERROR("Failed to load demo model: %s", fileName);
    return nullptr;
  }
  demoModels.AddElement(&demoModel);
  return demoModel;
}

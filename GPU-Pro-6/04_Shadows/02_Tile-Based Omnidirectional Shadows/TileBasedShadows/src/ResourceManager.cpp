#include <stdafx.h>
#include <Demo.h>
#include <ResourceManager.h>

void ResourceManager::Release()
{ 
  SAFE_DELETE_PLIST(shaders);
  SAFE_DELETE_PLIST(textures);
  SAFE_DELETE_PLIST(materials);
  SAFE_DELETE_PLIST(fonts); 
  SAFE_DELETE_PLIST(demoMeshes);
}

OGL_Shader* ResourceManager::LoadShader(const char *fileName, unsigned int permutationMask)
{
  for(unsigned int i=0; i<shaders.GetSize(); i++)
  {
    if((strcmp(shaders[i]->GetName(), fileName) == 0) && (shaders[i]->GetPermutationMask() == permutationMask))
    {
      return shaders[i];
    }
  }
  OGL_Shader *shader = new OGL_Shader;
  if(!shader)
    return NULL;
  if(!shader->Load(fileName, permutationMask))
  {
    SAFE_DELETE(shader);
    return NULL;
  }
  shaders.AddElement(&shader);
  return shader;
}

OGL_Texture* ResourceManager::LoadTexture(const char *fileName)
{
  for(unsigned int i=0; i<textures.GetSize(); i++)
  {
    if(strcmp(textures[i]->GetName(), fileName) == 0)
      return textures[i];
  }
  OGL_Texture *texture = new OGL_Texture;
  if(!texture)
    return NULL;
  if(!texture->LoadFromFile(fileName))
  {
    SAFE_DELETE(texture);
    return NULL;
  }
  textures.AddElement(&texture);
  return texture;
}

Material* ResourceManager::LoadMaterial(const char *fileName)
{
  for(unsigned int i=0; i<materials.GetSize(); i++)
  {
    if(strcmp(materials[i]->GetName(), fileName) == 0)
      return materials[i];
  }
  Material *material = new Material;
  if(!material)
    return NULL;
  if(!material->Load(fileName))
  {
    SAFE_DELETE(material);
    return NULL;
  }
  materials.AddElement(&material);
  return material;
}

Font* ResourceManager::LoadFont(const char *fileName)
{
  for(unsigned int i=0; i<fonts.GetSize(); i++)
  {
    if(strcmp(fonts[i]->GetName(), fileName) == 0)
      return fonts[i];
  }
  Font *font = new Font;
  if(!font)
    return NULL;
  if(!font->Load(fileName))
  {
    SAFE_DELETE(font);
    return NULL;
  }
  fonts.AddElement(&font); 
  return font;
}

DemoMesh* ResourceManager::LoadDemoMesh(const char *fileName)
{
  DemoMesh *demoMesh = new DemoMesh;
  if(!demoMesh)
    return NULL;
  if(!demoMesh->Load(fileName))
  {
    SAFE_DELETE(demoMesh);
    return NULL;
  }
  demoMeshes.AddElement(&demoMesh);
  return demoMesh;
}




#ifndef RESOURCE_MANAGER_H
#define RESOURCE_MANAGER_H

#include <List.h>
#include <DX12_Shader.h>
#include <DX12_Texture.h>
#include <Material.h>
#include <Font.h>
#include <DemoModel.h>

// ResourceManager
//
// Manages resources (shaders, textures, materials, fonts, demo-models).
class ResourceManager
{
public:
  ResourceManager()
  {
  }

  ~ResourceManager()
  {
    Release();
  }

  void Release();

  DX12_Shader* LoadShader(const char *fileName, UINT permutationMask=0);

  DX12_Texture* LoadTexture(const char *name, textureFlags flags=NONE_TEXTURE_FLAG);

  DX12_Texture* CreateTexture(const Image **images, UINT numImages, textureFlags flags=NONE_TEXTURE_FLAG);

  DX12_Texture* CreateTexture(const TextureDesc &desc, const char *name, const DX12_TextureData *initData=nullptr);

  Material* LoadMaterial(const char *fileName);

  Material* GetMaterial(UINT index)
  {
    assert(index < materials.GetSize());
    return materials[index];
  }

  Font* LoadFont(const char *fileName);

  Font* GetFont(UINT index)
  {
    assert(index < fonts.GetSize());
    return fonts[index];
  }

  DemoModel* LoadDemoModel(const char *fileName);

  DemoModel* GetDemoModel(UINT index) const
  {
    assert(index < demoModels.GetSize());
    return demoModels[index];
  }

  UINT GetNumMaterials() const
  {
    return materials.GetSize();
  }

  UINT GetNumFonts() const
  {
    return fonts.GetSize();
  }

  UINT GetNumDemoModels() const
  {
    return demoModels.GetSize();
  }

private:
  List<DX12_Shader*> shaders;
  List<DX12_Texture*> textures;
  List<Material*> materials;
  List<Font*> fonts;
  List<DemoModel*> demoModels;

};

#endif

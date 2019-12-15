#ifndef DX12_TEXTURE_H
#define DX12_TEXTURE_H

#include <render_states.h>
#include <Image.h>
#include <DX12_IResource.h>

BITFLAGS_ENUM(UINT, textureFlags)
{
  NONE_TEXTURE_FLAG                 = 0,
  BACK_BUFFER_TEXTURE_FLAG          = 1,
  RENDER_TARGET_TEXTURE_FLAG        = 2,
  DEPTH_STENCIL_TARGET_TEXTURE_FLAG = 4,
  SRGB_READ_TEXTURE_FLAG            = 8,
  SRGB_WRITE_TEXTURE_FLAG           = 16,
  STATIC_TEXTURE_FLAG               = 32,
  UAV_TEXTURE_FLAG                  = 64
};

enum targetTypes
{
  TEXTURE_2D=0,
  TEXTURE_2D_ARRAY,
  TEXTURE_CUBE,
  TEXTURE_CUBE_ARRAY,
  TEXTURE_3D
};

struct TextureDesc
{
  TextureDesc() :
    width(0),
    height(0),
    depth(1),
    format(NONE_RENDER_FORMAT),
    numMips(1),
    target(TEXTURE_2D),
    flags(NONE_TEXTURE_FLAG),
    initResourceState(COMMON_RESOURCE_STATE),
    clearDepth(1.0f),
    clearStencil(0)
  {
  }

  UINT width;
  UINT height;
  UINT depth;
  renderFormats format;
  UINT numMips;
  targetTypes target;
  textureFlags flags;
  resourceStates initResourceState;
  Color clearColor;
  float clearDepth;
  unsigned char clearStencil;
};


class DX12_TextureData
{
public:
  DX12_TextureData(const Image **images, UINT numImages);

  ~DX12_TextureData();

  operator D3D12_SUBRESOURCE_DATA* () const
  {
    return resourceData;
  }

private:
  template <typename T>
  void ExpandData(const T *src, T *dest, UINT width, UINT height, UINT depth,
    UINT arrayIndex, UINT channelSize)
  {
    for(UINT z=0; z<depth; z++)
    {
      for(UINT y=0; y<height; y++)
      {
        for(UINT x=0; x<width; x++)
        {
          UINT srcIndex = 3 * channelSize * ((arrayIndex * width * height * depth) + (z * height * width) + (y * width) + x);
          UINT destIndex = 4 * channelSize * ((z * height * width) + (y * width) + x);
          dest[destIndex] = src[srcIndex];
          dest[destIndex + (1 * channelSize)] = src[srcIndex + (1 * channelSize)];
          dest[destIndex + (2 * channelSize)] = src[srcIndex + (2 * channelSize)];
          dest[destIndex + (3 * channelSize)] = 255;
        }
      }
    }
  }

  D3D12_SUBRESOURCE_DATA *resourceData;
  UINT numSubresources;
  bool expandData;
};


// DX12_Texture
//
class DX12_Texture : public DX12_IResource
{
public:
  friend class DX12_ResourceDescTable;

  DX12_Texture():
    resourceState(COMMON_RESOURCE_STATE)
  {
    name[0] = 0;
  }

  bool Create(const TextureDesc &desc, const char *name, const DX12_TextureData *initData=nullptr, UINT backBufferIndex=0);

  bool Create(const Image **images, UINT numImages, textureFlags flags=NONE_TEXTURE_FLAG);

  bool LoadFromFile(const char *fileName, textureFlags flags=NONE_TEXTURE_FLAG);

  virtual ID3D12Resource* GetResource() const override
  {
    return texture.Get();
  }

  virtual ID3D12Resource* GetUploadHeap() const override
  {
    return ((textureDesc.flags & STATIC_TEXTURE_FLAG) && (!(textureDesc.flags & BACK_BUFFER_TEXTURE_FLAG))) ? uploadHeap.Get() : nullptr;
  }

  virtual void SetResourceState(resourceStates resourceState) override
  {
    this->resourceState = resourceState;
  }

  virtual resourceStates GetResourceState() const override
  {
    return resourceState;
  }

  virtual UINT GetNumSubresources() const override
  {
    return (textureDesc.depth * textureDesc.numMips);
  }

  const char* GetName() const
  {
    return name;
  }

  const TextureDesc& GetTextureDesc() const
  {
    return textureDesc;
  }

private:
  DescHandle CreateSrv(UINT backBufferIndex, int mipIndex=-1) const;

  DescHandle CreateUav(UINT backBufferIndex, UINT mipIndex=0) const;

  ComPtr<ID3D12Resource> texture;
  ComPtr<ID3D12Resource> uploadHeap;
  resourceStates resourceState;

  char name[DEMO_MAX_FILENAME];
  TextureDesc textureDesc;

};

#endif

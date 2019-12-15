#ifndef DX12_DEPTH_STENCIL_TARGET_H
#define DX12_DEPTH_STENCIL_TARGET_H

#include <DX12_Texture.h>

enum depthStencilViewTypes
{
  DEFAULT_DSV_TYPE=0,
  READ_ONLY_DSV_TYPE,
  NUM_DSV_TYPES
};

class DX12_CmdList;

// DX12_DepthStencilTarget
//
class DX12_DepthStencilTarget
{
public:
  DX12_DepthStencilTarget() :
    texture(nullptr),
    clearMask(0)
  {
    memset(dsvDescHandles, 0, sizeof(DescHandle) * NUM_DSV_TYPES);
  }

  ~DX12_DepthStencilTarget()
  {
    Release();
  }

  void Release();

  bool Create(const TextureDesc &desc, const char *name);

  void Clear(ID3D12GraphicsCommandList *cmdList) const;

  DX12_Texture* GetTexture() const
  {
    return texture;
  }

  DescHandle GetDsv(depthStencilViewTypes dsvType) const
  {
    return dsvDescHandles[dsvType];
  }

  DescHandle CreatePerArraySliceDsv(UINT arraySlice, depthStencilViewTypes dsvType) const;

private:
  DX12_Texture *texture;
  DescHandle dsvDescHandles[NUM_DSV_TYPES];
  UINT clearMask;

};

#endif
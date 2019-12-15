#ifndef DX12_RENDER_TARGET_H
#define DX12_RENDER_TARGET_H

#include <DX12_Texture.h>

enum renderTargetViewTypes
{
  DEFAULT_RTV_TYPE=0,
  SRGB_RTV_TYPE
};

class DX12_CmdList;
   
// DX12_RenderTarget
//
class DX12_RenderTarget
{
public:
	DX12_RenderTarget():
    textures(nullptr),
    numTextures(0),
    rtvDescHandles(nullptr)
	{
	}

	~DX12_RenderTarget()
	{
		Release();
	}

	void Release();

	bool Create(const TextureDesc &desc, const char *name);

	void Clear(ID3D12GraphicsCommandList *cmdList) const;

  DX12_Texture* GetTexture() const;

  DescHandle GetRtv(renderTargetViewTypes rtvType=DEFAULT_RTV_TYPE) const;

  DescHandle CreatePerArraySliceRtv(UINT arraySlice, renderTargetViewTypes rtvType=DEFAULT_RTV_TYPE) const;

private:	
  DX12_Texture *textures; 
  UINT numTextures;
  DescHandle *rtvDescHandles;

};

#endif
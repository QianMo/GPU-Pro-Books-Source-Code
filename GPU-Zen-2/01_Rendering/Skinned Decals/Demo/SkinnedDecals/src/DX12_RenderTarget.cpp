#include <stdafx.h>
#include <Demo.h>
#include <render_states.h>
#include <DX12_RenderTarget.h>

void DX12_RenderTarget::Release()
{
  SAFE_DELETE_ARRAY(textures);
  SAFE_DELETE_ARRAY(rtvDescHandles);
}

bool DX12_RenderTarget::Create(const TextureDesc &desc, const char *name)
{
  const bool backBuffer = (desc.flags & BACK_BUFFER_TEXTURE_FLAG);

  numTextures = backBuffer ? NUM_BACKBUFFERS : 1;
  textures = new DX12_Texture[numTextures];
  if(!textures)
    return false;

  for(UINT i=0; i<numTextures; i++)
  {
    if(!textures[i].Create(desc, name, nullptr, i))
      return false;
  }

  UINT numHandles = (desc.flags & SRGB_WRITE_TEXTURE_FLAG) ? (numTextures*2) : numTextures;
  rtvDescHandles = new DescHandle[numHandles];
  if(!rtvDescHandles)
    return false;

  renderFormats rtvFormat = (desc.flags & SRGB_WRITE_TEXTURE_FLAG) ? RenderFormat::ConvertFromSrgbFormat(desc.format) : desc.format;
  if(rtvFormat == NONE_RENDER_FORMAT)
    return false;
  D3D12_RENDER_TARGET_VIEW_DESC rtvDesc = {};
  rtvDesc.Format = RenderFormat::GetDx12RenderFormat(rtvFormat).rtvFormat;
  if(rtvDesc.Format == DXGI_FORMAT_UNKNOWN)
    return false;
  switch(desc.target)
  {
  case TEXTURE_2D:
  case TEXTURE_CUBE:
    rtvDesc.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
    break;

  case TEXTURE_2D_ARRAY:
    rtvDesc.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2DARRAY;
    rtvDesc.Texture2DArray.ArraySize = desc.depth;
    break;

  case TEXTURE_CUBE_ARRAY:
    rtvDesc.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2DARRAY;
    rtvDesc.Texture2DArray.ArraySize = desc.depth / 6;
    break;

  case TEXTURE_3D:
    rtvDesc.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE3D;
    rtvDesc.Texture3D.WSize = -1;
    break;
  }
  for(UINT i=0; i<numTextures; i++)
  {
    Demo::renderer->GetRtvHeap()->AddDescHandle(rtvDescHandles[i]);
    Demo::renderer->GetDevice()->CreateRenderTargetView(textures[i].GetResource(), &rtvDesc, rtvDescHandles[i].cpuDescHandle);  
  }

  if(desc.flags & SRGB_WRITE_TEXTURE_FLAG)
  {
    rtvFormat = RenderFormat::ConvertToSrgbFormat(rtvFormat);
    if(rtvFormat == NONE_RENDER_FORMAT)
      return false;
    rtvDesc.Format = RenderFormat::GetDx12RenderFormat(rtvFormat).rtvFormat;
    if(rtvDesc.Format == DXGI_FORMAT_UNKNOWN)
      return false;
    for(UINT i=0; i<numTextures; i++)
    {
      Demo::renderer->GetRtvHeap()->AddDescHandle(rtvDescHandles[numTextures+i]);
      Demo::renderer->GetDevice()->CreateRenderTargetView(textures[i].GetResource(), &rtvDesc, rtvDescHandles[numTextures+i].cpuDescHandle);  
    }
  }

	return true;
}

void DX12_RenderTarget::Clear(ID3D12GraphicsCommandList *cmdList) const
{
  assert(cmdList != nullptr);
  const UINT index = (numTextures > 1) ? Demo::renderer->GetBackBufferIndex() : 0;
  cmdList->ClearRenderTargetView(rtvDescHandles[index].cpuDescHandle, textures[0].GetTextureDesc().clearColor, 0, nullptr);
}

DX12_Texture* DX12_RenderTarget::GetTexture() const
{
  UINT index = (numTextures > 1) ? Demo::renderer->GetBackBufferIndex() : 0;
  return &textures[index];
}

DescHandle DX12_RenderTarget::GetRtv(renderTargetViewTypes rtvType) const
{
  UINT index = (numTextures > 1) ? Demo::renderer->GetBackBufferIndex() : 0;
  if(rtvType == SRGB_RTV_TYPE)
  {
    assert(textures[0].GetTextureDesc().flags & SRGB_WRITE_TEXTURE_FLAG);
    index += numTextures;
  }
  return rtvDescHandles[index];
}

DescHandle DX12_RenderTarget::CreatePerArraySliceRtv(UINT arraySlice, renderTargetViewTypes rtvType) const
{
  const TextureDesc &textureDesc = textures[0].GetTextureDesc();
  assert(((textureDesc.target == TEXTURE_2D_ARRAY) || (textureDesc.target == TEXTURE_CUBE_ARRAY)) && 
         ((textureDesc.flags & BACK_BUFFER_TEXTURE_FLAG) == 0) && (arraySlice < textureDesc.depth));

  renderFormats rtvFormat;
  if(rtvType == DEFAULT_RTV_TYPE)
    rtvFormat = (textureDesc.flags & SRGB_WRITE_TEXTURE_FLAG) ? RenderFormat::ConvertFromSrgbFormat(textureDesc.format) : textureDesc.format;
  else
    rtvFormat = (textureDesc.flags & SRGB_WRITE_TEXTURE_FLAG) ? RenderFormat::ConvertToSrgbFormat(textureDesc.format) : textureDesc.format;
  assert(rtvFormat != NONE_RENDER_FORMAT);
  
  D3D12_RENDER_TARGET_VIEW_DESC rtvDesc = {};
  rtvDesc.Format = RenderFormat::GetDx12RenderFormat(rtvFormat).rtvFormat;
  assert(rtvDesc.Format != DXGI_FORMAT_UNKNOWN);
  rtvDesc.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2DARRAY;
  rtvDesc.Texture2DArray.FirstArraySlice = arraySlice;
  rtvDesc.Texture2DArray.ArraySize = 1;
  
  DescHandle rtvDescHandle;
  Demo::renderer->GetRtvHeap()->AddDescHandle(rtvDescHandle);
  Demo::renderer->GetDevice()->CreateRenderTargetView(textures[0].GetResource(), &rtvDesc, rtvDescHandle.cpuDescHandle);
  return rtvDescHandle;
}

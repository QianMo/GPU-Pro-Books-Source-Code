#include <stdafx.h>
#include <Demo.h>
#include <render_states.h>
#include <DX12_DepthStencilTarget.h>

void DX12_DepthStencilTarget::Release()
{
  SAFE_DELETE(texture);
}

bool DX12_DepthStencilTarget::Create(const TextureDesc &desc, const char *name)
{
  clearMask = (desc.format == DEPTH24_STENCIL8_RENDER_FORMAT) ? (D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL) : D3D12_CLEAR_FLAG_DEPTH;

  texture = new DX12_Texture;
  if(!texture)
    return false;
  if(!texture->Create(desc, name))
    return false;

  D3D12_DEPTH_STENCIL_VIEW_DESC dsvDesc = {};
  dsvDesc.Format = RenderFormat::GetDx12RenderFormat(desc.format).rtvFormat;
  if(dsvDesc.Format == DXGI_FORMAT_UNKNOWN)
    return false;
  dsvDesc.Flags = D3D12_DSV_FLAG_NONE;
  switch(desc.target)
  {
  case TEXTURE_2D:
  case TEXTURE_CUBE:
    dsvDesc.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
    break;

  case TEXTURE_2D_ARRAY:
    dsvDesc.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2DARRAY;
    dsvDesc.Texture2DArray.ArraySize = desc.depth;
    break;

  case TEXTURE_CUBE_ARRAY:
    dsvDesc.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2DARRAY;
    dsvDesc.Texture2DArray.ArraySize = desc.depth / 6;
    break;

  case TEXTURE_3D:
    assert(false && "Texture 3D depth stencil view not support!");
    break;
  }
  Demo::renderer->GetDsvHeap()->AddDescHandle(dsvDescHandles[DEFAULT_DSV_TYPE]);
  Demo::renderer->GetDevice()->CreateDepthStencilView(texture->GetResource(), &dsvDesc, dsvDescHandles[DEFAULT_DSV_TYPE].cpuDescHandle);

  dsvDesc.Flags = (desc.format == DEPTH24_STENCIL8_RENDER_FORMAT) ? (D3D12_DSV_FLAG_READ_ONLY_DEPTH | D3D12_DSV_FLAG_READ_ONLY_STENCIL) : D3D12_DSV_FLAG_READ_ONLY_DEPTH;
  Demo::renderer->GetDsvHeap()->AddDescHandle(dsvDescHandles[READ_ONLY_DSV_TYPE]);
  Demo::renderer->GetDevice()->CreateDepthStencilView(texture->GetResource(), &dsvDesc, dsvDescHandles[READ_ONLY_DSV_TYPE].cpuDescHandle);

  return true;
}

void DX12_DepthStencilTarget::Clear(ID3D12GraphicsCommandList *cmdList) const
{
  assert(cmdList != nullptr);
  cmdList->ClearDepthStencilView(dsvDescHandles[DEFAULT_DSV_TYPE].cpuDescHandle, (D3D12_CLEAR_FLAGS)clearMask, 
    texture->GetTextureDesc().clearDepth, texture->GetTextureDesc().clearStencil, 0, nullptr);
}

DescHandle DX12_DepthStencilTarget::CreatePerArraySliceDsv(UINT arraySlice, depthStencilViewTypes dsvType) const
{
  const TextureDesc &textureDesc = texture->GetTextureDesc();
  assert(((textureDesc.target == TEXTURE_2D_ARRAY) || (textureDesc.target == TEXTURE_CUBE_ARRAY)) &&
         (arraySlice < textureDesc.depth));

  D3D12_DEPTH_STENCIL_VIEW_DESC dsvDesc = {};
  dsvDesc.Format = RenderFormat::GetDx12RenderFormat(textureDesc.format).rtvFormat;
  assert(dsvDesc.Format != DXGI_FORMAT_UNKNOWN);
  if(dsvType == DEFAULT_DSV_TYPE)
    dsvDesc.Flags = D3D12_DSV_FLAG_NONE;
  else
    dsvDesc.Flags = (textureDesc.format == DEPTH24_STENCIL8_RENDER_FORMAT) ? (D3D12_DSV_FLAG_READ_ONLY_DEPTH | D3D12_DSV_FLAG_READ_ONLY_STENCIL) : D3D12_DSV_FLAG_READ_ONLY_DEPTH;
  dsvDesc.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2DARRAY;
  dsvDesc.Texture2DArray.FirstArraySlice = arraySlice;
  dsvDesc.Texture2DArray.ArraySize = 1;
 
  DescHandle dsvDescHandle;
  Demo::renderer->GetDsvHeap()->AddDescHandle(dsvDescHandle);
  Demo::renderer->GetDevice()->CreateDepthStencilView(texture->GetResource(), &dsvDesc, dsvDescHandle.cpuDescHandle);
  return dsvDescHandle;
}



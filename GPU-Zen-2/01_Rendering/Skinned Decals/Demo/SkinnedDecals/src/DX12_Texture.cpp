#include <stdafx.h>
#include <Demo.h>
#include <DX12_Texture.h>

DX12_TextureData::DX12_TextureData(const Image **images, UINT numImages)
{
  assert(images != nullptr);
  resourceData = nullptr;
  const UINT numMips = images[0]->GetNumMipMaps();
  const UINT arraySize = (images[0]->IsCube()) ? 6 : 1;
  const renderFormats texFormat = images[0]->GetFormat();
  numSubresources = arraySize * numMips * numImages;
  assert(numSubresources > 0);
  resourceData = new D3D12_SUBRESOURCE_DATA[numSubresources];
  assert(resourceData != nullptr);

  UINT index = 0;
  for(UINT i=0; i<numImages; i++)
  {
    for(UINT j=0; j<arraySize; j++)
    {
      for(UINT k=0; k<numMips; k++)
      {
        const UINT width = images[0]->GetWidth(k);
        const UINT height = images[0]->GetHeight(k);
        const UINT depth = (images[0]->IsCube()) ? 1 : images[0]->GetDepth(k);
        switch(texFormat)
        {
        case BGR8_RENDER_FORMAT:
        case BGR8_SRGB_RENDER_FORMAT:
        case RGB8_RENDER_FORMAT:
        {
          const unsigned char *src = images[i]->GetData(k);
          unsigned char *dest = new unsigned char[width * height * depth * 4];
          ExpandData(src, dest, width, height, depth, j, 1);
          resourceData[index].pData = dest;
          resourceData[index].RowPitch = width * 4;
          resourceData[index].SlicePitch = width * height * 4;
          index++;
          expandData = true;
          break;
        }

        case RGB16_RENDER_FORMAT:
        {
          const unsigned char *src = images[i]->GetData(k);
          unsigned char *dest = new unsigned char[width * height * depth * 8];
          ExpandData(src, dest, width, height, depth, j, 2);
          resourceData[index].pData = dest;
          resourceData[index].RowPitch = width * 8;
          resourceData[index].SlicePitch = width * height * 8;
          index++;
          expandData = true;
          break;
        }

        case RGB16F_RENDER_FORMAT:
        {
          const float *src = (float*)images[i]->GetData(k);
          float *dest = new float[width * height * depth * 8];
          ExpandData(src, dest, width, height, depth, j, 2);
          resourceData[index].pData = dest;
          resourceData[index].RowPitch = width * 8;
          resourceData[index].SlicePitch = width * height * 8;
          index++;
          expandData = true;
          break;
        }

        default:
          const UINT faceSize = images[i]->GetSize(k, 1) / 6;
          resourceData[index].pData = images[i]->GetData(k) + (j * faceSize);
          resourceData[index].RowPitch = images[i]->GetPitch(k);
          resourceData[index].SlicePitch = faceSize;
          index++;
          expandData = false;
          break;
        }
      }
    }
  }
}

DX12_TextureData::~DX12_TextureData()
{
  if(expandData)
  {
    for(UINT i=0; i<numSubresources; i++)
      SAFE_DELETE_ARRAY(resourceData[i].pData);
  }
  SAFE_DELETE_ARRAY(resourceData);
}


bool DX12_Texture::Create(const TextureDesc &desc, const char *name, const DX12_TextureData *initData, UINT backBufferIndex)
{
  strcpy(this->name, name);
  textureDesc = desc;

  const bool backBuffer = (desc.flags & BACK_BUFFER_TEXTURE_FLAG);
  const bool depthStencil = (desc.flags & DEPTH_STENCIL_TARGET_TEXTURE_FLAG);

  if(backBuffer && (!depthStencil))
  {
    if(FAILED(Demo::renderer->GetSwapChain()->GetBuffer(backBufferIndex, IID_PPV_ARGS(&texture))))
    {
      return false;
    }

#ifdef _DEBUG
    wchar_t wcharName[DEMO_MAX_FILENAME];
    swprintf(wcharName, DEMO_MAX_FILENAME - 1, L"Texture: %hs [%i]", name, backBufferIndex);
    texture->SetName(wcharName);
#endif
  }
  else
  {
    D3D12_RESOURCE_DESC resourceDesc;
    resourceDesc.Dimension = (desc.target == TEXTURE_3D) ? D3D12_RESOURCE_DIMENSION_TEXTURE3D : D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    resourceDesc.Alignment = 0;
    resourceDesc.Width = desc.width;
    resourceDesc.Height = desc.height;
    resourceDesc.DepthOrArraySize = desc.depth;
    resourceDesc.MipLevels = desc.numMips;
    resourceDesc.Format = RenderFormat::GetDx12RenderFormat(desc.format).resourceFormat;
    if(resourceDesc.Format == DXGI_FORMAT_UNKNOWN)
      return false;
    resourceDesc.SampleDesc.Count = 1;
    resourceDesc.SampleDesc.Quality = 0;
    resourceDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    resourceDesc.Flags = D3D12_RESOURCE_FLAG_NONE;
    if(desc.flags & RENDER_TARGET_TEXTURE_FLAG)
      resourceDesc.Flags |= D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    else if(desc.flags & DEPTH_STENCIL_TARGET_TEXTURE_FLAG)
      resourceDesc.Flags |= D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
    if(desc.flags & UAV_TEXTURE_FLAG)
      resourceDesc.Flags |= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

    if((desc.flags & RENDER_TARGET_TEXTURE_FLAG) || (desc.flags & DEPTH_STENCIL_TARGET_TEXTURE_FLAG))
    {
      resourceState = desc.initResourceState;
      D3D12_CLEAR_VALUE optimizedClearValue = {};
      optimizedClearValue.Format = RenderFormat::GetDx12RenderFormat(desc.format).rtvFormat;
      if(desc.flags & RENDER_TARGET_TEXTURE_FLAG)
      {
        memcpy(optimizedClearValue.Color, desc.clearColor, sizeof(Color));
      }
      else
      {
        optimizedClearValue.DepthStencil.Depth = desc.clearDepth;
        optimizedClearValue.DepthStencil.Stencil = desc.clearStencil;
      }
      if(FAILED(Demo::renderer->GetDevice()->CreateCommittedResource(&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT), D3D12_HEAP_FLAG_NONE, &resourceDesc,
         static_cast<D3D12_RESOURCE_STATES>(resourceState), &optimizedClearValue, IID_PPV_ARGS(&texture))))
      {
        return false;
      }
    }
    else
    {
      resourceState = (desc.flags & STATIC_TEXTURE_FLAG) ? COPY_DEST_RESOURCE_STATE : desc.initResourceState;
      if(FAILED(Demo::renderer->GetDevice()->CreateCommittedResource(&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT), D3D12_HEAP_FLAG_NONE, &resourceDesc,
         static_cast<D3D12_RESOURCE_STATES>(resourceState), nullptr, IID_PPV_ARGS(&texture))))
      {
        return false;
      }
    }

#ifdef _DEBUG
    wchar_t wcharName[DEMO_MAX_FILENAME];
    swprintf(wcharName, DEMO_MAX_FILENAME - 1, L"Texture: %hs", name);
    texture->SetName(wcharName);
#endif

    if(desc.flags & STATIC_TEXTURE_FLAG)
    {
      const UINT subresourceCount = GetNumSubresources();
      const UINT64 uploadBufferSize = GetRequiredIntermediateSize(texture.Get(), 0, subresourceCount);
      if(FAILED(Demo::renderer->GetDevice()->CreateCommittedResource(&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD), D3D12_HEAP_FLAG_NONE,
         &CD3DX12_RESOURCE_DESC::Buffer(uploadBufferSize), D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&uploadHeap))))
      {
        return false;
      }
      if(initData)
      {
        // copy to upload heap
        DX12_UploadHelper uploadHelper(this);
        if(!uploadHelper.CopySubresources(*initData, subresourceCount))
        {
          return false;
        }

        // upload to GPU memory
        UploadCmd uploadCmd;
        uploadCmd.resource = this;
        Demo::renderer->GetCmdList(DEFAULT_CMD_LIST_ID)->AddGpuCmd(uploadCmd, INIT_GPU_CMD_ORDER);

        // transition resource state
        ResourceBarrier barriers[1];
        barriers[0].barrierType = TRANSITION_RESOURCE_BARRIER_TYPE;
        barriers[0].transition.resource = this;
        barriers[0].transition.newResourceState = desc.initResourceState;

        ResourceBarrierCmd barrierCmd;
        barrierCmd.barriers = barriers; 
        barrierCmd.numBarriers = _countof(barriers);
        Demo::renderer->GetCmdList(DEFAULT_CMD_LIST_ID)->AddGpuCmd(barrierCmd, INIT_GPU_CMD_ORDER);
      }
    }
  }

  return true;
}

bool DX12_Texture::Create(const Image **images, UINT numImages, textureFlags flags)
{
  if((!images) || (numImages == 0) || ((flags & STATIC_TEXTURE_FLAG) == 0))
  {
    return false;
  }

  TextureDesc desc;
  desc.width = images[0]->GetWidth();
  desc.height = images[0]->GetHeight();
  desc.depth = images[0]->IsCube() ? (6 * numImages) : images[0]->GetDepth();
  desc.format = images[0]->GetFormat();
  desc.numMips = images[0]->GetNumMipMaps();
  desc.flags = flags;
  desc.initResourceState = (NON_PIXEL_SHADER_RESOURCE_RESOURCE_STATE | PIXEL_SHADER_RESOURCE_RESOURCE_STATE);

  if(images[0]->IsCube())
  {
    desc.target = (numImages == 1) ? TEXTURE_CUBE : TEXTURE_CUBE_ARRAY;
  }
  else if(images[0]->Is3D())
  {
    assert(numImages == 1);
    desc.target = TEXTURE_3D;
  }
  else if(images[0]->Is2D())
  {
    desc.target = (numImages == 1) ? TEXTURE_2D : TEXTURE_2D_ARRAY;
  }
  else
  {
    return false;
  }

  DX12_TextureData initData(images, numImages);
  if(!Create(desc, images[0]->GetName(), &initData))
    return false;

  return true;
}

bool DX12_Texture::LoadFromFile(const char *fileName, textureFlags flags)
{
  Image image;
  if(!image.Load(fileName))
    return false;
  const Image *pImage = &image;
  return Create(&pImage, 1, flags);
}

DescHandle DX12_Texture::CreateSrv(UINT backBufferIndex, int mipIndex) const
{
  renderFormats srvFormat = (textureDesc.flags & SRGB_READ_TEXTURE_FLAG) ? RenderFormat::ConvertToSrgbFormat(textureDesc.format) : textureDesc.format;
  assert((backBufferIndex < NUM_BACKBUFFERS) && (mipIndex > -2) && (mipIndex < int(textureDesc.numMips)) && (srvFormat != NONE_RENDER_FORMAT));

  D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
  srvDesc.Format = RenderFormat::GetDx12RenderFormat(srvFormat).srvFormat;
  srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
  switch(textureDesc.target)
  {
  case TEXTURE_2D:
    srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srvDesc.Texture2D.MipLevels = mipIndex;
    break;

  case TEXTURE_2D_ARRAY:
    srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
    srvDesc.Texture2DArray.MipLevels = mipIndex;
    srvDesc.Texture2DArray.ArraySize = textureDesc.depth;
    break;

  case TEXTURE_CUBE:
    srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURECUBE;
    srvDesc.TextureCube.MipLevels = mipIndex;
    break;

  case TEXTURE_CUBE_ARRAY:
    srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURECUBEARRAY;
    srvDesc.TextureCubeArray.MipLevels = mipIndex;
    srvDesc.TextureCubeArray.NumCubes = textureDesc.depth / 6;
    break;

  case TEXTURE_3D:
    srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE3D;
    srvDesc.Texture3D.MipLevels = mipIndex;
    break;
  }

  DescHandle descHandle;
  Demo::renderer->GetCbvSrvUavHeap(backBufferIndex)->AddDescHandle(descHandle);
  Demo::renderer->GetDevice()->CreateShaderResourceView(texture.Get(), &srvDesc, descHandle.cpuDescHandle);

  return descHandle;
}

DescHandle DX12_Texture::CreateUav(UINT backBufferIndex, UINT mipIndex) const
{
  assert((backBufferIndex < NUM_BACKBUFFERS) && (textureDesc.flags | UAV_TEXTURE_FLAG) && (mipIndex < textureDesc.numMips));

  D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
  uavDesc.Format = RenderFormat::GetDx12RenderFormat(textureDesc.format).srvFormat;
  switch(textureDesc.target)
  {
  case TEXTURE_2D:
    uavDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    uavDesc.Texture2D.MipSlice = mipIndex;
    break;

  case TEXTURE_2D_ARRAY:
    uavDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
    uavDesc.Texture2DArray.MipSlice = mipIndex;
    uavDesc.Texture2DArray.ArraySize = textureDesc.depth;
    break;
    
  case TEXTURE_3D:
    uavDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE3D;
    uavDesc.Texture3D.MipSlice = mipIndex;
    uavDesc.Texture3D.WSize = textureDesc.depth;
    break;

  default:
    assert(false && "Unsupported UAV target!");
    break;
  }

  DescHandle descHandle;
  Demo::renderer->GetCbvSrvUavHeap(backBufferIndex)->AddDescHandle(descHandle);
  Demo::renderer->GetDevice()->CreateUnorderedAccessView(texture.Get(), nullptr, &uavDesc, descHandle.cpuDescHandle);
  
  return descHandle;
}

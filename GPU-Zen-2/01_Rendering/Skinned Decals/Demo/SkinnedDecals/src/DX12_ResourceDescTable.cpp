#include <stdafx.h>
#include <Demo.h>
#include <DX12_ResourceDescTable.h>

void DX12_ResourceDescTable::AddTextureSrv(DX12_Texture *texture, int mipIndex)
{
  assert(texture != nullptr);
  for(UINT i=0; i<NUM_BACKBUFFERS; i++)
  {
    DescHandle descHandle = texture->CreateSrv(i, mipIndex);
    if(descHandles[i].cpuDescHandle.ptr == 0)
      descHandles[i] = descHandle;
  }
}

void DX12_ResourceDescTable::AddTextureUav(DX12_Texture *texture, UINT mipIndex)
{
  assert(texture != nullptr);
  for(UINT i=0; i<NUM_BACKBUFFERS; i++)
  {
    DescHandle descHandle = texture->CreateUav(i, mipIndex);
    if(descHandles[i].cpuDescHandle.ptr == 0)
      descHandles[i] = descHandle;
  }
}

void DX12_ResourceDescTable::AddBufferSrv(DX12_Buffer *buffer)
{
  assert(buffer != nullptr);
  for(UINT i=0; i<NUM_BACKBUFFERS; i++)
  {
    DescHandle descHandle = buffer->CreateSrv(i);
    if(descHandles[i].cpuDescHandle.ptr == 0)
      descHandles[i] = descHandle;
  }
}

void DX12_ResourceDescTable::AddBufferUav(DX12_Buffer *buffer)
{
  assert(buffer != nullptr);
  for(UINT i=0; i<NUM_BACKBUFFERS; i++)
  {
    DescHandle descHandle = buffer->CreateUav(i);
    if(descHandles[i].cpuDescHandle.ptr == 0)
      descHandles[i] = descHandle;
  }
}

DescHandle DX12_ResourceDescTable::GetBaseDescHandle() const
{
  return descHandles[Demo::renderer->GetBackBufferIndex()];
}



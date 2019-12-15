#ifndef DX12_RESOURCE_DESC_TABLE_H
#define DX12_RESOURCE_DESC_TABLE_H

#include <render_states.h>

class DX12_Texture;
class DX12_Buffer;

// DX12_ResourceDescTable
//
class DX12_ResourceDescTable
{
public:
  DX12_ResourceDescTable()
  {
     memset(descHandles, 0, sizeof(DescHandle) * NUM_BACKBUFFERS);
  }

  void AddTextureSrv(DX12_Texture *texture, int mipIndex=-1);

  void AddTextureUav(DX12_Texture *texture, UINT mipIndex=0);

  void AddBufferSrv(DX12_Buffer *buffer);

  void AddBufferUav(DX12_Buffer *buffer);

  DescHandle GetBaseDescHandle() const;

private:
  DescHandle descHandles[NUM_BACKBUFFERS];

};

#endif

#ifndef DX12_SAMPLER_DESC_TABLE_H
#define DX12_SAMPLER_DESC_TABLE_H

#include <DX12_RootSignature.h>

// DX12_SamplerDescTable
//
class DX12_SamplerDescTable
{
public:
  DX12_SamplerDescTable()
  {
    memset(&descHandle, 0, sizeof(DescHandle));
  }
   
  void AddSampler(const SamplerDesc &desc);

  DescHandle GetBaseDescHandle() const
  {
    return descHandle;
  }

private:
  DescHandle descHandle;

};

#endif

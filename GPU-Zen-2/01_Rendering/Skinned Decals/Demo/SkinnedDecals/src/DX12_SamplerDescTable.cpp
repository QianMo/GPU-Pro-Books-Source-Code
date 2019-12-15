#include <stdafx.h>
#include <Demo.h>
#include <DX12_SamplerDescTable.h>

void DX12_SamplerDescTable::AddSampler(const SamplerDesc &desc)
{
  DescHandle samplerDescHandle;
  Demo::renderer->GetSamplerHeap()->AddDescHandle(samplerDescHandle);
  Demo::renderer->GetDevice()->CreateSampler((D3D12_SAMPLER_DESC*)&desc, samplerDescHandle.cpuDescHandle);
  if(descHandle.cpuDescHandle.ptr == 0)
    descHandle = samplerDescHandle;
}
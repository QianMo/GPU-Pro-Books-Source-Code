#ifndef DX12_DESC_HEAP_H
#define DX12_DESC_HEAP_H

#include <render_states.h>

enum descriptorHeapTypes
{
  CBV_SRV_UAV_DESCRIPTOR_HEAP_TYPE = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
  SAMPLER_DESCRIPTOR_HEAP_TYPE     = D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER,
  RTV_DESCRIPTOR_HEAP_TYPE         = D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
  DSV_DESCRIPTOR_HEAP_TYPE         = D3D12_DESCRIPTOR_HEAP_TYPE_DSV
};

struct DescHeapDesc
{
  DescHeapDesc() :
    descHeapType(CBV_SRV_UAV_DESCRIPTOR_HEAP_TYPE),
    maxNumDescs(0),
    shaderVisible(false)
  {
  }

  descriptorHeapTypes descHeapType;
  UINT maxNumDescs;
  bool shaderVisible;
};

// DX12_DescHeap
//
class DX12_DescHeap
{
public:
  DX12_DescHeap() :
    descSize(0),
    currentNumDescs(0)
  {
  }

  bool Create(const DescHeapDesc &desc, const char *name);

  void AddDescHandle(DescHandle &descHandle);

  ID3D12DescriptorHeap* GetDescHeap() const
  {
    return descHeap.Get();
  }

  const DescHeapDesc& GetDesc() const
  {
    return desc;
  }

private:
  ComPtr<ID3D12DescriptorHeap> descHeap;
  UINT descSize;
  UINT currentNumDescs;
  DescHeapDesc desc;

};

#endif

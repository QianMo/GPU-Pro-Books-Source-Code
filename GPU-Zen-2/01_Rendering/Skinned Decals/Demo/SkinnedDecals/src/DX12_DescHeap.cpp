#include <stdafx.h>
#include <Demo.h>
#include <DX12_DescHeap.h>

bool DX12_DescHeap::Create(const DescHeapDesc &desc, const char *name)
{
  assert(name != nullptr);
  this->desc = desc;

  D3D12_DESCRIPTOR_HEAP_DESC heapDesc = {};
  heapDesc.NumDescriptors = desc.maxNumDescs;
  heapDesc.Type = static_cast<D3D12_DESCRIPTOR_HEAP_TYPE>(desc.descHeapType);
  heapDesc.Flags = desc.shaderVisible ? D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE : D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
  if(FAILED(Demo::renderer->GetDevice()->CreateDescriptorHeap(&heapDesc, IID_PPV_ARGS(&descHeap))))
  {
    return false;
  }

#ifdef _DEBUG
  wchar_t wcharName[DEMO_MAX_STRING];
  swprintf(wcharName, DEMO_MAX_STRING - 1, L"Descriptor heap: %hs", name);
  descHeap->SetName(wcharName);
#endif

  descSize = Demo::renderer->GetDevice()->GetDescriptorHandleIncrementSize(heapDesc.Type);

  return true;
}

void DX12_DescHeap::AddDescHandle(DescHandle &descHandle)
{
  assert(currentNumDescs < desc.maxNumDescs);

  CD3DX12_CPU_DESCRIPTOR_HANDLE tmpCpuDescHandle(descHeap->GetCPUDescriptorHandleForHeapStart());
  tmpCpuDescHandle.Offset(currentNumDescs, descSize);
  descHandle.cpuDescHandle = tmpCpuDescHandle;

  if(desc.shaderVisible)
  {
    CD3DX12_GPU_DESCRIPTOR_HANDLE tmpGpuDescHandle(descHeap->GetGPUDescriptorHandleForHeapStart());
    tmpGpuDescHandle.Offset(currentNumDescs, descSize);
    descHandle.gpuDescHandle = tmpGpuDescHandle;
  }
  else
  {
    descHandle.gpuDescHandle.ptr = 0;
  }

  currentNumDescs++;
}
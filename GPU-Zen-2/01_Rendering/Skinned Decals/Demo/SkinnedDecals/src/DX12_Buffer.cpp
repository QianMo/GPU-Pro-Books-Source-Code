#include <stdafx.h>
#include <Demo.h>
#include <DX12_Buffer.h>

void DX12_Buffer::Release()
{
  if(mappedBuffer)
    buffer->Unmap(0, nullptr);
}

bool DX12_Buffer::Create(const BufferDesc &desc, const char *name)
{
  if(desc.bufferType == NONE_BUFFER)
    return false;

  bufferDesc = desc;
  
  bufferSize = desc.elementSize * desc.numElements;
  if(desc.bufferType == CONSTANT_BUFFER)
  {
     bufferSize = (bufferSize + (D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT - 1)) & ~(D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT - 1);
  }

  if(desc.flags & CPU_WRITE_BUFFER_FLAG)
  {
    if(desc.flags & DYNAMIC_BUFFER_FLAG)
    {
      resourceState = GENERIC_READ_RESOURCE_STATE;
      if(FAILED(Demo::renderer->GetDevice()->CreateCommittedResource(&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD), D3D12_HEAP_FLAG_NONE,
         &CD3DX12_RESOURCE_DESC::Buffer(bufferSize * NUM_BACKBUFFERS), static_cast<D3D12_RESOURCE_STATES>(resourceState), nullptr, IID_PPV_ARGS(&buffer))))
      {
        return false;
      }
      if(FAILED(buffer->Map(0, nullptr, reinterpret_cast<void**>(&mappedBuffer))))
      {
        return false;
      }
    }
    else
    {
      if(FAILED(Demo::renderer->GetDevice()->CreateCommittedResource(&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD), D3D12_HEAP_FLAG_NONE,
        &CD3DX12_RESOURCE_DESC::Buffer(bufferSize), D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&uploadHeap))))
      {
        return false;
      }
      resourceState = COPY_DEST_RESOURCE_STATE;
      if(FAILED(Demo::renderer->GetDevice()->CreateCommittedResource(&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT), D3D12_HEAP_FLAG_NONE,
         &CD3DX12_RESOURCE_DESC::Buffer(bufferSize), static_cast<D3D12_RESOURCE_STATES>(resourceState), nullptr, IID_PPV_ARGS(&buffer))))
      {
        return false;
      }
    }
  }
  else if(desc.flags & CPU_READ_BUFFER_FLAG)
  {
    resourceState = COPY_DEST_RESOURCE_STATE;
    if(FAILED(Demo::renderer->GetDevice()->CreateCommittedResource(&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK), D3D12_HEAP_FLAG_NONE,
      &CD3DX12_RESOURCE_DESC::Buffer(bufferSize), static_cast<D3D12_RESOURCE_STATES>(resourceState), nullptr, IID_PPV_ARGS(&buffer))))
    {
      return false;
    }
  }
  else
  {
    resourceState = COMMON_RESOURCE_STATE;
    if(FAILED(Demo::renderer->GetDevice()->CreateCommittedResource(&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT), D3D12_HEAP_FLAG_NONE,
       &CD3DX12_RESOURCE_DESC::Buffer(bufferSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS), static_cast<D3D12_RESOURCE_STATES>(resourceState),
       nullptr, IID_PPV_ARGS(&buffer))))
    {
      return false;
    }
  }

#ifdef _DEBUG
  wchar_t wcharName[DEMO_MAX_STRING];
  switch(desc.bufferType)
  {
  case VERTEX_BUFFER:
    swprintf(wcharName, DEMO_MAX_STRING - 1, L"Vertex buffer: %hs", name);
    break;
  case INDEX_BUFFER:
    swprintf(wcharName, DEMO_MAX_STRING - 1, L"Index buffer: %hs", name);
    break;
  case STRUCTURED_BUFFER:
    swprintf(wcharName, DEMO_MAX_STRING - 1, L"Structured buffer: %hs", name);
    break;
  case CONSTANT_BUFFER:
    swprintf(wcharName, DEMO_MAX_STRING - 1, L"Constant buffer: %hs", name);
    break;
  }
  buffer->SetName(wcharName);
#endif

  return true;
}

DescHandle DX12_Buffer::CreateSrv(UINT backBufferIndex) const
{
  assert(backBufferIndex < NUM_BACKBUFFERS);
  UINT index = ((bufferDesc.flags & CPU_WRITE_BUFFER_FLAG) && (bufferDesc.flags & DYNAMIC_BUFFER_FLAG)) ? backBufferIndex : 0;

  D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
  srvDesc.Format = DXGI_FORMAT_UNKNOWN;
  srvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
  srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
  srvDesc.Buffer.FirstElement = index * bufferDesc.numElements;
  srvDesc.Buffer.NumElements = bufferDesc.numElements;
  srvDesc.Buffer.StructureByteStride = bufferDesc.elementSize;
  srvDesc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;
  DescHandle descHandle;
  Demo::renderer->GetCbvSrvUavHeap(backBufferIndex)->AddDescHandle(descHandle);
  Demo::renderer->GetDevice()->CreateShaderResourceView(buffer.Get(), &srvDesc, descHandle.cpuDescHandle);
  
  return descHandle;
}

DescHandle DX12_Buffer::CreateUav(UINT backBufferIndex) const
{
  assert(backBufferIndex < NUM_BACKBUFFERS);

  D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
  uavDesc.Format = DXGI_FORMAT_UNKNOWN;
  uavDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
  uavDesc.Buffer.FirstElement = 0;
  uavDesc.Buffer.NumElements = bufferDesc.numElements;
  uavDesc.Buffer.StructureByteStride = bufferDesc.elementSize;
  uavDesc.Buffer.CounterOffsetInBytes = 0;
  uavDesc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_NONE;
  DescHandle descHandle;
  Demo::renderer->GetCbvSrvUavHeap(backBufferIndex)->AddDescHandle(descHandle);
  Demo::renderer->GetDevice()->CreateUnorderedAccessView(buffer.Get(), nullptr, &uavDesc, descHandle.cpuDescHandle);

  return descHandle;
}

bool DX12_Buffer::Update(const void *bufferData, UINT elementCount)
{
  if((!(bufferDesc.flags & CPU_WRITE_BUFFER_FLAG)) || (elementCount > bufferDesc.numElements))
    return false;

  if(bufferDesc.flags & DYNAMIC_BUFFER_FLAG)
  {
    const UINT offset = Demo::renderer->GetBackBufferIndex() * bufferSize;
    memcpy(mappedBuffer + offset, bufferData, bufferDesc.elementSize * elementCount);
  }
  else
  {
    // copy to upload heap
    D3D12_SUBRESOURCE_DATA uploadData = {};
    uploadData.pData = bufferData;
    uploadData.RowPitch = bufferSize;
    uploadData.SlicePitch = uploadData.RowPitch;
    DX12_UploadHelper uploadHelper(this);
    if(!uploadHelper.CopySubresources<1>(&uploadData))
      return false;

    // upload to GPU memory
    UploadCmd uploadCmd;
    uploadCmd.resource = this;
    Demo::renderer->GetCmdList(DEFAULT_CMD_LIST_ID)->AddGpuCmd(uploadCmd, INIT_GPU_CMD_ORDER);

    // transition resource state
    ResourceBarrier barriers[1];
    barriers[0].barrierType = TRANSITION_RESOURCE_BARRIER_TYPE;
    barriers[0].transition.resource = this;
    switch(bufferDesc.bufferType)
    {
    case VERTEX_BUFFER:
    case CONSTANT_BUFFER:
      barriers[0].transition.newResourceState = VERTEX_AND_CONSTANT_BUFFER_RESOURCE_STATE;
      break;
    case INDEX_BUFFER:
      barriers[0].transition.newResourceState = INDEX_BUFFER_RESOURCE_STATE;
      break;
    case STRUCTURED_BUFFER:
      barriers[0].transition.newResourceState = COMMON_RESOURCE_STATE;
      break;
    }

    ResourceBarrierCmd barrierCmd;
    barrierCmd.barriers = barriers;
    barrierCmd.numBarriers = _countof(barriers);
    Demo::renderer->GetCmdList(DEFAULT_CMD_LIST_ID)->AddGpuCmd(barrierCmd, INIT_GPU_CMD_ORDER);
  }

  return true;
}

GpuVirtualAddress DX12_Buffer::GetGpuVirtualAddress() const
{
  const UINT offset = ((bufferDesc.flags & CPU_WRITE_BUFFER_FLAG) && (bufferDesc.flags & DYNAMIC_BUFFER_FLAG)) ?
                      (Demo::renderer->GetBackBufferIndex() * bufferSize) : 0;
  return (buffer->GetGPUVirtualAddress() + offset);
}

D3D12_VERTEX_BUFFER_VIEW DX12_Buffer::GetVertexBufferView() const
{
  assert(bufferDesc.bufferType == VERTEX_BUFFER);

  const UINT offset = ((bufferDesc.flags & CPU_WRITE_BUFFER_FLAG) && (bufferDesc.flags & DYNAMIC_BUFFER_FLAG)) ?
                      (Demo::renderer->GetBackBufferIndex() * bufferSize) : 0;

  D3D12_VERTEX_BUFFER_VIEW vertexBufferView;
  vertexBufferView.BufferLocation = buffer->GetGPUVirtualAddress() + offset;
  vertexBufferView.StrideInBytes = bufferDesc.elementSize;
  vertexBufferView.SizeInBytes = bufferSize;

  return vertexBufferView;
}

D3D12_INDEX_BUFFER_VIEW DX12_Buffer::GetIndexBufferView() const
{
  assert(bufferDesc.bufferType == INDEX_BUFFER);

  const UINT offset = ((bufferDesc.flags & CPU_WRITE_BUFFER_FLAG) && (bufferDesc.flags & DYNAMIC_BUFFER_FLAG)) ?
                      (Demo::renderer->GetBackBufferIndex() * bufferSize) : 0;

  D3D12_INDEX_BUFFER_VIEW indexBufferView;
  indexBufferView.BufferLocation = buffer->GetGPUVirtualAddress() + offset;
  indexBufferView.SizeInBytes = bufferSize;
  indexBufferView.Format = RenderFormat::GetDx12RenderFormat(bufferDesc.elementFormat).resourceFormat;

  return indexBufferView;
}

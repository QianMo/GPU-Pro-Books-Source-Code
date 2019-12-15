#include <stdafx.h>
#include <Demo.h>
#include <DX12_TimerQuery.h>

bool DX12_TimerQuery::Create(const char *name)
{
  uint64_t gpuFrequency;
  Demo::renderer->GetCmdQueue()->GetTimestampFrequency(&gpuFrequency);
  gpuTickDelta = 1000.0 / ((double)gpuFrequency);

  D3D12_QUERY_HEAP_DESC queryHeapDesc;
  queryHeapDesc.Count = 2,
  queryHeapDesc.NodeMask = 1;
  queryHeapDesc.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
  if(FAILED(Demo::renderer->GetDevice()->CreateQueryHeap(&queryHeapDesc, IID_PPV_ARGS(&queryHeap))))
  {
    return false;
  }

  D3D12_RESOURCE_DESC bufferDesc;
  bufferDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
  bufferDesc.Alignment = 0;
  bufferDesc.Width = sizeof(uint64_t) * 2;
  bufferDesc.Height = 1;
  bufferDesc.DepthOrArraySize = 1;
  bufferDesc.MipLevels = 1;
  bufferDesc.Format = DXGI_FORMAT_UNKNOWN;
  bufferDesc.SampleDesc.Count = 1;
  bufferDesc.SampleDesc.Quality = 0;
  bufferDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
  bufferDesc.Flags = D3D12_RESOURCE_FLAG_NONE;
  if(FAILED(Demo::renderer->GetDevice()->CreateCommittedResource(&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK), D3D12_HEAP_FLAG_NONE, &bufferDesc,
     D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&readbackBuffer))))
  {
    return false;
  }

#ifdef _DEBUG
  if(name)
  {
    wchar_t wcharName[DEMO_MAX_STRING];
    swprintf(wcharName, DEMO_MAX_STRING - 1, L"Timer query heap: %hs", name);
    queryHeap->SetName(wcharName);
    swprintf(wcharName, DEMO_MAX_STRING - 1, L"Timer query buffer: %hs", name);
    readbackBuffer->SetName(wcharName);
  }
#endif

  return true;
}

void DX12_TimerQuery::BeginQuery(const DX12_CmdList &cmdList)
{
  if(frameIndex < 0)
    frameIndex = Demo::renderer->GetBackBufferIndex();

  if(frameIndex == Demo::renderer->GetBackBufferIndex())
  {
    uint64_t *timeStampBuffer;
    D3D12_RANGE range;
    range.Begin = 0;
    range.End = sizeof(uint64_t) * 2;
    readbackBuffer->Map(0, &range, reinterpret_cast<void**>(&timeStampBuffer));
    uint64_t timeStampStart = timeStampBuffer[0];
    uint64_t timeStampEnd = timeStampBuffer[1];
    D3D12_RANGE emptyRange = {};
    readbackBuffer->Unmap(0, &emptyRange);
    if(timeStampEnd < timeStampStart)
    {
      gpuElapsedTime = 0.0f;
    }
    else
    {
      gpuElapsedTime = static_cast<float>(gpuTickDelta * (timeStampEnd - timeStampStart));
    }

    cmdList.GetCmdList()->EndQuery(queryHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0);
  }
}

void DX12_TimerQuery::EndQuery(const DX12_CmdList &cmdList)
{
  if(frameIndex == Demo::renderer->GetBackBufferIndex())
  {
    cmdList.GetCmdList()->EndQuery(queryHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 1);
    cmdList.GetCmdList()->ResolveQueryData(queryHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, readbackBuffer.Get(), 0);
  }
}


#include <stdafx.h>
#include <Demo.h>
#include <DX12_Fence.h>

void DX12_Fence::Release()
{
  if(fenceEvent)
    CloseHandle(fenceEvent);
}

bool DX12_Fence::Create(const char *name)
{
  if(FAILED(Demo::renderer->GetDevice()->CreateFence(fenceValues[Demo::renderer->GetBackBufferIndex()], D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence))))
  {
    return false;
  }

#ifdef _DEBUG
  wchar_t wcharName[DEMO_MAX_STRING];
  swprintf(wcharName, DEMO_MAX_STRING - 1, L"Fence: %hs", name);
  fence->SetName(wcharName);
#endif

  fenceValues[Demo::renderer->GetBackBufferIndex()]++;

  fenceEvent = CreateEventEx(nullptr, nullptr, 0, EVENT_ALL_ACCESS);
  if(!fenceEvent)
    return false;

  return true;
}

void DX12_Fence::WaitForGpu()
{
  // schedule signal command in queue
  Demo::renderer->GetCmdQueue()->Signal(fence.Get(), fenceValues[Demo::renderer->GetBackBufferIndex()]);

  // wait until fence has been processed
  fence->SetEventOnCompletion(fenceValues[Demo::renderer->GetBackBufferIndex()], fenceEvent);
  WaitForSingleObjectEx(fenceEvent, INFINITE, FALSE);

  // increment fence value for current frame
  fenceValues[Demo::renderer->GetBackBufferIndex()]++;
}

void DX12_Fence::MoveToNextFrame(UINT lastBackBufferIndex, UINT nextBackBufferIndex)
{
  // schedule signal command in queue
  const UINT64 lastFenceValue = fenceValues[lastBackBufferIndex];
  Demo::renderer->GetCmdQueue()->Signal(fence.Get(), lastFenceValue);

  // if next frame is not ready to be rendered yet, wait until it is ready
  if(fence->GetCompletedValue() < fenceValues[nextBackBufferIndex])
  {
    fence->SetEventOnCompletion(fenceValues[nextBackBufferIndex], fenceEvent);
    WaitForSingleObjectEx(fenceEvent, INFINITE, FALSE);
  }

  // set fence value for next frame
  fenceValues[nextBackBufferIndex] = lastFenceValue + 1;
}

#ifndef DX12_FENCE_H
#define DX12_FENCE_H

// DX12_Fence
//
class DX12_Fence
{
public:
  DX12_Fence() :
    fenceEvent(nullptr)
  {
    memset(fenceValues, 0, sizeof(UINT64) * NUM_BACKBUFFERS);
  }

  ~DX12_Fence()
  {
    Release();
  }

  void Release();

  bool Create(const char *name);

  void WaitForGpu();

  void MoveToNextFrame(UINT lastBackBufferIndex, UINT nextBackBufferIndex);

private:
  ComPtr<ID3D12Fence> fence;
  UINT64 fenceValues[NUM_BACKBUFFERS];
  HANDLE fenceEvent;

};

#endif 
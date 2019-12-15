#ifndef DX12_TIMER_QUERY_H
#define DX12_TIMER_QUERY_H

class DX12_CmdList;

// DX12_TimerQuery
//
class DX12_TimerQuery
{
public:
  DX12_TimerQuery() :
    frameIndex(-1),
    gpuTickDelta(0.0),
    gpuElapsedTime(0.0f)
  {
  }

  bool Create(const char *name);

  void BeginQuery(const DX12_CmdList &cmdList);

  void EndQuery(const DX12_CmdList &cmdList);

  float GetGpuElapsedTime() const
  {
    return gpuElapsedTime;
  }

private:
  ComPtr<ID3D12QueryHeap> queryHeap;
  ComPtr<ID3D12Resource> readbackBuffer;
  int frameIndex;
  double gpuTickDelta;
  float gpuElapsedTime;

};

#endif
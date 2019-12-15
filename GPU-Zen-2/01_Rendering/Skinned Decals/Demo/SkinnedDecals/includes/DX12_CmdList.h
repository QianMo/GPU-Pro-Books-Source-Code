#ifndef DX12_CMD_LIST_H
#define DX12_CMD_LIST_H

#include <List.h>
#include <ThreadedTask.h>
#include <GpuCmd.h>

enum cmdListTypes
{
  DIRECT_CMD_LIST_TYPE  = D3D12_COMMAND_LIST_TYPE_DIRECT,
  BUNDLE_CMD_LIST_TYPE  = D3D12_COMMAND_LIST_TYPE_BUNDLE,
  COMPUTE_CMD_LIST_TYPE = D3D12_COMMAND_LIST_TYPE_COMPUTE,
  COPY_CMD_LIST_TYPE    = D3D12_COMMAND_LIST_TYPE_COPY
};

enum cmdListOrders
{
  DEFAULT_CMD_LIST_ORDER=0,
  POSTPROCESS_CMD_LIST_ORDER,
  FINAL_CMD_LIST_ORDER
};

struct CmdListDesc
{
  CmdListDesc() :
    cmdListType(DIRECT_CMD_LIST_TYPE),
    cmdListOrder(DEFAULT_CMD_LIST_ORDER),
    cmdPoolSize(0),
    record(false)
  {
  }

  cmdListTypes cmdListType;
  cmdListOrders cmdListOrder;
  UINT cmdPoolSize;
  bool record;
};

// DX12_CmdList
//
class DX12_CmdList: public ThreadedTask
{
public:
  DX12_CmdList() :
    cmdData(nullptr),
    cmdDataSize(0),
    cmdPosition(0)
  {
  }

  ~DX12_CmdList()
  {
    Release();
  }

  void Release();

  bool Create(const CmdListDesc &desc, const char *name);

  bool Reset();

  void AddGpuCmd(const IGpuCmd &cmd, gpuCmdOrders order, float sortingPriority=0.0f);

  void SortGpuCmds();

  void PatchGpuCmds();

  virtual void Run() override;

  ID3D12GraphicsCommandList* GetCmdList() const
  {
    return cmdList.Get();
  }

  const CmdListDesc& GetDesc() const
  {
    return desc;
  }

private:
  void SetDrawStates(const BaseDrawCmd &cmd);

  void SetDrawShaderParams(const BaseDrawCmd &cmd);

  void SetComputeShaderParams(const BaseComputeCmd &cmd);

  void Upload(const UploadCmd &cmd);

  void Readback(const ReadbackCmd &cmd);

  void SetResourceBarrier(const ResourceBarrierCmd &cmd);

  void ClearRenderTargetView(const ClearRtvCmd &cmd);

  void ClearDepthStencilView(const ClearDsvCmd &cmd);

  void Draw(const DrawCmd &cmd);

  void DrawIndirect(const IndirectDrawCmd &cmd);

  void Dispatch(const ComputeCmd &cmd);

  void DispatchIndirect(const IndirectComputeCmd &cmd);

  void BeginTimerQuery(const BeginTimerQueryCmd &cmd);

  void EndTimerQuery(const EndTimerQueryCmd &cmd);

  void BeginGpuMarker(const BeginGpuMarkerCmd &cmd);

  void EndGpuMarker(const EndGpuMarkerCmd &cmd);

  ComPtr<ID3D12CommandAllocator> cmdAllocators[NUM_BACKBUFFERS];
  ComPtr<ID3D12GraphicsCommandList> cmdList;
  List<SortedGpuCmd> sortedGpuCmds;
  BYTE *cmdData;
  UINT cmdDataSize;
  UINT cmdPosition;
  CmdListDesc desc;
  LastGpuStates lastGpuStates;

};

#endif 
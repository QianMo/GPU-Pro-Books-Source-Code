#ifndef GPU_CMD_H
#define GPU_CMD_H

#include <render_states.h>

#define MAX_MARKER_NAME 64

class DX12_IResource;
class DX12_RenderTarget;
class DX12_DepthStencilTarget;
class DX12_PipelineState;
class DX12_RootSignature;
class DX12_CmdSignature;
class DX12_Buffer;
class DX12_TimerQuery;
class ViewportSet;
class ScissorRectSet;

enum gpuCmdOrders
{
  INIT_GPU_CMD_ORDER = 0,
  START_GPU_CMD_ORDER,
  CLEAR_GPU_CMD_ORDER,
  SKINNING_GPU_CMD_ORDER,
  DECAL_GPU_CMD_ORDER,
  PRE_BASE_GPU_CMD_ORDER,
  BASE_GPU_CMD_ORDER,
  BASE_ALPHA_TESTED_GPU_CMD_ORDER,
  POST_BASE_GPU_CMD_ORDER,
  SKY_GPU_CMD_ORDER,
  POST_PROCESS_GPU_CMD_ORDER,
  GUI_GPU_CMD_ORDER,
  COPY_BACK_BUFFER_GPU_CMD_ORDER,
  END_GPU_CMD_ORDER
};

enum gpuCmdModes
{
  UPLOAD_GPU_CMD_MODE=0,
  READBACK_GPU_CMD_MODE,
  RESOURCE_BARRIER_GPU_CMD_MODE,
  CLEAR_RTV_GPU_CMD_MODE,
  CLEAR_DSV_GPU_CMD_MODE,
  DRAW_GPU_CMD_MODE,
  INDIRECT_DRAW_GPU_CMD_MODE,
  COMPUTE_GPU_CMD_MODE,
  INDIRECT_COMPUTE_GPU_CMD_MODE,
  BEGIN_TIMER_QUERY_GPU_CMD_MODE,
  END_TIMER_QUERY_GPU_CMD_MODE,
  BEGIN_GPU_MARKER_GPU_CMD_MODE,
  END_GPU_MARKER_GPU_CMD_MODE
};

struct TransitionBarrier
{
  friend class DX12_CmdList;

  DX12_IResource *resource;
  resourceStates newResourceState;

private:
  resourceStates oldResourceState;
};

struct AliasingBarrier
{
  DX12_IResource *resourceBefore;
  DX12_IResource *resourceAfter;
};

struct UavBarrier
{
  DX12_IResource *resource;
};

struct ResourceBarrier
{
  ResourceBarrier()
  {
    memset(this, 0, sizeof(this));
  }

  resourceBarrrierTypes barrierType;
  union
  {
    TransitionBarrier transition;
    AliasingBarrier aliasing;
    UavBarrier uav;
  };
};

struct RootConst
{
  UINT constData[MAX_NUM_ROOT_CONSTS];
  UINT numConsts;
};

struct RootParam
{
  RootParam()
  {
    memset(this, 0, sizeof(RootParam));
  }

  rootParamTypes rootParamType;
  union
  {
    GpuDescHandle baseGpuDescHandle;
    RootConst rootConst;
    GpuVirtualAddress bufferLocation;
  };
};


class IGpuCmd
{
public:
  friend class DX12_CmdList;

private:
  virtual gpuCmdModes GetGpuCmdMode() const = 0;

  virtual void CopyToCmdData(BYTE *cmdData, UINT &cmdPosition) const = 0;
};

class UploadCmd: public IGpuCmd
{
public:
  UploadCmd():
    resource(nullptr)
  {
  }

  DX12_IResource *resource;

private:
  virtual gpuCmdModes GetGpuCmdMode() const override
  {
    return UPLOAD_GPU_CMD_MODE;
  }

  virtual void CopyToCmdData(BYTE *cmdData, UINT &cmdPosition) const override
  {
    UINT cmdSize = sizeof(UploadCmd);
    memcpy(&cmdData[cmdPosition], this, cmdSize);
    cmdPosition += cmdSize;
  }
};

class ReadbackCmd: public IGpuCmd
{
public:
  ReadbackCmd():
    destResource(nullptr),
    srcResource(nullptr)
  {
  }

  DX12_IResource *destResource;
  DX12_IResource *srcResource;

private:
  virtual gpuCmdModes GetGpuCmdMode() const override
  {
    return READBACK_GPU_CMD_MODE;
  }

  virtual void CopyToCmdData(BYTE *cmdData, UINT &cmdPosition) const override
  {
    UINT cmdSize = sizeof(ReadbackCmd);
    memcpy(&cmdData[cmdPosition], this, cmdSize);
    cmdPosition += cmdSize;
  }
};

class ResourceBarrierCmd: public IGpuCmd
{
public:
  ResourceBarrierCmd():
    barriers(nullptr),
    numBarriers(0)
  {
  }

  ResourceBarrier *barriers;
  UINT numBarriers;

private:
  virtual gpuCmdModes GetGpuCmdMode() const override
  {
    return RESOURCE_BARRIER_GPU_CMD_MODE;
  }

  virtual void CopyToCmdData(BYTE *cmdData, UINT &cmdPosition) const override
  {
    ResourceBarrierCmd *resourceBarrierCmd = reinterpret_cast<ResourceBarrierCmd*>(&cmdData[cmdPosition]);
    UINT cmdSize = sizeof(ResourceBarrierCmd);
    memcpy(resourceBarrierCmd, this, cmdSize);
    cmdPosition += cmdSize;

    assert((numBarriers > 0) && (numBarriers <= MAX_NUM_RESOURCE_BARRIERS));
    ResourceBarrier *barrierData = reinterpret_cast<ResourceBarrier*>(&cmdData[cmdPosition]);
    UINT barrierDataSize = sizeof(ResourceBarrier) * numBarriers;
    memcpy(barrierData, barriers, barrierDataSize);
    resourceBarrierCmd->barriers = barrierData;
    cmdPosition += barrierDataSize;
  }
};

class ClearRtvCmd: public IGpuCmd
{
public:
  ClearRtvCmd():
    renderTarget(nullptr)
  {
  }

  DX12_RenderTarget *renderTarget;

private:
  virtual gpuCmdModes GetGpuCmdMode() const override
  {
    return CLEAR_RTV_GPU_CMD_MODE;
  }

  virtual void CopyToCmdData(BYTE *cmdData, UINT &cmdPosition) const override
  {
    UINT cmdSize = sizeof(ClearRtvCmd);
    memcpy(&cmdData[cmdPosition], this, cmdSize);
    cmdPosition += cmdSize;
  }
};

class ClearDsvCmd: public IGpuCmd
{
public:
  ClearDsvCmd():
    depthStencilTarget(nullptr)
  {
  }

  DX12_DepthStencilTarget *depthStencilTarget;

private:
  virtual gpuCmdModes GetGpuCmdMode() const override
  {
    return CLEAR_DSV_GPU_CMD_MODE;
  }

  virtual void CopyToCmdData(BYTE *cmdData, UINT &cmdPosition) const override
  {
    UINT cmdSize = sizeof(ClearDsvCmd);
    memcpy(&cmdData[cmdPosition], this, cmdSize);
    cmdPosition += cmdSize;
  }
};

class BaseDrawCmd: public IGpuCmd
{
public:
  BaseDrawCmd():
    rtvCpuDescHandles(nullptr),
    numRenderTargets(0),
    viewportSet(nullptr),
    scissorRectSet(nullptr),
    vertexBuffers(nullptr),
    numVertexBuffers(0),
    indexBuffer(nullptr),
    primitiveTopology(UNDEFINED_PRIMITIVE_TOPOLOGY),
    stencilRef(0),
    pipelineState(nullptr),
    rootParams(nullptr),
    numRootParams(0)
  {
    memset(&dsvCpuDescHandle, 0, sizeof(CpuDescHandle));
  }

  CpuDescHandle *rtvCpuDescHandles;
  UINT numRenderTargets;
  CpuDescHandle dsvCpuDescHandle;
  ViewportSet *viewportSet;
  ScissorRectSet *scissorRectSet;
  DX12_Buffer **vertexBuffers;
  UINT numVertexBuffers;
  DX12_Buffer *indexBuffer;
  primitiveTopologies primitiveTopology;
  Vector4 blendFactor;
  UINT stencilRef;
  DX12_PipelineState *pipelineState;
  RootParam *rootParams;
  UINT numRootParams;
};

class DrawCmd: public BaseDrawCmd
{
public:
  DrawCmd():
    firstIndex(0),
    baseVertexIndex(0),
    numElements(0),
    numInstances(1)
  {
  }

  UINT firstIndex;
  UINT baseVertexIndex;
  UINT numElements;
  UINT numInstances;

private:
  virtual gpuCmdModes GetGpuCmdMode() const override
  {
    return DRAW_GPU_CMD_MODE;
  }

  virtual void CopyToCmdData(BYTE *cmdData, UINT &cmdPosition) const override
  {
    DrawCmd *drawCmd = (DrawCmd*)(&cmdData[cmdPosition]);
    UINT cmdSize = sizeof(DrawCmd);
    memcpy(drawCmd, this, cmdSize);
    cmdPosition += cmdSize;

    assert(numRenderTargets <= MAX_NUM_MRTS);
    if(numRenderTargets > 0)
    {
      CpuDescHandle *rtvData = reinterpret_cast<CpuDescHandle*>(&cmdData[cmdPosition]);
      UINT rtvDataSize = sizeof(CpuDescHandle) * numRenderTargets;
      memcpy(rtvData, rtvCpuDescHandles, rtvDataSize);
      drawCmd->rtvCpuDescHandles = rtvData;
      cmdPosition += rtvDataSize;
    }

    assert(numVertexBuffers <= MAX_NUM_VERTEX_BUFFERS);
    if(numVertexBuffers > 0)
    {
      DX12_Buffer **vertexBufferData = reinterpret_cast<DX12_Buffer**>(&cmdData[cmdPosition]);
      UINT vertexBufferDataSize = sizeof(DX12_Buffer*) * numVertexBuffers;
      memcpy(vertexBufferData, vertexBuffers, vertexBufferDataSize);
      drawCmd->vertexBuffers = vertexBufferData;
      cmdPosition += vertexBufferDataSize;
    }

    assert(numRootParams <= MAX_NUM_ROOT_PARAMS);
    if(numRootParams > 0)
    {
      RootParam *rootParamData = reinterpret_cast<RootParam*>(&cmdData[cmdPosition]);
      UINT rootParamDataSize = sizeof(RootParam) * numRootParams;
      memcpy(rootParamData, rootParams, rootParamDataSize);
      drawCmd->rootParams = rootParamData;
      cmdPosition += rootParamDataSize;
    }
  }
};

class IndirectDrawCmd: public BaseDrawCmd
{
public:
  IndirectDrawCmd():
    cmdSignature(nullptr),
    argBuffer(nullptr),
    countBuffer(nullptr),
    argBufferOffset(0),
    countBufferOffset(0),
    maxCmdCount(0)
  {
  }

  DX12_CmdSignature *cmdSignature;
  DX12_Buffer *argBuffer;
  DX12_Buffer *countBuffer;
  UINT64 argBufferOffset;
  UINT64 countBufferOffset;
  UINT maxCmdCount;

private:
  virtual gpuCmdModes GetGpuCmdMode() const override
  {
    return INDIRECT_DRAW_GPU_CMD_MODE;
  }

  virtual void CopyToCmdData(BYTE *cmdData, UINT &cmdPosition) const override
  {
    IndirectDrawCmd *indirectDrawCmd = reinterpret_cast<IndirectDrawCmd*>(&cmdData[cmdPosition]);
    UINT cmdSize = sizeof(IndirectDrawCmd);
    memcpy(indirectDrawCmd, this, cmdSize);
    cmdPosition += cmdSize;

    assert(numRenderTargets <= MAX_NUM_MRTS);
    if(numRenderTargets > 0)
    {
      CpuDescHandle *rtvData = reinterpret_cast<CpuDescHandle*>(&cmdData[cmdPosition]);
      UINT rtvDataSize = sizeof(CpuDescHandle) * numRenderTargets;
      memcpy(rtvData, rtvCpuDescHandles, rtvDataSize);
      indirectDrawCmd->rtvCpuDescHandles = rtvData;
      cmdPosition += rtvDataSize;
    }

    assert(numVertexBuffers <= MAX_NUM_VERTEX_BUFFERS);
    if(numVertexBuffers > 0)
    {
      DX12_Buffer **vertexBufferData = reinterpret_cast<DX12_Buffer**>(&cmdData[cmdPosition]);
      UINT vertexBufferDataSize = sizeof(DX12_Buffer*) * numVertexBuffers;
      memcpy(vertexBufferData, vertexBuffers, vertexBufferDataSize);
      indirectDrawCmd->vertexBuffers = vertexBufferData;
      cmdPosition += vertexBufferDataSize;
    }

    assert(numRootParams <= MAX_NUM_ROOT_PARAMS);
    if(numRootParams > 0)
    {
      RootParam *rootParamData = reinterpret_cast<RootParam*>(&cmdData[cmdPosition]);
      UINT rootParamDataSize = sizeof(RootParam) * numRootParams;
      memcpy(rootParamData, rootParams, rootParamDataSize);
      indirectDrawCmd->rootParams = rootParamData;
      cmdPosition += rootParamDataSize;
    }
  }
};

class BaseComputeCmd: public IGpuCmd
{
public:
  BaseComputeCmd():
    pipelineState(nullptr),
    rootParams(nullptr),
    numRootParams(0)
  {
  }

  DX12_PipelineState *pipelineState;
  RootParam *rootParams;
  UINT numRootParams;
};

class ComputeCmd: public BaseComputeCmd
{
public:
  ComputeCmd():
    numThreadGroupsX(0),
    numThreadGroupsY(0),
    numThreadGroupsZ(0)
  {
  }

  UINT numThreadGroupsX;
  UINT numThreadGroupsY;
  UINT numThreadGroupsZ;

private:
  virtual gpuCmdModes GetGpuCmdMode() const override
  {
    return COMPUTE_GPU_CMD_MODE;
  }

  virtual void CopyToCmdData(BYTE *cmdData, UINT &cmdPosition) const override
  {
    ComputeCmd *computeCmd = reinterpret_cast<ComputeCmd*>(&cmdData[cmdPosition]);
    UINT cmdSize = sizeof(ComputeCmd);
    memcpy(computeCmd, this, cmdSize);
    cmdPosition += cmdSize;

    assert(numRootParams <= MAX_NUM_ROOT_PARAMS);
    if(numRootParams > 0)
    {
      RootParam *rootParamData = reinterpret_cast<RootParam*>(&cmdData[cmdPosition]);
      UINT rootParamDataSize = sizeof(RootParam) * numRootParams;
      memcpy(rootParamData, rootParams, rootParamDataSize);
      computeCmd->rootParams = rootParamData;
      cmdPosition += rootParamDataSize;
    }
  }
};

class IndirectComputeCmd: public BaseComputeCmd
{
public:
  IndirectComputeCmd():
    cmdSignature(nullptr),
    argBuffer(nullptr),
    countBuffer(nullptr),
    argBufferOffset(0),
    countBufferOffset(0),
    maxCmdCount(0)
  {
  }

  DX12_CmdSignature *cmdSignature;
  DX12_Buffer *argBuffer;
  DX12_Buffer *countBuffer;
  UINT64 argBufferOffset;
  UINT64 countBufferOffset;
  UINT maxCmdCount;

private:
  virtual gpuCmdModes GetGpuCmdMode() const override
  {
    return INDIRECT_COMPUTE_GPU_CMD_MODE;
  }

  virtual void CopyToCmdData(BYTE *cmdData, UINT &cmdPosition) const override
  {
    IndirectComputeCmd *indirectComputeCmd = reinterpret_cast<IndirectComputeCmd*>(&cmdData[cmdPosition]);
    UINT cmdSize = sizeof(IndirectComputeCmd);
    memcpy(indirectComputeCmd, this, cmdSize);
    cmdPosition += cmdSize;

    assert(numRootParams <= MAX_NUM_ROOT_PARAMS);
    if(numRootParams > 0)
    {
      RootParam *rootParamData = reinterpret_cast<RootParam*>(&cmdData[cmdPosition]);
      UINT rootParamDataSize = sizeof(RootParam) * numRootParams;
      memcpy(rootParamData, rootParams, rootParamDataSize);
      indirectComputeCmd->rootParams = rootParamData;
      cmdPosition += rootParamDataSize;
    }
  }
};

class BeginTimerQueryCmd: public IGpuCmd
{
public:
  BeginTimerQueryCmd():
    timerQuery(nullptr)
  {
  }

  DX12_TimerQuery *timerQuery;

private:
  virtual gpuCmdModes GetGpuCmdMode() const override
  {
    return BEGIN_TIMER_QUERY_GPU_CMD_MODE;
  }

  virtual void CopyToCmdData(BYTE *cmdData, UINT &cmdPosition) const override
  {
    UINT cmdSize = sizeof(BeginTimerQueryCmd);
    memcpy(&cmdData[cmdPosition], this, cmdSize);
    cmdPosition += cmdSize;
  }
};

class EndTimerQueryCmd: public IGpuCmd
{
public:
  EndTimerQueryCmd():
    timerQuery(nullptr)
  {
  }

  DX12_TimerQuery *timerQuery;

private:
  virtual gpuCmdModes GetGpuCmdMode() const override
  {
    return END_TIMER_QUERY_GPU_CMD_MODE;
  }

  virtual void CopyToCmdData(BYTE *cmdData, UINT &cmdPosition) const override
  {
    UINT cmdSize = sizeof(EndTimerQueryCmd);
    memcpy(&cmdData[cmdPosition], this, cmdSize);
    cmdPosition += cmdSize;
  }
};


class BeginGpuMarkerCmd: public IGpuCmd
{
public:
  BeginGpuMarkerCmd()
  {
    markerName[0] = 0;
  }

  void SetMarkerName(const char *name)
  {
    _snprintf(markerName, MAX_MARKER_NAME - 1, name);
  }

  const char* GetMarkerName() const
  {
    return markerName;
  }

private:
  virtual gpuCmdModes GetGpuCmdMode() const override
  {
    return BEGIN_GPU_MARKER_GPU_CMD_MODE;
  }

  virtual void CopyToCmdData(BYTE *cmdData, UINT &cmdPosition) const override
  {
    UINT cmdSize = sizeof(BeginGpuMarkerCmd);
    memcpy(&cmdData[cmdPosition], this, cmdSize);
    cmdPosition += cmdSize;
  }

  char markerName[MAX_MARKER_NAME];
};

class EndGpuMarkerCmd: public IGpuCmd
{
private:
  virtual gpuCmdModes GetGpuCmdMode() const override
  {
    return END_GPU_MARKER_GPU_CMD_MODE;
  }

  virtual void CopyToCmdData(BYTE *cmdData, UINT &cmdPosition) const override
  {
    UINT cmdSize = sizeof(EndGpuMarkerCmd);
    memcpy(&cmdData[cmdPosition], this, cmdSize);
    cmdPosition += cmdSize;
  }
};

class LastGpuStates
{
public:
  LastGpuStates()
  {
    Reset();
  }

  void Reset()
  {
    memset(this, 0, sizeof(LastGpuStates));
    graphicsRootSignature = true;
    resetDraw = true;
  }

  CpuDescHandle rtvCpuDescHandles[MAX_NUM_MRTS];
  UINT numRenderTargets;
  CpuDescHandle dsvCpuDescHandle;
  ViewportSet *viewportSet;
  ScissorRectSet *scissorRectSet;
  DX12_Buffer *vertexBuffers[MAX_NUM_VERTEX_BUFFERS];
  UINT numVertexBuffers;
  DX12_Buffer *indexBuffer;
  primitiveTopologies primitiveTopology;
  Vector4 blendFactor;
  UINT stencilRef;
  DX12_PipelineState *pipelineState;
  DX12_RootSignature *rootSignature;
  bool graphicsRootSignature;
  bool resetDraw;
};


class SortedGpuCmd
{
public:
  friend class DX12_CmdList;

  SortedGpuCmd() :
    gpuCmd(nullptr),
    order(INIT_GPU_CMD_ORDER),
    sortingPriority(0.0f),
    ID(0),
    cmdMode(UPLOAD_GPU_CMD_MODE)
  {
  }

  UINT GetID() const
  {
    return ID;
  }

  IGpuCmd *gpuCmd;
  gpuCmdOrders order;
  float sortingPriority; // higher sorting priorities will be rendered first

private:
  UINT ID;
  gpuCmdModes cmdMode;
};

#endif



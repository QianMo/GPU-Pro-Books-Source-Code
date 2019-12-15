#ifndef DX12_RENDERER_H
#define DX12_RENDERER_H

#include <List.h>
#include <render_states.h>
#include <RenderFormat.h>
#include <DX12_DescHeap.h>
#include <DX12_RenderTarget.h>
#include <DX12_DepthStencilTarget.h>
#include <DX12_Buffer.h>    
#include <DX12_RootSignature.h>
#include <DX12_PipelineState.h>
#include <DX12_CmdSignature.h>
#include <DX12_Fence.h>
#include <DX12_CmdList.h>
#include <DX12_TimerQuery.h>
#include <DX12_Helpers.h>
#include <ViewportSet.h>
#include <ScissorRectSet.h>
#include <IPostProcessor.h>
#include <Camera.h>

enum cmdListPools
{
  DEFAULT_CMD_LIST_CMD_POOL=1000,
  POSTPROCESS_CMD_LIST_CMP_POOL=200,
  FINAL_CMD_LIST_CMD_POOL=10
};

enum cmdListIDs
{
  DEFAULT_CMD_LIST_ID=0,
  POSTPROCESS_CMD_LIST_ID,
  FINAL_CMD_LIST_ID
};

enum fenceIDs
{
  DEFAULT_FENCE_ID=0
};

enum renderTargetIDs
{
  BACK_BUFFER_RT_ID=0,
  ACCUM_BUFFER_RT_ID
};

enum depthStencilTargetIDs
{
  MAIN_DEPTH_DST_ID=0
};

enum viewportSetIDs
{
  DEFAULT_VIEWPORT_SET_ID=0
};

enum scissorRectSetIDs
{
  DEFAULT_SCISSOR_RECT_SET_ID=0
};

enum cameraIDs
{
  MAIN_CAMERA_ID=0
};

struct DeviceFeatureSupport
{
  bool logicalOp;
  bool typedUavLoads;
  bool rasterizerOrderedViews;
  conservativeRasterizationTiers conservativeRasterizationTier;
};

// DX12_Renderer
//
// Manages DirectX 12 rendering.
class DX12_Renderer
{
public:
  DX12_Renderer() :
    backBufferReadbackBuffer(nullptr),
    backBufferIndex(0),
    captureScreenIndex(-1)
  {
  }

  ~DX12_Renderer()
  {
    Destroy();
  }

  void Destroy();

  bool Create();

  const DeviceFeatureSupport& GetDeviceFeatureSupport() const
  {
    return supportedDeviceFeatures;
  }

  DX12_RenderTarget* CreateRenderTarget(const TextureDesc &desc, const char *name);

  DX12_RenderTarget* GetRenderTarget(UINT index) const
  {
    assert(index < renderTargets.GetSize());
    return renderTargets[index];
  }

  DX12_DepthStencilTarget* CreateDepthStencilTarget(const TextureDesc &desc, const char *name);

  DX12_DepthStencilTarget* GetDepthStencilTarget(UINT index) const
  {
    assert(index < depthStencilTargets.GetSize());
    return depthStencilTargets[index];
  }

  DX12_Buffer* CreateBuffer(const BufferDesc &desc, const char *name);

  DX12_Buffer* GetBuffer(UINT index) const
  {
    assert(index < buffers.GetSize());
    return buffers[index];
  }

  ViewportSet* CreateViewportSet(Viewport *viewports, UINT numViewports);

  ViewportSet* GetViewportSet(UINT index)
  {
    assert(index < viewportSets.GetSize());
    return viewportSets[index];
  }

  ScissorRectSet* CreateScissorRectSet(ScissorRect *scissorRects, UINT numScissorRects);

  ScissorRectSet* GetScissorRectSet(UINT index)
  {
    assert(index < scissorRectSets.GetSize());
    return scissorRectSets[index];
  }

  DX12_RootSignature* CreateRootSignature(const RootSignatureDesc &desc, const char *name);

  DX12_PipelineState* CreatePipelineState(const PipelineStateDesc &desc, const char *name);

  DX12_CmdSignature* CreateCmdSignature(const CmdSignatureDesc &desc, const char *name);

  DX12_Fence* CreateFence(const char *name);

  DX12_CmdList* CreateCmdList(const CmdListDesc &desc, const char *name);

  DX12_CmdList* GetCmdList(UINT index) const
  {
    assert(index < cmdLists.GetSize());
    return cmdLists[index];
  }

  DX12_TimerQuery* CreateTimerQuery(const char *name);

  Camera* CreateCamera(float fovy, float aspectRatio, float nearClipDistance, float farClipDistance);

  Camera* GetCamera(UINT index) const
  {
    assert(index < cameras.GetSize());
    return cameras[index];
  }

  template<class T> T* CreatePostProcessor()
  {
    T *postProcessor = new T;
    if(!postProcessor)
      return nullptr;
    if(!postProcessor->Create())
    {
      SAFE_DELETE(postProcessor);
      return nullptr;
    }
    postProcessors.AddElement((IPostProcessor**)(&postProcessor));
    return postProcessor;
  }

  IPostProcessor* GetPostProcessor(const char *name) const;

  bool FinalizeInit();

  void BeginFrame();

  void EndFrame();

  void CaptureScreen();

  ID3D12Device* GetDevice() const
  {
    return device.Get();
  }

  ID3D12CommandQueue* GetCmdQueue() const
  {
    return cmdQueue.Get();
  }

  IDXGISwapChain3* GetSwapChain() const
  {
    return swapChain.Get();
  }

  UINT GetBackBufferIndex() const
  {
    return backBufferIndex;
  }

  DX12_DescHeap* GetRtvHeap()
  {
    return &rtvHeap;
  }

  DX12_DescHeap* GetDsvHeap()
  {
    return &dsvHeap;
  }

  DX12_DescHeap* GetCbvSrvUavHeap(UINT backBufferIndex)
  {
    return &cbvSrvUavHeaps[backBufferIndex];
  }

  DX12_DescHeap* GetSamplerHeap()
  {
    return &samplerHeap;
  }

private:
  bool CreateDefaultObjects();

  void ExecuteCmdLists(bool render);

  void SaveScreenshot();

  ComPtr<ID3D12Device> device;
  ComPtr<ID3D12CommandQueue> cmdQueue;
  ComPtr<IDXGISwapChain3> swapChain;
  DeviceFeatureSupport supportedDeviceFeatures;

  DX12_DescHeap rtvHeap;
  DX12_DescHeap dsvHeap;
  DX12_DescHeap cbvSrvUavHeaps[NUM_BACKBUFFERS];
  DX12_DescHeap samplerHeap;
  List<DX12_RenderTarget*> renderTargets;
  List<DX12_DepthStencilTarget*> depthStencilTargets;
  List<DX12_Buffer*> buffers;
  List<ViewportSet*> viewportSets;
  List<ScissorRectSet*> scissorRectSets;
  List<DX12_RootSignature*> rootSignatures;
  List<DX12_PipelineState*> pipelineStates;
  List<DX12_CmdSignature*> cmdSignatures;
  List<DX12_Fence*> fences;
  List<DX12_CmdList*> cmdLists;
  List<DX12_TimerQuery*> timerQueries;
  List<Camera*> cameras;
  List<IPostProcessor*> postProcessors;

  List<DX12_CmdList*> sortedCmdLists;
  DX12_Buffer *backBufferReadbackBuffer;
  UINT backBufferIndex;
  int captureScreenIndex;

};

#endif 
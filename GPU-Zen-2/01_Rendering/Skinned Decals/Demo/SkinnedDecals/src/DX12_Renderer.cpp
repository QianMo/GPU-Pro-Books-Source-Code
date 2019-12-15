#include <stdafx.h>
#include <Demo.h>
#include <DX12_Renderer.h>

void DX12_Renderer::Destroy()
{
  if(fences.GetSize() > 0)
    fences[DEFAULT_FENCE_ID]->WaitForGpu();

  SAFE_DELETE_PLIST(renderTargets);
  SAFE_DELETE_PLIST(depthStencilTargets);
  SAFE_DELETE_PLIST(buffers);
  SAFE_DELETE_PLIST(viewportSets);
  SAFE_DELETE_PLIST(scissorRectSets);
  SAFE_DELETE_PLIST(pipelineStates);
  SAFE_DELETE_PLIST(rootSignatures);
  SAFE_DELETE_PLIST(cmdSignatures);
  SAFE_DELETE_PLIST(fences);
  SAFE_DELETE_PLIST(cmdLists);
  SAFE_DELETE_PLIST(timerQueries);
  SAFE_DELETE_PLIST(cameras);
  SAFE_DELETE_PLIST(postProcessors);
}

bool DX12_Renderer::Create()
{
#ifdef _DEBUG
  // enable DX12 debug layer
  ComPtr<ID3D12Debug> debugController;
  if(SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debugController))))
  {
    debugController->EnableDebugLayer();
  }
#endif

  // create DXGI factory
  ComPtr<IDXGIFactory4> factory;
  if(FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory))))
  {
    LOG_ERROR("Failed to create DXGI factory!");
    return false;
  }

  // create DX12 device
  ComPtr<IDXGIAdapter1> adapter;

#ifdef USE_WARP_DEVICE
  if(!SUCCEEDED(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter))))
  {
    LOG_ERROR("Failed to enumerate warp adapter!");
    return false;
  }

  if(!SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device))))
  {
    LOG_ERROR("Failed to create warp DX12 device with minimum Feature Level 11.0!");
    return false;
  }
#else
  bool succeeded = false;
  for(UINT i=0; factory->EnumAdapters1(i, &adapter)!=DXGI_ERROR_NOT_FOUND; i++)
  {
    DXGI_ADAPTER_DESC1 desc;
    adapter->GetDesc1(&desc);
    if(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)
      continue;
    if(SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device))))
    {
      succeeded = true;
      break;
    }
  }
  
  if(!succeeded)
  {
    LOG_ERROR("Failed to create DX12 device with minimum Feature Level 11.0!");
    return false;
  }
#endif

  // query device features
  D3D12_FEATURE_DATA_D3D12_OPTIONS options;
  if(FAILED(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &options, sizeof(options))))
  {
    LOG_ERROR("Failed to query DX12 device features!");
    return false;
  }
  supportedDeviceFeatures.logicalOp = (options.OutputMergerLogicOp > 0);
  supportedDeviceFeatures.typedUavLoads = (options.TypedUAVLoadAdditionalFormats > 0);
  supportedDeviceFeatures.rasterizerOrderedViews = (options.ROVsSupported > 0);
  supportedDeviceFeatures.conservativeRasterizationTier = (conservativeRasterizationTiers)options.ConservativeRasterizationTier;

#if _DEBUG
  // filter debug messages
  ComPtr<ID3D12InfoQueue> infoQueue;
  if(SUCCEEDED(device.Get()->QueryInterface(IID_PPV_ARGS(&infoQueue))))
  {
    D3D12_MESSAGE_SEVERITY messageSeverties[] =
    {
      D3D12_MESSAGE_SEVERITY_INFO
    };

    D3D12_MESSAGE_ID messageIDs[] =
    {
      D3D12_MESSAGE_ID_CREATEINPUTLAYOUT_EMPTY_LAYOUT,
      D3D12_MESSAGE_ID_UNSTABLE_POWER_STATE
    };

    D3D12_INFO_QUEUE_FILTER filter = {};
    filter.DenyList.NumSeverities = _countof(messageSeverties);
    filter.DenyList.pSeverityList = messageSeverties;
    filter.DenyList.NumIDs = _countof(messageIDs);
    filter.DenyList.pIDList = messageIDs;
    infoQueue->PushStorageFilter(&filter);
  }
#endif

  // create DX12 command queue
  D3D12_COMMAND_QUEUE_DESC queueDesc = {};
  queueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
  queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
  if(FAILED(device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&cmdQueue))))
  {
    LOG_ERROR("Failed to create DX12 command queue!");
    return false;
  }

  // create swap chain
  DXGI_SWAP_CHAIN_DESC swapChainDesc = {};
  swapChainDesc.BufferCount = NUM_BACKBUFFERS;
  swapChainDesc.BufferDesc.Width = SCREEN_WIDTH;
  swapChainDesc.BufferDesc.Height = SCREEN_HEIGHT;
  swapChainDesc.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
  swapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
  swapChainDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
  swapChainDesc.OutputWindow = Demo::window->GetHWnd();
  swapChainDesc.SampleDesc.Count = 1;
  swapChainDesc.Windowed = TRUE;
  ComPtr<IDXGISwapChain> tmpSwapChain;
  if(FAILED(factory->CreateSwapChain(cmdQueue.Get(), &swapChainDesc, &tmpSwapChain)))
  {
    LOG_ERROR("Failed to create DXGI swap chain!");
    return false;
  }
  tmpSwapChain.As(&swapChain);
  backBufferIndex = swapChain->GetCurrentBackBufferIndex();

  // disable fullscreen switch when ALT+ENTER is pressed
  if(FAILED(factory->MakeWindowAssociation(Demo::window->GetHWnd(), DXGI_MWA_NO_ALT_ENTER)))
  {
    return false;
  }

  // create DX12 render target view descriptor heap
  DescHeapDesc desc;
  desc.descHeapType = RTV_DESCRIPTOR_HEAP_TYPE;
  desc.maxNumDescs = MAX_NUM_RTV_DESCS;
  if(!rtvHeap.Create(desc, "RTV"))
  {
    LOG_ERROR("Failed to create DX12 render target view descriptor heap!");
    return false;
  }

  // create DX12 depth stencil view descriptor heap
  desc.descHeapType = DSV_DESCRIPTOR_HEAP_TYPE;
  desc.maxNumDescs = MAX_NUM_DSV_DESCS;
  if(!dsvHeap.Create(desc, "DSV"))
  {
    LOG_ERROR("Failed to create DX12 depth stencil view descriptor heap!");
    return false;
  }

  // create DX12 constant buffer view/ shader resource view/ unordered access view descriptor heaps
  desc.descHeapType = CBV_SRV_UAV_DESCRIPTOR_HEAP_TYPE;
  desc.maxNumDescs = MAX_NUM_CBV_SRV_UAV_DESCS;
  desc.shaderVisible = true;
  for(UINT i=0; i<NUM_BACKBUFFERS; i++)
  {
    if(!cbvSrvUavHeaps[i].Create(desc, "CBV_SRV_UAV"))
    {
      LOG_ERROR("Failed to create DX12 constant buffer view/ shader resource view/ unordered access view descriptor heap!");
      return false;
    }
  }

  // create DX12 sampler descriptor heaps
  desc.descHeapType = SAMPLER_DESCRIPTOR_HEAP_TYPE;
  desc.maxNumDescs = MAX_NUM_SAMPLER_DESCS;
  desc.shaderVisible = true;
  if(!samplerHeap.Create(desc, "Sampler"))
  {
    LOG_ERROR("Failed to create DX12 sampler descriptor heap!");
    return false;
  }

  if(!CreateDefaultObjects())
    return false;

  return true;
}

bool DX12_Renderer::CreateDefaultObjects()
{
  // DEFAULT_CMD_LIST
  {
    CmdListDesc desc;
    desc.cmdListOrder = DEFAULT_CMD_LIST_ORDER;
    desc.cmdPoolSize = DEFAULT_CMD_LIST_CMD_POOL;
    desc.record = true;
    if(!CreateCmdList(desc, "Default"))
      return false;
  }

  // POSTPROCESS_CMD_LIST
  {
    CmdListDesc desc;
    desc.cmdListOrder = POSTPROCESS_CMD_LIST_ORDER;
    desc.cmdPoolSize = POSTPROCESS_CMD_LIST_CMP_POOL;
    if(!CreateCmdList(desc, "Postprocess"))
      return false;
  }

  // FINAL_CMD_LIST
  {
    CmdListDesc desc;
    desc.cmdListOrder = FINAL_CMD_LIST_ORDER;
    desc.cmdPoolSize = FINAL_CMD_LIST_CMD_POOL;
    if(!CreateCmdList(desc, "Final"))
      return false;
  }

  // DEFAULT_FENCE
  if(!CreateFence("Default"))
    return false;

  // BACK_BUFFER_RT
  {
    TextureDesc desc;
    desc.width = SCREEN_WIDTH;
    desc.height = SCREEN_HEIGHT;
    desc.format = RGBA8_RENDER_FORMAT;
    desc.flags = (BACK_BUFFER_TEXTURE_FLAG | RENDER_TARGET_TEXTURE_FLAG | SRGB_WRITE_TEXTURE_FLAG);
    desc.initResourceState = PRESENT_RESOURCE_STATE;
    if(!CreateRenderTarget(desc, "Backbuffer render target"))
      return false;
  }

  // ACCUM_BUFFER_RT
  {
    TextureDesc desc;
    desc.width = SCREEN_WIDTH;
    desc.height = SCREEN_HEIGHT;
    desc.format = RGBA16F_RENDER_FORMAT;
    desc.flags = RENDER_TARGET_TEXTURE_FLAG;
    desc.initResourceState = PIXEL_SHADER_RESOURCE_RESOURCE_STATE;;
    if(!CreateRenderTarget(desc, "Accumulation render target"))
      return false;
  }

  // MAIN_DEPTH_DST
  {
    TextureDesc desc;
    desc.width = SCREEN_WIDTH;
    desc.height = SCREEN_HEIGHT;
    desc.format = DEPTH24_STENCIL8_RENDER_FORMAT;
    desc.flags = DEPTH_STENCIL_TARGET_TEXTURE_FLAG;
    desc.initResourceState = PIXEL_SHADER_RESOURCE_RESOURCE_STATE;
    if(!CreateDepthStencilTarget(desc, "Main depth-stencil target"))
      return false;
  }

  // DEFAULT_VIEWPORT_SET
  Viewport viewport;
  viewport.width = SCREEN_WIDTH;
  viewport.height = SCREEN_HEIGHT;
  if(!Demo::renderer->CreateViewportSet(&viewport, 1))
    return false;

  // DEFAULT_SCISSOR_RECT_SET
  ScissorRect scissorRect;
  scissorRect.right = SCREEN_WIDTH;
  scissorRect.bottom = SCREEN_HEIGHT;
  if(!Demo::renderer->CreateScissorRectSet(&scissorRect, 1))
    return false;

  // MAIN_CAMERA
  if(!CreateCamera(80.0f, (float)SCREEN_WIDTH / (float)SCREEN_HEIGHT, 2.0f, 5000.0f))
    return false;

  {
    BufferDesc desc;
    desc.bufferType = STRUCTURED_BUFFER;
    desc.elementSize = RenderFormat::GetBytesPerPixel(RGBA8_RENDER_FORMAT);
    desc.numElements = SCREEN_WIDTH * SCREEN_HEIGHT;
    desc.flags = CPU_READ_BUFFER_FLAG;
    backBufferReadbackBuffer = CreateBuffer(desc, "Backbuffer readback");
    if(!backBufferReadbackBuffer)
      return false;
  }

  return true;
}

DX12_RenderTarget* DX12_Renderer::CreateRenderTarget(const TextureDesc &desc, const char *name)
{
  assert(name != nullptr);
  DX12_RenderTarget *renderTarget = new DX12_RenderTarget;
  if(!renderTarget)
    return nullptr;
  if(!renderTarget->Create(desc, name))
  {
    SAFE_DELETE(renderTarget);
    LOG_ERROR("Failed to create render-target: %s", name);
    return nullptr;
  }
  renderTargets.AddElement(&renderTarget);
  return renderTarget;
}

DX12_DepthStencilTarget* DX12_Renderer::CreateDepthStencilTarget(const TextureDesc &desc, const char *name)
{
  assert(name != nullptr);
  DX12_DepthStencilTarget *depthStencilTarget = new DX12_DepthStencilTarget;
  if(!depthStencilTarget)
    return nullptr;
  if(!depthStencilTarget->Create(desc, name))
  {
    SAFE_DELETE(depthStencilTarget);
    LOG_ERROR("Failed to create depth-stencil-target: %s", name);
    return nullptr;
  }
  depthStencilTargets.AddElement(&depthStencilTarget);
  return depthStencilTarget;
}

DX12_Buffer* DX12_Renderer::CreateBuffer(const BufferDesc &desc, const char *name)
{
  assert(name != nullptr);
  DX12_Buffer *buffer = new DX12_Buffer;
  if(!buffer)
    return nullptr;
  if(!buffer->Create(desc, name))
  {
    SAFE_DELETE(buffer);
    LOG_ERROR("Failed to create buffer: %s", name);
    return nullptr;
  }
  buffers.AddElement(&buffer);
  return buffer;
}

ScissorRectSet* DX12_Renderer::CreateScissorRectSet(ScissorRect *scissorRects, UINT numScissorRects)
{
  assert((scissorRects != nullptr) && (numScissorRects > 0) && (numScissorRects <= MAX_NUM_SCISSOR_RECTS));
  for(UINT i=0; i<scissorRectSets.GetSize(); i++)
  {
    if(scissorRectSets[i]->GetNumScissorRects() == numScissorRects)
    {
      bool isEqual = true;
      for(UINT j=0; j<numScissorRects; j++)
      {
        if(scissorRectSets[i]->GetScissorRects()[j] != scissorRects[j])
        {
          isEqual = false;
          break;
        }
      }
      if(isEqual)
        return scissorRectSets[i];
    }
  }
  ScissorRectSet *scissorRectSet = new ScissorRectSet;
  if(!scissorRectSet)
    return nullptr;
  scissorRectSet->Create(scissorRects, numScissorRects);
  scissorRectSets.AddElement(&scissorRectSet);
  return scissorRectSet;
}

ViewportSet* DX12_Renderer::CreateViewportSet(Viewport *viewports, UINT numViewports)
{
  assert((viewports != nullptr) && (numViewports > 0) && (numViewports <= MAX_NUM_VIEWPORTS));
  for(UINT i=0; i<viewportSets.GetSize(); i++)
  {
    if(viewportSets[i]->GetNumViewports() == numViewports)
    {
      bool isEqual = true;
      for(UINT j=0; j<numViewports; j++)
      {
        if(viewportSets[i]->GetViewports()[j] != viewports[j])
        {
          isEqual = false;
          break;
        }
      }
      if(isEqual)
        return viewportSets[i];
    }
  }
  ViewportSet *viewportSet = new ViewportSet;
  if(!viewportSet)
    return nullptr;
  viewportSet->Create(viewports, numViewports);
  viewportSets.AddElement(&viewportSet);
  return viewportSet;
}

DX12_RootSignature* DX12_Renderer::CreateRootSignature(const RootSignatureDesc &desc, const char *name)
{
  assert(name != nullptr);
  for(UINT i=0; i<rootSignatures.GetSize(); i++)
  {
    if(rootSignatures[i]->GetDesc() == desc)
    {
      return rootSignatures[i];
    }
  }
  DX12_RootSignature *rootSignature = new DX12_RootSignature;
  if(!rootSignature)
    return nullptr;
  if(!rootSignature->Create(desc, name))
  {
    SAFE_DELETE(rootSignature);
    LOG_ERROR("Failed to create root signature: %s", name);
    return nullptr;
  }
  rootSignatures.AddElement(&rootSignature);
  return rootSignature;
}

DX12_PipelineState* DX12_Renderer::CreatePipelineState(const PipelineStateDesc &desc, const char *name)
{
  assert(name != nullptr);
  for(UINT i=0; i<pipelineStates.GetSize(); i++)
  {
    if(pipelineStates[i]->GetDesc() == desc)
    {
      return pipelineStates[i];
    }
  }
  DX12_PipelineState *pipelineState = new DX12_PipelineState;
  if(!pipelineState)
    return nullptr;
  if(!pipelineState->Create(desc, name))
  {
    SAFE_DELETE(pipelineState);
    LOG_ERROR("Failed to create pipeline state: %s", name);
    return nullptr;
  }
  pipelineStates.AddElement(&pipelineState);
  return pipelineState;
}

DX12_CmdSignature* DX12_Renderer::CreateCmdSignature(const CmdSignatureDesc &desc, const char *name)
{
  assert((desc.numArgDescs <= MAX_NUM_CMD_SIGNATURE_ARGS) && (name != nullptr));
  for(UINT i=0; i<cmdSignatures.GetSize(); i++)
  {
    if(cmdSignatures[i]->GetDesc() == desc)
    {
      return cmdSignatures[i];
    }
  }
  DX12_CmdSignature *cmdSignature = new DX12_CmdSignature;
  if(!cmdSignature)
    return nullptr;
  if(!cmdSignature->Create(desc, name))
  {
    SAFE_DELETE(cmdSignature);
    LOG_ERROR("Failed to create command signature: %s", name);
    return nullptr;
  }
  cmdSignatures.AddElement(&cmdSignature);
  return cmdSignature;
}

DX12_Fence* DX12_Renderer::CreateFence(const char *name)
{
  assert(name != nullptr);
  DX12_Fence *fence = new DX12_Fence;
  if(!fence)
    return nullptr;
  if(!fence->Create(name))
  {
    SAFE_DELETE(fence);
    LOG_ERROR("Failed to create fence: %s", name);
    return nullptr;
  }
  fences.AddElement(&fence);
  return fence;
}

DX12_CmdList* DX12_Renderer::CreateCmdList(const CmdListDesc &desc, const char *name)
{
  assert((cmdLists.GetSize() < MAX_NUM_CMD_LISTS) && (name != nullptr));
  DX12_CmdList *cmdList = new DX12_CmdList;
  if(!cmdList)
    return nullptr;
  if(!cmdList->Create(desc, name))
  {
    SAFE_DELETE(cmdList);
    LOG_ERROR("Failed to create command list: %s", name);
    return nullptr;
  }
  cmdLists.AddElement(&cmdList);
  return cmdList;
}

DX12_TimerQuery* DX12_Renderer::CreateTimerQuery(const char *name)
{
  assert(name != nullptr);
  DX12_TimerQuery *timerQuery = new DX12_TimerQuery;
  if(!timerQuery)
    return nullptr;
  if(!timerQuery->Create(name))
  {
    SAFE_DELETE(timerQuery);
    LOG_ERROR("Failed to create timer query: %s", name);
    return nullptr;
  }
  timerQueries.AddElement(&timerQuery);
  return timerQuery;
}

Camera* DX12_Renderer::CreateCamera(float fovy, float aspectRatio, float nearClipDistance, float farClipDistance)
{
  Camera *camera = new Camera;
  if(!camera)
    return nullptr;
  if(!camera->Init(fovy, aspectRatio, nearClipDistance, farClipDistance))
  {
    SAFE_DELETE(camera);
    return nullptr;
  }
  cameras.AddElement(&camera);
  return camera;
}

IPostProcessor* DX12_Renderer::GetPostProcessor(const char *name) const
{
  if(!name)
    return nullptr;
  for(UINT i=0; i<postProcessors.GetSize(); i++)
  {
    if(strcmp(name, postProcessors[i]->GetName()) == 0)
      return postProcessors[i];
  }
  return nullptr;
}

// compare-function passed to qsort
static int CompareCmdLists(const void *a, const void *b)
{
  const DX12_CmdList *cA = *((DX12_CmdList**)a);
  const DX12_CmdList *cB = *((DX12_CmdList**)b);

  if(cA->GetDesc().cmdListOrder < cB->GetDesc().cmdListOrder)
    return -1;
  else if(cA->GetDesc().cmdListOrder > cB->GetDesc().cmdListOrder)
    return 1;
  return 0;
}

void DX12_Renderer::ExecuteCmdLists(bool render)
{
  // sort commands
  sortedCmdLists.Clear();
  for(UINT i=0; i<cmdLists.GetSize(); i++)
  {
    cmdLists[i]->SortGpuCmds();
    sortedCmdLists.AddElement(&cmdLists[i]);
  }
  sortedCmdLists.Sort(CompareCmdLists);

  // patch commands
  for(UINT i=0; i<sortedCmdLists.GetSize(); i++)
  {
    sortedCmdLists[i]->PatchGpuCmds();
  }

  // execute command lists
  ID3D12CommandList *commandLists[MAX_NUM_CMD_LISTS];
  for(UINT i=0; i<sortedCmdLists.GetSize(); i++)
  {
    Demo::threadManager->ScheduleTask(sortedCmdLists[i]);
    commandLists[i] = sortedCmdLists[i]->GetCmdList();
  }
  Demo::threadManager->WaitForTasks();
  cmdQueue->ExecuteCommandLists(sortedCmdLists.GetSize(), commandLists);

  if(render)
  {
    swapChain->Present(VSYNC_ENABLED, 0);
    UINT lastBackBufferIndex = backBufferIndex;
    backBufferIndex = swapChain->GetCurrentBackBufferIndex();
    fences[DEFAULT_FENCE_ID]->MoveToNextFrame(lastBackBufferIndex, backBufferIndex);
  }
  else
  {
    fences[DEFAULT_FENCE_ID]->WaitForGpu();
  }
}

bool DX12_Renderer::FinalizeInit()
{
  ExecuteCmdLists(false);
  return true;
}

void DX12_Renderer::BeginFrame()
{
  for(UINT i=0; i<cmdLists.GetSize(); i++)
    cmdLists[i]->Reset();

  if(captureScreenIndex == backBufferIndex)
  {
    SaveScreenshot();
    captureScreenIndex = -1;
  }

  DX12_CmdList *cmdList = GetCmdList(DEFAULT_CMD_LIST_ID);
  {
    ResourceBarrier barriers[3];
    barriers[0].barrierType = TRANSITION_RESOURCE_BARRIER_TYPE;
    barriers[0].transition.resource = GetRenderTarget(BACK_BUFFER_RT_ID)->GetTexture();
    barriers[0].transition.newResourceState = RENDER_TARGET_RESOURCE_STATE;
    barriers[1].barrierType = TRANSITION_RESOURCE_BARRIER_TYPE;
    barriers[1].transition.resource = GetRenderTarget(ACCUM_BUFFER_RT_ID)->GetTexture();
    barriers[1].transition.newResourceState = RENDER_TARGET_RESOURCE_STATE;
    barriers[2].barrierType = TRANSITION_RESOURCE_BARRIER_TYPE;
    barriers[2].transition.resource = GetDepthStencilTarget(MAIN_DEPTH_DST_ID)->GetTexture();
    barriers[2].transition.newResourceState = DEPTH_WRITE_RESOURCE_STATE;
   
    ResourceBarrierCmd cmd;
    cmd.barriers = barriers;
    cmd.numBarriers = _countof(barriers);
    cmdList->AddGpuCmd(cmd, START_GPU_CMD_ORDER);
  }

  {
    ClearRtvCmd clearCmd;
    clearCmd.renderTarget = GetRenderTarget(ACCUM_BUFFER_RT_ID);
    cmdList->AddGpuCmd(clearCmd, CLEAR_GPU_CMD_ORDER);
  }

  {
    ClearDsvCmd clearCmd;
    clearCmd.depthStencilTarget = GetDepthStencilTarget(MAIN_DEPTH_DST_ID);
    cmdList->AddGpuCmd(clearCmd, CLEAR_GPU_CMD_ORDER);
  }
}

void DX12_Renderer::EndFrame()
{
  for(UINT i=0; i<postProcessors.GetSize(); i++)
    postProcessors[i]->Execute();

  ResourceBarrier barriers[1];
  barriers[0].barrierType = TRANSITION_RESOURCE_BARRIER_TYPE;
  barriers[0].transition.resource = GetRenderTarget(BACK_BUFFER_RT_ID)->GetTexture();
  barriers[0].transition.newResourceState = PRESENT_RESOURCE_STATE;

  ResourceBarrierCmd cmd;
  cmd.barriers = barriers;
  cmd.numBarriers = _countof(barriers);
  GetCmdList(FINAL_CMD_LIST_ID)->AddGpuCmd(cmd, END_GPU_CMD_ORDER);

  ExecuteCmdLists(true);
}

void DX12_Renderer::CaptureScreen()
{
  if(captureScreenIndex < 0)
  {
    DX12_CmdList *cmdList = GetCmdList(FINAL_CMD_LIST_ID);
    {
      ResourceBarrier barriers[1];
      barriers[0].barrierType = TRANSITION_RESOURCE_BARRIER_TYPE;
      barriers[0].transition.resource = GetRenderTarget(BACK_BUFFER_RT_ID)->GetTexture();
      barriers[0].transition.newResourceState = COPY_SOURCE_RESOURCE_STATE;

      ResourceBarrierCmd cmd;
      cmd.barriers = barriers;
      cmd.numBarriers = _countof(barriers);
      cmdList->AddGpuCmd(cmd, COPY_BACK_BUFFER_GPU_CMD_ORDER);
    }

    {
      ReadbackCmd cmd;
      cmd.destResource = backBufferReadbackBuffer;
      cmd.srcResource = GetRenderTarget(BACK_BUFFER_RT_ID)->GetTexture();
      cmdList->AddGpuCmd(cmd, COPY_BACK_BUFFER_GPU_CMD_ORDER);
    }

    {
      ResourceBarrier barriers[1];
      barriers[0].barrierType = TRANSITION_RESOURCE_BARRIER_TYPE;
      barriers[0].transition.resource = GetRenderTarget(BACK_BUFFER_RT_ID)->GetTexture();
      barriers[0].transition.newResourceState = RENDER_TARGET_RESOURCE_STATE;

      ResourceBarrierCmd cmd;
      cmd.barriers = barriers;
      cmd.numBarriers = _countof(barriers);
      cmdList->AddGpuCmd(cmd, COPY_BACK_BUFFER_GPU_CMD_ORDER);
    }

    captureScreenIndex = backBufferIndex;
  }
}

void DX12_Renderer::SaveScreenshot()
{
  // try to find a not existing path for screen-shot	
  char filePath[DEMO_MAX_FILEPATH];
  for(UINT i=0; i<1000; i++)
  {
    sprintf(filePath, "../Data/screenshots/screen%d.bmp", i);
    if(!Demo::fileManager->FilePathExists(filePath))
      break;
    if(i == 999)
      return;
  }

  unsigned char *destData = new unsigned char[SCREEN_WIDTH * SCREEN_HEIGHT * 3];
  if(!destData)
    return;

  // read back buffer data
  unsigned char* mappedBuffer;
  backBufferReadbackBuffer->GetResource()->Map(0, nullptr, reinterpret_cast<void**>(&mappedBuffer));
  const UINT flipOffset = SCREEN_HEIGHT - 1;
  for(UINT y=0; y<SCREEN_HEIGHT; y++)
  {
    for(UINT x=0; x<SCREEN_WIDTH; x++)
    {
      UINT destIndex = 3 * (((flipOffset - y) * SCREEN_WIDTH) + x);
      UINT srcIndex = 4 * ((y * SCREEN_WIDTH) + x);
      destData[destIndex] = mappedBuffer[srcIndex + 2];
      destData[destIndex + 1] = mappedBuffer[srcIndex + 1];
      destData[destIndex + 2] = mappedBuffer[srcIndex];
    }
  }
  backBufferReadbackBuffer->GetResource()->Unmap(0, nullptr);

  // fill up BMP info-header
  BITMAPINFOHEADER infoHeader;
  memset(&infoHeader, 0, sizeof(BITMAPINFOHEADER));
  infoHeader.biSize = sizeof(BITMAPINFOHEADER);
  infoHeader.biWidth = SCREEN_WIDTH;
  infoHeader.biHeight = SCREEN_HEIGHT;
  infoHeader.biPlanes = 1;
  infoHeader.biBitCount = 24;
  infoHeader.biCompression = BI_RGB;
  infoHeader.biSizeImage = infoHeader.biWidth * infoHeader.biHeight * 3;

  // fill up BMP file-header
  BITMAPFILEHEADER fileHeader;
  memset(&fileHeader, 0, sizeof(BITMAPFILEHEADER));
  fileHeader.bfOffBits = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER);
  fileHeader.bfSize = fileHeader.bfOffBits + infoHeader.biSizeImage;
  fileHeader.bfType = 0x4D42;

  // write file-header/ info-header/ data to bitmap-file
  FILE *file = nullptr;
  fopen_s(&file, filePath, "wb");
  if(file != nullptr)
  {
    fwrite(&fileHeader, sizeof(BITMAPFILEHEADER), 1, file);
    fwrite(&infoHeader, sizeof(BITMAPINFOHEADER), 1, file);
    fwrite(destData, 1, SCREEN_WIDTH * SCREEN_HEIGHT * 3, file);
    fclose(file);
  }

  SAFE_DELETE_ARRAY(destData);
}
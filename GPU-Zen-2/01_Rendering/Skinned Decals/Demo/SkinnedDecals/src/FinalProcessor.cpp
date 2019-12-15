#include <stdafx.h>
#include <Demo.h>
#include <FinalProcessor.h>

bool FinalProcessor::Create()
{	
  backBufferRT = Demo::renderer->GetRenderTarget(BACK_BUFFER_RT_ID);
  if(!backBufferRT)
    return false;

  DX12_Shader *shader = Demo::resourceManager->LoadShader("shaders/finalPass.sdr");
  if(!shader)
    return false;
  
  RootSignatureDesc rootSignatureDesc;
  rootSignatureDesc.numRootParamDescs = 1;
  rootSignatureDesc.rootParamDescs[0].rootParamType = DESC_TABLE_ROOT_PARAM;
  rootSignatureDesc.rootParamDescs[0].rootDescTableDesc.numRanges = 1;
  rootSignatureDesc.rootParamDescs[0].rootDescTableDesc.ranges[0].rangeType = SRV_ROOT_DESC_TABLE_RANGE;
  rootSignatureDesc.rootParamDescs[0].rootDescTableDesc.ranges[0].numDescs = 1;
  rootSignatureDesc.rootParamDescs[0].rootDescTableDesc.ranges[0].baseShaderReg = 0;
  rootSignatureDesc.rootParamDescs[0].rootDescTableDesc.ranges[0].regSpace = 0;
  rootSignatureDesc.rootParamDescs[0].shaderVisibility = PS_SHADER_VIS;
  DX12_RootSignature *rootSignature = Demo::renderer->CreateRootSignature(rootSignatureDesc, "Final processor");
  if(!rootSignature)
    return false;
  
  PipelineStateDesc pipelineStateDesc(GRAPHICS_PIPELINE_STATE);
  pipelineStateDesc.rootSignature = rootSignature;
  pipelineStateDesc.graphics.shader = shader;
  pipelineStateDesc.graphics.numRenderTargets = 1;
  pipelineStateDesc.graphics.rtvFormats[0] = RenderFormat::ConvertToSrgbFormat(backBufferRT->GetTexture()->GetTextureDesc().format);
  pipelineStateDesc.graphics.depthStencilDesc.depthTest = false;
  pipelineStateDesc.graphics.depthStencilDesc.depthMask = false;
  pipelineState = Demo::renderer->CreatePipelineState(pipelineStateDesc, "Final processor");
  if(!pipelineState)
    return false;

  descTable.AddTextureSrv(Demo::renderer->GetRenderTarget(ACCUM_BUFFER_RT_ID)->GetTexture());
  
  return true;
}

void FinalProcessor::Execute()
{
  if(!active)
    return;

  DX12_CmdList *cmdList = Demo::renderer->GetCmdList(POSTPROCESS_CMD_LIST_ID);

  SCOPED_GPU_MARKER(cmdList, POST_PROCESS_GPU_CMD_ORDER, "Final Processor");

  {
    ResourceBarrier barriers[1];
    barriers[0].barrierType = TRANSITION_RESOURCE_BARRIER_TYPE;
    barriers[0].transition.resource = Demo::renderer->GetRenderTarget(ACCUM_BUFFER_RT_ID)->GetTexture();
    barriers[0].transition.newResourceState = PIXEL_SHADER_RESOURCE_RESOURCE_STATE;

    ResourceBarrierCmd cmd;
    cmd.barriers = barriers;
    cmd.numBarriers = _countof(barriers);
    cmdList->AddGpuCmd(cmd, POST_PROCESS_GPU_CMD_ORDER);
  }

  {
    CpuDescHandle rtvCpuDescHandles[1] = 
    {
      backBufferRT->GetRtv(SRGB_RTV_TYPE).cpuDescHandle
    };

    RootParam rootParams[1];
    rootParams[0].rootParamType = DESC_TABLE_ROOT_PARAM;
    rootParams[0].baseGpuDescHandle = descTable.GetBaseDescHandle().gpuDescHandle;
    
    DrawCmd cmd;
    cmd.rtvCpuDescHandles = rtvCpuDescHandles;
    cmd.numRenderTargets = _countof(rtvCpuDescHandles);
    cmd.viewportSet = Demo::renderer->GetViewportSet(DEFAULT_VIEWPORT_SET_ID);
    cmd.scissorRectSet = Demo::renderer->GetScissorRectSet(DEFAULT_SCISSOR_RECT_SET_ID);
    cmd.primitiveTopology = TRIANGLELIST_PRIMITIVE_TOPOLOGY;
    cmd.firstIndex = 0;
    cmd.numElements = 3;
    cmd.pipelineState = pipelineState;
    cmd.rootParams = rootParams;
    cmd.numRootParams = _countof(rootParams);
    cmdList->AddGpuCmd(cmd, POST_PROCESS_GPU_CMD_ORDER);
  }
}

#include <stdafx.h>
#include <Demo.h>
#include <Sky.h>

bool Sky::Create()
{	
  RootSignatureDesc rootSignatureDesc;
  DX12_RootSignature *rootSignature = Demo::renderer->CreateRootSignature(rootSignatureDesc, "Sky");
  if(!rootSignature)
    return false;

  DX12_Shader *shader = Demo::resourceManager->LoadShader("shaders/sky.sdr");
  if(!shader)
    return false;

  PipelineStateDesc pipelineStateDesc(GRAPHICS_PIPELINE_STATE);
  pipelineStateDesc.rootSignature = rootSignature;
  pipelineStateDesc.graphics.shader = shader;
  pipelineStateDesc.graphics.numRenderTargets = 1;
  pipelineStateDesc.graphics.rtvFormats[0] = Demo::renderer->GetRenderTarget(ACCUM_BUFFER_RT_ID)->GetTexture()->GetTextureDesc().format;
  pipelineStateDesc.graphics.dsvFormat = Demo::renderer->GetDepthStencilTarget(MAIN_DEPTH_DST_ID)->GetTexture()->GetTextureDesc().format;
  pipelineStateDesc.graphics.depthStencilDesc.depthTest = true;
  pipelineStateDesc.graphics.depthStencilDesc.depthMask = false;
  pipelineStateDesc.graphics.depthStencilDesc.depthFunc = EQUAL_CMP_FUNC;
  pipelineState = Demo::renderer->CreatePipelineState(pipelineStateDesc, "Sky");
  if(!pipelineState)
    return false;

  return true;
}

void Sky::Execute()
{
  if(!active)
    return;

  DX12_CmdList *cmdList = Demo::renderer->GetCmdList(DEFAULT_CMD_LIST_ID);

  SCOPED_GPU_MARKER(cmdList, SKY_GPU_CMD_ORDER, "Sky");

  {
    ResourceBarrier barriers[1];
    barriers[0].transition.resource = Demo::renderer->GetDepthStencilTarget(MAIN_DEPTH_DST_ID)->GetTexture();
    barriers[0].transition.newResourceState = DEPTH_READ_RESOURCE_STATE;

    ResourceBarrierCmd cmd;
    cmd.barriers = barriers;
    cmd.numBarriers = _countof(barriers);
    cmdList->AddGpuCmd(cmd, SKY_GPU_CMD_ORDER);
  }

  {
    CpuDescHandle rtvCpuDescHandles[] = 
    {
      Demo::renderer->GetRenderTarget(ACCUM_BUFFER_RT_ID)->GetRtv().cpuDescHandle
    };

    DrawCmd cmd;
    cmd.rtvCpuDescHandles = rtvCpuDescHandles;
    cmd.numRenderTargets = _countof(rtvCpuDescHandles);
    cmd.dsvCpuDescHandle = Demo::renderer->GetDepthStencilTarget(MAIN_DEPTH_DST_ID)->GetDsv(READ_ONLY_DSV_TYPE).cpuDescHandle;
    cmd.viewportSet = Demo::renderer->GetViewportSet(DEFAULT_VIEWPORT_SET_ID);
    cmd.scissorRectSet = Demo::renderer->GetScissorRectSet(DEFAULT_SCISSOR_RECT_SET_ID);
    cmd.primitiveTopology = TRIANGLELIST_PRIMITIVE_TOPOLOGY;
    cmd.firstIndex = 0;
    cmd.numElements = 3;
    cmd.pipelineState = pipelineState;
    cmdList->AddGpuCmd(cmd, SKY_GPU_CMD_ORDER);
  }
}

#include <stdafx.h>
#include <Demo.h>
#include <Sky.h>

bool Sky::Create()
{	
  sceneRT = Demo::renderer->GetRenderTarget(GBUFFERS_RT_ID);
  if(!sceneRT)
    return false;

  // render only into the accumulation render-target of the GBuffers
  RtConfigDesc desc;
  desc.numColorBuffers = 1;
  rtConfig = Demo::renderer->CreateRenderTargetConfig(desc);
  if(!rtConfig)
    return false;

  skyShader = Demo::resourceManager->LoadShader("shaders/sky.sdr");
  if(!skyShader)
    return false;

  // only render sky, where stencil buffer is still 0
  DepthStencilDesc depthStencilDesc;
  depthStencilDesc.stencilTest = true;
  depthStencilDesc.stencilRef = 1;
  depthStencilDesc.stencilFunc = GREATER_COMP_FUNC;
  depthStencilState = Demo::renderer->CreateDepthStencilState(depthStencilDesc);
  if(!depthStencilState)
    return false;

  return true;
}

void Sky::Execute()
{
  if(!active)
    return;
  GpuCmd gpuCmd(DRAW_CM);
  gpuCmd.order = SKY_CO;
  gpuCmd.draw.renderTarget = sceneRT;
  gpuCmd.draw.renderTargetConfig = rtConfig;
  gpuCmd.draw.shader = skyShader;
  Demo::renderer->SetupPostProcessSurface(gpuCmd.draw);
  gpuCmd.draw.depthStencilState = depthStencilState;
  Demo::renderer->AddGpuCmd(gpuCmd); 
}

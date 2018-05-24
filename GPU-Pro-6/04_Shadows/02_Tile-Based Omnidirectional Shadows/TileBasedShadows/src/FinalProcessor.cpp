#include <stdafx.h>
#include <Demo.h>
#include <FinalProcessor.h>

bool FinalProcessor::Create()
{	
  sceneRT = Demo::renderer->GetRenderTarget(GBUFFERS_RT_ID);
  if(!sceneRT)
    return false;
  backBufferRT = Demo::renderer->GetRenderTarget(BACK_BUFFER_RT_ID);
  if(!backBufferRT)
    return false;

  RtConfigDesc desc;
  desc.flags = SRGB_WRITE_RTCF;
  rtConfig = Demo::renderer->CreateRenderTargetConfig(desc);
  if(!rtConfig)
    return false;
  
  finalPassShader = Demo::resourceManager->LoadShader("shaders/finalPass.sdr");
  if(!finalPassShader)
    return false;

  return true;
}

void FinalProcessor::Execute()
{
  if(!active)
    return;
  GpuCmd gpuCmd(DRAW_CM);	
  gpuCmd.order = POST_PROCESS_CO;
  gpuCmd.draw.renderTarget = backBufferRT;
  gpuCmd.draw.renderTargetConfig = rtConfig;
  gpuCmd.draw.textures[COLOR_TEX_ID] = sceneRT->GetTexture();
  gpuCmd.draw.shader = finalPassShader;
  Demo::renderer->SetupPostProcessSurface(gpuCmd.draw);
  Demo::renderer->AddGpuCmd(gpuCmd); 
}

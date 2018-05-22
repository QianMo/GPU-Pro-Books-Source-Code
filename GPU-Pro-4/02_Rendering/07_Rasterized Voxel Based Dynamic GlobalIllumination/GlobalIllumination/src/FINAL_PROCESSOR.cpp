#include <stdafx.h>
#include <DEMO.h>
#include <FINAL_PROCESSOR.h>

bool FINAL_PROCESSOR::Create()
{	
	sceneRT = DEMO::renderer->GetRenderTarget(GBUFFER_RT_ID);
  if(!sceneRT)
    return false;
	backBufferRT = DEMO::renderer->GetRenderTarget(BACK_BUFFER_RT_ID);
	if(!backBufferRT)
		return false;
	
	finalPassShader = DEMO::resourceManager->LoadShader("shaders/finalPass.sdr");
	if(!finalPassShader)
		return false;

	return true;
}

DX11_RENDER_TARGET* FINAL_PROCESSOR::GetOutputRT() const
{
	return backBufferRT;
}

void FINAL_PROCESSOR::AddSurfaces()
{
	SURFACE surface;
	surface.renderTarget = backBufferRT;
	surface.renderOrder = POST_PROCESS_RO;
	surface.colorTexture = sceneRT->GetTexture();
	surface.shader = finalPassShader;
	DEMO::renderer->SetupPostProcessSurface(surface);
	DEMO::renderer->AddSurface(surface); 
}

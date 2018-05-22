#include <stdafx.h>
#include <DEMO.h>
#include <SKY.h>

bool SKY::Create()
{	
	sceneRT = DEMO::renderer->GetRenderTarget(GBUFFER_RT_ID);
	if(!sceneRT)
		return false;

	// only render into the accumulation render-target of the GBuffer
	RT_CONFIG_DESC desc;
	desc.numColorBuffers = 1;
	rtConfig = DEMO::renderer->CreateRenderTargetConfig(desc);
	if(!rtConfig)
		return false;

	skyShader = DEMO::resourceManager->LoadShader("shaders/sky.sdr");
	if(!skyShader)
		return false;

	// only render sky, where stencil buffer is still 0
	DEPTH_STENCIL_DESC depthStencilDesc;
	depthStencilDesc.stencilTest = true;
	depthStencilDesc.stencilRef = 1;
	depthStencilDesc.stencilFunc = GREATER_COMP_FUNC;
	depthStencilState = DEMO::renderer->CreateDepthStencilState(depthStencilDesc);
	if(!depthStencilState)
		return false;

	return true;
}

DX11_RENDER_TARGET* SKY::GetOutputRT() const
{
	return sceneRT;
}

void SKY::AddSurfaces()
{
	SURFACE surface;
	surface.renderTarget = sceneRT;
	surface.renderTargetConfig = rtConfig;
	surface.renderOrder = SKY_RO;
	surface.shader = skyShader;
	DEMO::renderer->SetupPostProcessSurface(surface);
	surface.depthStencilState = depthStencilState;
	DEMO::renderer->AddSurface(surface); 
}

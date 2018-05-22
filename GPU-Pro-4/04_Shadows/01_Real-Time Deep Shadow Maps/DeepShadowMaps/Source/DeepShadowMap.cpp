#include "DeepShadowMap.h"
#include "MemoryLeakTracker.h"
#include <fstream>
#include "IncludeHandler.h"

DeepShadowMap::DeepShadowMap() : ICoreBase()
{
	startElementBufSRV = NULL;
	startElementBufUAV = NULL;
	startElementBuf = NULL;
	linkedListBufSRV = NULL;
	linkedListBufUAV = NULL;
	linkedListBuf = NULL;
	linkedListBuf2SRV = NULL;
	linkedListBuf2UAV = NULL;
	linkedListBuf2 = NULL;
	linkedListBuf3SRV = NULL;
	linkedListBuf3UAV = NULL;
	linkedListBuf3 = NULL;
}

CoreResult DeepShadowMap::Init(Core *core, UINT dim, DWORD numElementsInBuffer)
{
	vp.Width    = (float)dim;
	vp.Height   = (float)dim;
	vp.MinDepth = 0.0f;
	vp.MaxDepth = 1.0f;
	vp.TopLeftX = 0;
	vp.TopLeftY = 0;
	this->core = core;
	
	for(int effect = 0; effect < NUM_EFFECTS; effect++)
		dsmEffects[effect].dsmEffect = NULL;

	// Create linked list buffer
	// One element contains the depth and the next pointer
	D3D11_BUFFER_DESC bufDesc;
    ZeroMemory(&bufDesc, sizeof(bufDesc));
    bufDesc.BindFlags = D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;
    bufDesc.ByteWidth = (2 * sizeof(float) + sizeof(UINT)) * numElementsInBuffer;
    bufDesc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
    bufDesc.StructureByteStride = 2 * sizeof(float) + sizeof(UINT);

	// contains LinkedListEntryDepthAlphaNext
	HRESULT hr = core->GetDevice()->CreateBuffer(&bufDesc, NULL, &linkedListBuf);
	if(FAILED(hr))
	{
		CoreLog::Information(L"Error creating RW buf for deep shadow map: HR = 0x%x", hr);
		return CORE_MISC_ERROR;
	}

	// contains LinkedListEntryWithPrev
	bufDesc.ByteWidth = (2 * sizeof(float) + 2 * sizeof(UINT)) * numElementsInBuffer;
    bufDesc.StructureByteStride = 2 * sizeof(float) + 2 * sizeof(UINT);
	
	hr = core->GetDevice()->CreateBuffer(&bufDesc, NULL, &linkedListBuf2);
	if(FAILED(hr))
	{
		CoreLog::Information(L"Error creating RW buf for deep shadow map: HR = 0x%x", hr);
		return CORE_MISC_ERROR;
	}

	// contains LinkedListEntryNeighbors
	bufDesc.ByteWidth = (2 * sizeof(UINT)) * numElementsInBuffer;
    bufDesc.StructureByteStride = 2 * sizeof(UINT);

	hr = core->GetDevice()->CreateBuffer(&bufDesc, NULL, &linkedListBuf3);
	if(FAILED(hr))
	{
		CoreLog::Information(L"Error creating RW buf for deep shadow map: HR = 0x%x", hr);
		return CORE_MISC_ERROR;
	}

	D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc;
    ZeroMemory(&srvDesc, sizeof(srvDesc));
    srvDesc.ViewDimension = D3D11_SRV_DIMENSION_BUFFEREX;
    srvDesc.BufferEx.FirstElement = 0;
	srvDesc.Format = DXGI_FORMAT_UNKNOWN;
    srvDesc.BufferEx.NumElements = numElementsInBuffer;

	hr = core->GetDevice()->CreateShaderResourceView(linkedListBuf, &srvDesc, &linkedListBufSRV);
	if(FAILED(hr))
	{
		SAFE_RELEASE(linkedListBuf);
		CoreLog::Information(L"Error creating RW buf SRV for deep shadow map: HR = 0x%x", hr);
		return CORE_MISC_ERROR;
	}

	hr = core->GetDevice()->CreateShaderResourceView(linkedListBuf2, &srvDesc, &linkedListBuf2SRV);
	if(FAILED(hr))
	{
		SAFE_RELEASE(linkedListBuf);
		CoreLog::Information(L"Error creating RW buf SRV for deep shadow map: HR = 0x%x", hr);
		return CORE_MISC_ERROR;
	}

	hr = core->GetDevice()->CreateShaderResourceView(linkedListBuf3, &srvDesc, &linkedListBuf3SRV);
	if(FAILED(hr))
	{
		SAFE_RELEASE(linkedListBuf);
		CoreLog::Information(L"Error creating RW buf SRV for deep shadow map: HR = 0x%x", hr);
		return CORE_MISC_ERROR;
	}

	D3D11_UNORDERED_ACCESS_VIEW_DESC descUAV;
    ZeroMemory(&descUAV, sizeof(D3D11_UNORDERED_ACCESS_VIEW_DESC) );
    descUAV.Format = DXGI_FORMAT_UNKNOWN;
    descUAV.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
	descUAV.Buffer.FirstElement = 0;
	descUAV.Buffer.NumElements = numElementsInBuffer;
	descUAV.Buffer.Flags = D3D11_BUFFER_UAV_FLAG_COUNTER;
    
	hr = core->GetDevice()->CreateUnorderedAccessView(linkedListBuf, &descUAV, &linkedListBufUAV);
	if(FAILED(hr))
	{
		SAFE_RELEASE(linkedListBuf);
		SAFE_RELEASE(linkedListBufSRV);
		CoreLog::Information(L"Error creating RW buf UAV for deep shadow map: HR = 0x%x", hr);
		return CORE_MISC_ERROR;
	}

	descUAV.Buffer.Flags = 0;

	hr = core->GetDevice()->CreateUnorderedAccessView(linkedListBuf2, &descUAV, &linkedListBuf2UAV);
	if(FAILED(hr))
	{
		SAFE_RELEASE(linkedListBuf);
		SAFE_RELEASE(linkedListBufSRV);
		CoreLog::Information(L"Error creating RW buf UAV for deep shadow map: HR = 0x%x", hr);
		return CORE_MISC_ERROR;
	}

	hr = core->GetDevice()->CreateUnorderedAccessView(linkedListBuf3, &descUAV, &linkedListBuf3UAV);
	if(FAILED(hr))
	{
		SAFE_RELEASE(linkedListBuf);
		SAFE_RELEASE(linkedListBufSRV);
		CoreLog::Information(L"Error creating RW buf UAV for deep shadow map: HR = 0x%x", hr);
		return CORE_MISC_ERROR;
	}

	// Create start element buffer
	bufDesc.ByteWidth = (sizeof(UINT)) * numElementsInBuffer;
    bufDesc.StructureByteStride = sizeof(UINT);
	
	hr = core->GetDevice()->CreateBuffer(&bufDesc, NULL, &startElementBuf);
	if(FAILED(hr))
	{
		SAFE_RELEASE(linkedListBuf);
		SAFE_RELEASE(linkedListBufSRV);
		SAFE_RELEASE(linkedListBufUAV);
		CoreLog::Information(L"Error creating RW buf for deep shadow map: HR = 0x%x", hr);
		return CORE_MISC_ERROR;
	}

	hr = core->GetDevice()->CreateShaderResourceView(startElementBuf, &srvDesc, &startElementBufSRV);
	if(FAILED(hr))
	{
		SAFE_RELEASE(startElementBuf);
		SAFE_RELEASE(linkedListBuf);
		SAFE_RELEASE(linkedListBufSRV);
		SAFE_RELEASE(linkedListBufUAV);
		CoreLog::Information(L"Error creating RW buf SRV for deep shadow map: HR = 0x%x", hr);
		return CORE_MISC_ERROR;
	}

	hr = core->GetDevice()->CreateUnorderedAccessView(startElementBuf, &descUAV, &startElementBufUAV);
	if(FAILED(hr))
	{
		SAFE_RELEASE(startElementBuf);
		SAFE_RELEASE(startElementBufSRV);
		SAFE_RELEASE(linkedListBuf);
		SAFE_RELEASE(linkedListBufSRV);
		SAFE_RELEASE(linkedListBufUAV);
		CoreLog::Information(L"Error creating RW buf UAV for deep shadow map: HR = 0x%x", hr);
		return CORE_MISC_ERROR;
	}

	
	for(int effect = 0; effect < NUM_EFFECTS; effect++)
	{
		std::ifstream effectFile;
		effectFile.open(EffectPaths[effect] +L"\\DeepShadowMap.hlsl");
	
		ID3D10Blob *errors = NULL;
		IncludeHandler *ih = new IncludeHandler(effect);
		CoreResult cr = LoadEffectFromStream(core, effectFile, ih, 0, 0, &errors, &dsmEffects[effect].dsmEffect);
		SAFE_DELETE(ih);
		effectFile.close();
		if(cr != CORE_OK)
		{
			SAFE_RELEASE(startElementBuf);
			SAFE_RELEASE(startElementBufSRV);
			SAFE_RELEASE(startElementBufUAV);
			SAFE_RELEASE(linkedListBuf);
			SAFE_RELEASE(linkedListBufSRV);
			SAFE_RELEASE(linkedListBufUAV);
			CoreLog::Information(L"Error compiling shader for deep shadow map: HR = 0x%x", hr);
			if(errors)
			{
				CoreLog::Information((char *)errors->GetBufferPointer());
				MessageBoxA(NULL, (char *)errors->GetBufferPointer(), NULL, 0);
				errors->Release();
			}
			return cr;
		}

		dsmEffects[effect].dsmRenderTechnique = dsmEffects[effect].dsmEffect->GetTechniqueByName("Render");
		dsmEffects[effect].dsmSortTechnique = dsmEffects[effect].dsmEffect->GetTechniqueByName("Sort");
		dsmEffects[effect].dsmLinkTechnique = dsmEffects[effect].dsmEffect->GetTechniqueByName("Link");
		dsmEffects[effect].dsmEffectDimension = dsmEffects[effect].dsmEffect->GetVariableByName("Dimension")->AsScalar();
		dsmEffects[effect].dsmEffectAlpha = dsmEffects[effect].dsmEffect->GetVariableByName("Alpha")->AsScalar();
		dsmEffects[effect].dsmEffectWorldViewProj = dsmEffects[effect].dsmEffect->GetVariableByName("WorldViewProj")->AsMatrix();
		dsmEffects[effect].dsmEffectLinkedListBufWP = dsmEffects[effect].dsmEffect->GetVariableByName("LinkedListBufWP")->AsUnorderedAccessView();
		dsmEffects[effect].dsmEffectLinkedListBufDAN = dsmEffects[effect].dsmEffect->GetVariableByName("LinkedListBufDAN")->AsUnorderedAccessView();
		dsmEffects[effect].dsmEffectNeighborsBuf = dsmEffects[effect].dsmEffect->GetVariableByName("NeighborsBuf")->AsUnorderedAccessView();
		dsmEffects[effect].dsmEffectStartElementBuf = dsmEffects[effect].dsmEffect->GetVariableByName("StartElementBuf")->AsUnorderedAccessView();
		dsmEffects[effect].dsmEffectLinkedListBufWPRO = dsmEffects[effect].dsmEffect->GetVariableByName("LinkedListBufWPRO")->AsShaderResource();
		dsmEffects[effect].dsmEffectLinkedListBufDANRO = dsmEffects[effect].dsmEffect->GetVariableByName("LinkedListBufDANRO")->AsShaderResource();
		dsmEffects[effect].dsmEffectNeighborsBufRO = dsmEffects[effect].dsmEffect->GetVariableByName("NeighborsBufRO")->AsShaderResource();
		dsmEffects[effect].dsmEffectStartElementBufRO = dsmEffects[effect].dsmEffect->GetVariableByName("StartElementBufRO")->AsShaderResource();
		
	}
    
	return CORE_OK;
}

void DeepShadowMap::finalRelease()
{
	SAFE_RELEASE(startElementBufSRV);
	SAFE_RELEASE(startElementBufUAV);
	SAFE_RELEASE(startElementBuf);
	SAFE_RELEASE(linkedListBufSRV);
	SAFE_RELEASE(linkedListBufUAV);
	SAFE_RELEASE(linkedListBuf);
	SAFE_RELEASE(linkedListBuf2SRV);
	SAFE_RELEASE(linkedListBuf2UAV);
	SAFE_RELEASE(linkedListBuf2);
	SAFE_RELEASE(linkedListBuf3SRV);
	SAFE_RELEASE(linkedListBuf3UAV);
	SAFE_RELEASE(linkedListBuf3);
	for(int effect = 0; effect < NUM_EFFECTS; effect++)
		SAFE_RELEASE(dsmEffects[effect].dsmEffect);
}

void DeepShadowMap::Set(CoreCamera &lightCam, CoreMatrix4x4 &world, float alpha, int currentEffect)
{
	UINT val[4] = { -1, -1, -1, -1 };
	core->GetImmediateDeviceContext()->ClearUnorderedAccessViewUint(startElementBufUAV, val);
	core->SaveViewports();
	core->GetImmediateDeviceContext()->OMSetRenderTargets(0, NULL, NULL);
	core->GetImmediateDeviceContext()->RSSetViewports(1, &vp);
	lightCam.WorldViewProjectionToEffectVariable(dsmEffects[currentEffect].dsmEffectWorldViewProj, world);
	dsmEffects[currentEffect].dsmEffectLinkedListBufDAN->SetUnorderedAccessView(linkedListBufUAV);
	dsmEffects[currentEffect].dsmEffectDimension->SetInt((int)vp.Width);
	dsmEffects[currentEffect].dsmEffectStartElementBuf->SetUnorderedAccessView(startElementBufUAV);
	dsmEffects[currentEffect].dsmEffectAlpha->SetFloat(alpha);

	dsmEffects[currentEffect].dsmRenderTechnique->GetPassByIndex(0)->Apply(0, core->GetImmediateDeviceContext(), true);
}

void DeepShadowMap::ChangeAlpha(float alpha, int currentEffect)
{
	dsmEffects[currentEffect].dsmEffectAlpha->SetFloat(alpha);
	dsmEffects[currentEffect].dsmRenderTechnique->GetPassByIndex(0)->Apply(0, core->GetImmediateDeviceContext(), false);
}

void DeepShadowMap::ChangeLightCamera(CoreCamera &lightCam, CoreMatrix4x4 &world, int currentEffect)
{
	lightCam.WorldViewProjectionToEffectVariable(dsmEffects[currentEffect].dsmEffectWorldViewProj, world);
	dsmEffects[currentEffect].dsmRenderTechnique->GetPassByIndex(0)->Apply(0, core->GetImmediateDeviceContext(), false);
}

void DeepShadowMap::Unset(int currentEffect)
{
	core->SetDefaultRenderTarget();
	core->RestoreViewports();
	dsmEffects[currentEffect].dsmEffectLinkedListBufDAN->SetUnorderedAccessView(NULL);
}

void DeepShadowMap::SetShaderForRealRendering(ID3DX11EffectScalarVariable *effectDimension, ID3DX11EffectShaderResourceVariable *effectLinkedListBufWPRO, ID3DX11EffectShaderResourceVariable *effectNeighborsBufRO, ID3DX11EffectShaderResourceVariable *effectStartElementBuf)
{
	effectDimension->SetInt((int)vp.Width);
	effectStartElementBuf->SetResource(startElementBufSRV);
	effectLinkedListBufWPRO->SetResource(linkedListBuf2SRV);
	effectNeighborsBufRO->SetResource(linkedListBuf3SRV);
}

void DeepShadowMap::UnsetShaderForRealRendering(ID3DX11EffectScalarVariable *effectDimension, ID3DX11EffectShaderResourceVariable *effectLinkedListBufWPRO, ID3DX11EffectShaderResourceVariable *effectNeighborsBufRO, ID3DX11EffectShaderResourceVariable *effectStartElementBuf)
{
	effectStartElementBuf->SetResource(NULL);
	effectLinkedListBufWPRO->SetResource(NULL);
	effectNeighborsBufRO->SetResource(NULL);
}

void DeepShadowMap::SortLists(int currentEffect)
{
	dsmEffects[currentEffect].dsmEffectLinkedListBufWP->SetUnorderedAccessView(linkedListBuf2UAV);
	dsmEffects[currentEffect].dsmEffectStartElementBufRO->SetResource(startElementBufSRV);
	dsmEffects[currentEffect].dsmEffectLinkedListBufDANRO->SetResource(linkedListBufSRV);
	dsmEffects[currentEffect].dsmEffectDimension->SetInt((int)vp.Width);
	dsmEffects[currentEffect].dsmSortTechnique->GetPassByIndex(0)->Apply(0, core->GetImmediateDeviceContext(), true);
	int resx = (int)vp.Width / 16;
	if((int)vp.Width % 16 > 0)
		resx++;
	int resy = (int)vp.Width / 8;
	if((int)vp.Width % 8 > 0)
		resy++;
	core->GetImmediateDeviceContext()->Dispatch(resx, resy, 1);
	dsmEffects[currentEffect].dsmEffectLinkedListBufWP->SetUnorderedAccessView(NULL);
	dsmEffects[currentEffect].dsmEffectStartElementBufRO->SetResource(NULL);
	dsmEffects[currentEffect].dsmEffectLinkedListBufDANRO->SetResource(NULL);
	dsmEffects[currentEffect].dsmSortTechnique->GetPassByIndex(0)->Apply(0, core->GetImmediateDeviceContext(), true);
}

void DeepShadowMap::LinkLists(int currentEffect)
{
	dsmEffects[currentEffect].dsmEffectNeighborsBuf->SetUnorderedAccessView(linkedListBuf3UAV);
	dsmEffects[currentEffect].dsmEffectLinkedListBufWPRO->SetResource(linkedListBuf2SRV);
	dsmEffects[currentEffect].dsmEffectStartElementBufRO->SetResource(startElementBufSRV);
	dsmEffects[currentEffect].dsmEffectDimension->SetInt((int)vp.Width);
	dsmEffects[currentEffect].dsmLinkTechnique->GetPassByIndex(0)->Apply(0, core->GetImmediateDeviceContext(), true);
	int resx = (int)vp.Width / 16;
	if((int)vp.Width % 16 > 0)
		resx++;
	int resy = (int)vp.Width / 8;
	if((int)vp.Width % 8 > 0)
		resy++;
	core->GetImmediateDeviceContext()->Dispatch(resx, resy, 1);
	dsmEffects[currentEffect].dsmEffectNeighborsBuf->SetUnorderedAccessView(NULL);
	dsmEffects[currentEffect].dsmEffectStartElementBufRO->SetResource(NULL);
	dsmEffects[currentEffect].dsmEffectLinkedListBufWPRO->SetResource(NULL);
	dsmEffects[currentEffect].dsmLinkTechnique->GetPassByIndex(0)->Apply(0, core->GetImmediateDeviceContext(), true);
}
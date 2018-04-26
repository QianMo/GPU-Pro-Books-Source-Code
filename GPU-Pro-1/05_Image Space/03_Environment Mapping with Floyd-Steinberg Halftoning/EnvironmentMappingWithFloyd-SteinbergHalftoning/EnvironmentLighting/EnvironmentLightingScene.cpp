#include "DXUT.h"
#include "EnvironmentLightingScene.h"
#include "GeometryLoader.h"

EnvironmentLightingScene::EnvironmentLightingScene(ID3D10Device* device)
{
	this->device = device;

	camera.SetViewParams(
		&D3DXVECTOR3(10, 10 ,10),
		&D3DXVECTOR3(0, 0 ,0));
}

void EnvironmentLightingScene::loadEffectPool(const std::wstring& fileName)
{
	ID3D10Blob* compilationErrors = NULL;
	if(FAILED(
		D3DX10CreateEffectPoolFromFileW(fileName.c_str(), NULL, NULL, "fx_4_0", 0, 0, device, NULL, &effectPool, &compilationErrors, NULL)))
	{
		if(!compilationErrors)
			exit(-1); // Effect pool file not found.
		else
			MessageBoxA( NULL, (LPSTR)compilationErrors->GetBufferPointer(), "Failed to load effect file!", MB_OK);
		exit(-1);
	}
	effects[L"pool"] = effectPool->AsEffect();

}

void EnvironmentLightingScene::processMessage(HWND hWnd,  UINT uMsg, WPARAM wParam, LPARAM lParam, bool* trapped)
{
	camera.HandleMessages(hWnd, uMsg, wParam, lParam);
}

void EnvironmentLightingScene::animate(double dt, double t)
{
	camera.FrameMove(dt);
}

HRESULT EnvironmentLightingScene::releaseResources()
{

	unsigned int nDirectionalLights = directionalLightList.size();
	if(nDirectionalLights)
	{
		directionalLightBuffer->Release();

		depthStencilTextureArray->Release();
		shadowMapTextureArray->Release();
		shadowMapSRV->Release();
		shadowMapDSV->Release();
		shadowMapRTV->Release();
	}

	envMapCubeSRV->Release();
	quadInputLayout->Release();
	shadowInputLayout->Release();
	inputLayout->Release();
	meshDirectory.releaseAll();
	effects.releaseAll();
	return S_OK;
}

HRESULT EnvironmentLightingScene::createResources()
{
	loadEffectPool(L"effect/enginePool.fx");
	loadChildEffect(L"effect/standard.fx", L"standard");
	loadChildEffect(L"effect/textured.fx", L"textured");
	loadChildEffect(L"effect/shadow.fx", L"shadow");

	D3DX10CreateShaderResourceViewFromFile( device, L"media/glacier_cube.dds", NULL, NULL, &envMapCubeSRV, NULL);
	effects[L"pool"]->GetVariableByName("envMap")->AsShaderResource()->SetResource(envMapCubeSRV);

	ID3DX10Mesh* mesh;

	const D3D10_INPUT_ELEMENT_DESC quadElements[] =
	{
		{ "POSITION", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 0, D3D10_INPUT_PER_VERTEX_DATA, 0 },
		{ "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 16, D3D10_INPUT_PER_VERTEX_DATA, 0 }
	};

	D3DX10CreateMesh(device, quadElements, 2, "POSITION", 4, 2, 0, &mesh);

	struct QuadVertex
	{
		D3DXVECTOR4 pos;
		D3DXVECTOR2 tex;
	} svQuad[4];
	static const float fSize = 1.0f;
	svQuad[0].pos = D3DXVECTOR4(-fSize, fSize, 0.0f, 1.0f);
	svQuad[0].tex = D3DXVECTOR2(0.0f, 0.0f);
	svQuad[1].pos = D3DXVECTOR4(fSize, fSize, 0.0f, 1.0f);
	svQuad[1].tex = D3DXVECTOR2(1.0f, 0.0f);
	svQuad[2].pos = D3DXVECTOR4(-fSize, -fSize, 0.0f, 1.0f);
	svQuad[2].tex = D3DXVECTOR2(0.0f, 1.0f);
	svQuad[3].pos = D3DXVECTOR4(fSize, -fSize, 0.0f, 1.0f);
	svQuad[3].tex = D3DXVECTOR2(1.0f, 1.0f);

	mesh->SetVertexData(0, (void*)svQuad);

	unsigned short siQuad[6];
	siQuad[0] = 0;
	siQuad[1] = 1;
	siQuad[2] = 2;
	siQuad[3] = 2;
	siQuad[4] = 1;
	siQuad[5] = 3;

	mesh->SetIndexData(siQuad, 6);
	mesh->CommitToDevice();
	meshDirectory[L"quad"] = mesh;

	{
		const D3D10_INPUT_ELEMENT_DESC* elements;
		unsigned int nElements;
		mesh->GetVertexDescription(&elements, &nElements);
		D3D10_PASS_DESC passDesc;
		effects[L"shadow"]->GetTechniqueByName("showLights")->GetPassByIndex(0)->GetDesc(&passDesc);
		device->CreateInputLayout(
			elements, nElements,
			passDesc.pIAInputSignature,
			passDesc.IAInputSignatureSize,
			&quadInputLayout);
	}


	GeometryLoader::LoadMeshFromFile(L"media/bigbunny.dgb", device, &mesh);
	meshDirectory[L"bunny"] = mesh;

	{
		const D3D10_INPUT_ELEMENT_DESC* elements;
		unsigned int nElements;
		mesh->GetVertexDescription(&elements, &nElements);
		D3D10_PASS_DESC passDesc;
		effects[L"shadow"]->GetTechniqueByName("environmentLighted")->GetPassByIndex(0)->GetDesc(&passDesc);
		device->CreateInputLayout(
			elements, nElements,
			passDesc.pIAInputSignature,
			passDesc.IAInputSignatureSize,
			&inputLayout);
	}

	{
		const D3D10_INPUT_ELEMENT_DESC* elements;
		unsigned int nElements;
		mesh->GetVertexDescription(&elements, &nElements);
		D3D10_PASS_DESC passDesc;
		effects[L"shadow"]->GetTechniqueByName("toDirectionalShadowMap")->GetPassByIndex(0)->GetDesc(&passDesc);
		device->CreateInputLayout(
			elements, nElements,
			passDesc.pIAInputSignature,
			passDesc.IAInputSignatureSize,
			&shadowInputLayout);
	}


	sceneCenter = D3DXVECTOR3(0, 0, 0);
	sceneRadius = 10.5;

	shadowMapHeight = 128;
	shadowMapWidth = 128;
	ID3D10ShaderResourceView* envSRV;
	D3DX10_IMAGE_LOAD_INFO loadInfo;
		loadInfo.Width = D3DX10_DEFAULT;
		loadInfo.Height = D3DX10_DEFAULT;
		loadInfo.Depth = D3DX10_DEFAULT;
		loadInfo.FirstMipLevel = D3DX10_DEFAULT;
		loadInfo.Usage = D3D10_USAGE_IMMUTABLE;
		loadInfo.BindFlags = D3D10_BIND_SHADER_RESOURCE;
		loadInfo.CpuAccessFlags = 0;
		loadInfo.MiscFlags = D3DX10_DEFAULT;
		loadInfo.Format = DXGI_FORMAT_FROM_FILE;
		loadInfo.Filter = D3DX10_DEFAULT;
		loadInfo.MipFilter = D3DX10_DEFAULT;
		loadInfo.MipLevels = D3DX_DEFAULT;
		loadInfo.Format = DXGI_FORMAT_R32G32B32A32_FLOAT; // important
	D3DX10CreateShaderResourceViewFromFile(
		device, L"media/glacier.dds", &loadInfo, NULL, &envSRV, NULL);
	ID3D10Resource* envResource;
	envSRV->GetResource(&envResource);
	ID3D10Texture2D* envTexture = (ID3D10Texture2D*)envResource;

	envTexture->GetDesc(&tDesc);
	tDesc.Usage = D3D10_USAGE_STAGING;
	tDesc.BindFlags = 0;
	tDesc.CPUAccessFlags = D3D10_CPU_ACCESS_READ;

	device->CreateTexture2D(&tDesc, NULL, &sysEnvTexture);
	device->CopyResource(sysEnvTexture, envTexture);

	samplePhiThetaErrorDiffusion();

	envResource->Release();
	sysEnvTexture->Release();
	envSRV->Release();

	unsigned int nDirectionalLights = directionalLightList.size();

	if(nDirectionalLights)
	{
		D3D10_BUFFER_DESC directionalLightBufferDesc;
		directionalLightBufferDesc.BindFlags = D3D10_BIND_CONSTANT_BUFFER;
		directionalLightBufferDesc.ByteWidth = 512/*nDirectionalLights*/ * sizeof(DirectionalLight) + 4 * sizeof(unsigned int);
		directionalLightBufferDesc.CPUAccessFlags = D3D10_CPU_ACCESS_WRITE;
		directionalLightBufferDesc.MiscFlags = 0;
		directionalLightBufferDesc.Usage = D3D10_USAGE_DYNAMIC;

		device->CreateBuffer(&directionalLightBufferDesc, NULL, &directionalLightBuffer);

		D3D10_TEXTURE2D_DESC shadowMapTextureArrayDesc;
		shadowMapTextureArrayDesc.ArraySize = nDirectionalLights;
		shadowMapTextureArrayDesc.BindFlags = D3D10_BIND_RENDER_TARGET | D3D10_BIND_SHADER_RESOURCE;
		shadowMapTextureArrayDesc.CPUAccessFlags = 0;
		shadowMapTextureArrayDesc.Format = DXGI_FORMAT_R32_FLOAT;
		shadowMapTextureArrayDesc.Height = shadowMapHeight;
		shadowMapTextureArrayDesc.MipLevels = 1;
		shadowMapTextureArrayDesc.MiscFlags = 0;
		shadowMapTextureArrayDesc.SampleDesc.Count = 1;
		shadowMapTextureArrayDesc.SampleDesc.Quality = 0;
		shadowMapTextureArrayDesc.Usage = D3D10_USAGE_DEFAULT;
		shadowMapTextureArrayDesc.Width = shadowMapWidth;

		device->CreateTexture2D(&shadowMapTextureArrayDesc, NULL, &shadowMapTextureArray);

		D3D10_TEXTURE2D_DESC depthStencilTextureArrayDesc;
		depthStencilTextureArrayDesc.ArraySize = nDirectionalLights;
		depthStencilTextureArrayDesc.BindFlags = D3D10_BIND_DEPTH_STENCIL ;
		depthStencilTextureArrayDesc.CPUAccessFlags = 0;
		depthStencilTextureArrayDesc.Format = DXGI_FORMAT_D32_FLOAT;
		depthStencilTextureArrayDesc.Height = shadowMapHeight;
		depthStencilTextureArrayDesc.MipLevels = 1;
		depthStencilTextureArrayDesc.MiscFlags = 0;
		depthStencilTextureArrayDesc.SampleDesc.Count = 1;
		depthStencilTextureArrayDesc.SampleDesc.Quality = 0;
		depthStencilTextureArrayDesc.Usage = D3D10_USAGE_DEFAULT;
		depthStencilTextureArrayDesc.Width = shadowMapWidth;

		device->CreateTexture2D(&depthStencilTextureArrayDesc, NULL, &depthStencilTextureArray);

		D3D10_SHADER_RESOURCE_VIEW_DESC shadowMapSRVDesc;
		shadowMapSRVDesc.ViewDimension =  D3D10_SRV_DIMENSION_TEXTURE2DARRAY;
		shadowMapSRVDesc.Texture2DArray.ArraySize = nDirectionalLights;
		shadowMapSRVDesc.Texture2DArray.FirstArraySlice = 0;
		shadowMapSRVDesc.Texture2DArray.MipLevels = 1;
		shadowMapSRVDesc.Texture2DArray.MostDetailedMip = 0;
		shadowMapSRVDesc.Format = shadowMapTextureArrayDesc.Format;

		device->CreateShaderResourceView(shadowMapTextureArray, &shadowMapSRVDesc, &shadowMapSRV);

		D3D10_RENDER_TARGET_VIEW_DESC shadowMapRTVDesc;
		shadowMapRTVDesc.ViewDimension =  D3D10_RTV_DIMENSION_TEXTURE2DARRAY;
		shadowMapRTVDesc.Texture2DArray.ArraySize = nDirectionalLights;
		shadowMapRTVDesc.Texture2DArray.FirstArraySlice = 0;
		shadowMapRTVDesc.Texture2DArray.MipSlice = 0;
		shadowMapRTVDesc.Format = shadowMapTextureArrayDesc.Format;

		device->CreateRenderTargetView(shadowMapTextureArray, &shadowMapRTVDesc, &shadowMapRTV);

		D3D10_DEPTH_STENCIL_VIEW_DESC shadowMapDSVDesc;
		shadowMapDSVDesc.ViewDimension =  D3D10_DSV_DIMENSION_TEXTURE2DARRAY;
		shadowMapDSVDesc.Texture2DArray.ArraySize = nDirectionalLights;
		shadowMapDSVDesc.Texture2DArray.FirstArraySlice = 0;
		shadowMapDSVDesc.Texture2DArray.MipSlice = 0;
		shadowMapDSVDesc.Format = depthStencilTextureArrayDesc.Format;

		device->CreateDepthStencilView(depthStencilTextureArray, &shadowMapDSVDesc, &shadowMapDSV);
	}

	return S_OK;
}


void EnvironmentLightingScene::render()
{
	D3DXMATRIX modelMatrix;
	D3DXMatrixIdentity(&modelMatrix);
	effects[L"pool"]->GetVariableByName("modelMatrix")->AsMatrix()->SetMatrix((float*)&modelMatrix);
	effects[L"pool"]->GetVariableByName("modelMatrixInverse")->AsMatrix()->SetMatrix((float*)&modelMatrix);
	D3DXMATRIX modelViewProjMatrix;
	modelViewProjMatrix = modelMatrix * *camera.GetViewMatrix() * *camera.GetProjMatrix();
	effects[L"pool"]->GetVariableByName("modelViewProjMatrix")->AsMatrix()->SetMatrix((float*)&modelViewProjMatrix);

	D3DXMATRIX vpm;
	const D3DXVECTOR3& eyePosition = *camera.GetEyePt();
	D3DXMATRIX eyePosTranslationMatrix;
	D3DXMatrixTranslation(&eyePosTranslationMatrix, eyePosition.x, eyePosition.y, eyePosition.z);
	D3DXMatrixInverse(&vpm, NULL, &(eyePosTranslationMatrix * *camera.GetViewMatrix() * *camera.GetProjMatrix()));
	effects[L"pool"]->GetVariableByName("orientProjMatrixInverse")->AsMatrix()->SetMatrix((float*)&vpm);
	effects[L"pool"]->GetVariableByName("eyePosition")->AsVector()->SetFloatVector((float*)&eyePosition);

	effects[L"shadow"]->GetConstantBufferByName("spotlightBuffer")->SetConstantBuffer(directionalLightBuffer);

	static bool firstFrame = true;
	if(firstFrame)
	{
		renderShadowMaps();
		firstFrame = false;
	}
	device->OMSetRenderTargets(1, &swapChainRenderTargetView, swapChainDepthStencilView);
	D3D10_VIEWPORT vp;
	vp.Height = viewportHeight;
	vp.Width = viewportWidth;
	vp.MinDepth = 0;
	vp.MaxDepth = 1;
	vp.TopLeftX = 0;
	vp.TopLeftY = 0;
	device->RSSetViewports(1, &vp);

	effects[L"shadow"]->GetVariableByName("shadowMapArray")->AsShaderResource()->SetResource(shadowMapSRV);

	device->ClearRenderTargetView(swapChainRenderTargetView, D3DXVECTOR4(0.5, 0, 0, 1));
	device->ClearDepthStencilView(swapChainDepthStencilView, D3D10_CLEAR_DEPTH, 1.0f, 0);

	effects[L"shadow"]->GetTechniqueByName("environmentLighted")->GetPassByIndex(0)->Apply(0);

	device->IASetInputLayout(inputLayout);
	device->IASetPrimitiveTopology(D3D10_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
	meshDirectory[L"bunny"]->DrawSubset(0);

	device->IASetInputLayout(quadInputLayout);
	device->IASetPrimitiveTopology(D3D10_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
	effects[L"shadow"]->GetTechniqueByName("showLights")->GetPassByIndex(0)->Apply(0);
	meshDirectory[L"quad"]->DrawSubset(0);


//      <render cue="showLightsQuad" cameraCue="main" role="background" />
  //    <render cue="stats" role="text" />
}

HRESULT EnvironmentLightingScene::createSwapChainResources()
{
	device->OMGetRenderTargets(1, &swapChainRenderTargetView, &swapChainDepthStencilView);
	ID3D10Texture2D* defaultRenderTargetResource;
	swapChainRenderTargetView->GetResource((ID3D10Resource**)&defaultRenderTargetResource);
	
	D3D10_TEXTURE2D_DESC defaultTexture2DDesc;
	defaultRenderTargetResource->GetDesc(&defaultTexture2DDesc);
	defaultRenderTargetResource->Release();
	viewportWidth = defaultTexture2DDesc.Width;
	viewportHeight =  defaultTexture2DDesc.Height;

	camera.SetProjParams(1.57f, viewportWidth / (float)viewportHeight, 0.1, 1000.0);

	return S_OK;
}

HRESULT EnvironmentLightingScene::releaseSwapChainResources()
{
	swapChainDepthStencilView->Release();
	swapChainRenderTargetView->Release();
	return S_OK;
}

void EnvironmentLightingScene::loadChildEffect(const std::wstring& fileName, const std::wstring& name)
{
	ID3D10Blob* compilationErrors;
	ID3D10Effect* childEffect;
	if(FAILED(
		D3DX10CreateEffectFromFileW(fileName.c_str(), NULL, NULL, "fx_4_0", 0, D3D10_EFFECT_COMPILE_CHILD_EFFECT,
		device, effectPool, NULL, &childEffect, &compilationErrors, NULL)))
	{
		if(!compilationErrors)
			exit(-1); // Child effect file not found
		else
			MessageBoxA( NULL, (LPSTR)compilationErrors->GetBufferPointer(), "Failed to load effect file!", MB_OK);
		exit(-1); // TODO: CLEAN EXIT
	}
	if(childEffect)
	{
		if(effects[name] == NULL)
			effects[name] = childEffect;
		else
			exit(-1); // duplicate child effect name
	}
}

ID3D10EffectTechnique* EnvironmentLightingScene::getTechnique(const std::wstring& effectName, const std::string& techniqueName)
{
	ID3D10Effect* effect = getEffect(effectName);
	if(!effect)
		exit(-1);
	ID3D10EffectTechnique* technique = effect->GetTechniqueByName(techniqueName.c_str());
	if(!technique)
		exit(-1);
	return technique;
}

void EnvironmentLightingScene::samplePhiTheta()
{
	srand (8111634 );

	D3D10_MAPPED_TEXTURE2D pix;
	sysEnvTexture->Map(0, D3D10_MAP_READ, 0, &pix);
	D3DXVECTOR4* hdri = (D3DXVECTOR4*)pix.pData;
	unsigned int pitch = pix.RowPitch / sizeof(D3DXVECTOR4);
	float theta = 0;
	for(unsigned int j=0; j<tDesc.Height; j++, theta += D3DX_PI / tDesc.Height)
	{
		D3DXVECTOR4* rowi = hdri;
		float phi = 0;
		for(unsigned int i=0; i<tDesc.Width; i++, phi += 2.0 * D3DX_PI / tDesc.Width)
		{
			float geomFactor = sin(theta);
			float rr = ((double)rand() / RAND_MAX) * 5470;
			if(geomFactor * hdri->z > rr + 1e-35f)
			{
				DirectionalLight d;
				D3DXVECTOR3 direction = -D3DXVECTOR3(
					cos(phi) * sin(theta),
					cos(theta),
					-sin(phi) * sin(theta)
					);
				d.direction = D3DXVECTOR4(direction.x, direction.y, direction.z, 0);
				d.radiance = D3DXVECTOR4(hdri->x, hdri->y, hdri->z, 0) / hdri->z * 0.01;
				if(hdri->z < 0.001)
					bool breki = true;
				D3DXMATRIX lightViewMatrix, lightProjMatrix;

				D3DXMatrixLookAtLH(&lightViewMatrix, &(sceneCenter - direction * sceneRadius), &sceneCenter, &D3DXVECTOR3(1, 0, 0));

				D3DXMatrixOrthoLH(&lightProjMatrix, 2*sceneRadius, 2*sceneRadius, 0, 2*sceneRadius);

				D3DXMatrixMultiply(&d.lightViewProjMatrix, &lightViewMatrix, &lightProjMatrix);
				D3DXMatrixTranspose(&d.lightViewProjMatrix, &d.lightViewProjMatrix);
				directionalLightList.push_back(d);
			}
			hdri++;
		}
		hdri = rowi + pitch;
	}

	sysEnvTexture->Unmap(0);
}

void EnvironmentLightingScene::samplePhiThetaErrorDiffusion()
{
	D3D10_MAPPED_TEXTURE2D pix;
	sysEnvTexture->Map(0, D3D10_MAP_READ, 0, &pix);
	D3DXVECTOR4* hdri = (D3DXVECTOR4*)pix.pData;
	unsigned int pitch = pix.RowPitch / sizeof(D3DXVECTOR4);

	float treshold = 440;
//	float treshold = 2440;
	float slideCorrector = 1;
	float carry = 0;
	float carryDiagonal = 0;
	float* carryRow = new float[tDesc.Width];
	for(unsigned int ki=0; ki<tDesc.Width; ki++)
		carryRow[ki] = 0;
	float theta = 0;
	for(unsigned int j=0; j<tDesc.Height; j++, theta += D3DX_PI / tDesc.Height)
	{
		float nextWeight = 8.0 / 8.0 - sin(theta) * 3.0/8.0 * slideCorrector;
		float lowWeight = 0.0 / 8.0 + sin(theta) * 2.0 / 8.0 * slideCorrector;
		float diaWeight = 0.0 / 8.0 +  sin(theta) * 1.0 / 8.0 * slideCorrector;

		D3DXVECTOR4* rowi = hdri;
		float phi = 0;
		for(unsigned int i=0; i<tDesc.Width; i++, phi += 2.0 * D3DX_PI / tDesc.Width)
		{
			float geomFactor = sin(theta);
			float pimportance = geomFactor * hdri->z;


			float importance = pimportance + carry + carryRow[i];

			if(importance > treshold * 0.07 && pimportance > 1e-30f)
			{
				importance -= treshold;
				DirectionalLight d;
				D3DXVECTOR3 direction = -D3DXVECTOR3(
					cos(phi) * sin(theta),
					cos(theta),
					-sin(phi) * sin(theta)
					);
				d.direction = D3DXVECTOR4(direction.x, direction.y, direction.z, 0);
				d.radiance = D3DXVECTOR4(hdri->x, hdri->y, hdri->z, 0) / hdri->z * 0.01;
				D3DXMATRIX lightViewMatrix, lightProjMatrix;

				D3DXMatrixLookAtLH(&lightViewMatrix, &(sceneCenter - direction * sceneRadius), &sceneCenter, &D3DXVECTOR3(1, 0, 0));

				D3DXMatrixOrthoLH(&lightProjMatrix, 2*sceneRadius, 2*sceneRadius, 0, 2*sceneRadius);

				D3DXMatrixMultiply(&d.lightViewProjMatrix, &lightViewMatrix, &lightProjMatrix);
				D3DXMatrixTranspose(&d.lightViewProjMatrix, &d.lightViewProjMatrix);
				directionalLightList.push_back(d);
			}
			carry = importance * nextWeight;
			carryRow[i] = carryDiagonal + importance * lowWeight;
			carryDiagonal = importance * diaWeight;

			hdri++;
		}
		j++;
		theta += D3DX_PI / tDesc.Height;
		hdri--;
		hdri += pitch;

		for(int i=tDesc.Width-1; i>=0; i--, phi -= 2.0 * D3DX_PI / tDesc.Width)
		{
			if(j == tDesc.Height-1)
				bool brekke = true;
			float geomFactor = sin(theta);
			float pimportance = geomFactor * hdri->z;

			float importance = pimportance + carry + carryRow[i];

			if(importance > treshold * 0.07 && pimportance > 1e-30f)
			{
				importance -= treshold;
				DirectionalLight d;
				D3DXVECTOR3 direction = -D3DXVECTOR3(
					cos(phi) * sin(theta),
					cos(theta),
					-sin(phi) * sin(theta)
					);
				d.direction = D3DXVECTOR4(direction.x, direction.y, direction.z, 0);
				d.radiance = D3DXVECTOR4(hdri->x, hdri->y, hdri->z, 0) / hdri->z * 0.01;
				D3DXMATRIX lightViewMatrix, lightProjMatrix;

				D3DXMatrixLookAtLH(&lightViewMatrix, &(sceneCenter - direction * sceneRadius), &sceneCenter, &D3DXVECTOR3(1, 0, 0));

				D3DXMatrixOrthoLH(&lightProjMatrix, 2*sceneRadius, 2*sceneRadius, 0, 2*sceneRadius);

				D3DXMatrixMultiply(&d.lightViewProjMatrix, &lightViewMatrix, &lightProjMatrix);
				D3DXMatrixTranspose(&d.lightViewProjMatrix, &d.lightViewProjMatrix);
				directionalLightList.push_back(d);
			}
			carry = importance * nextWeight;
			carryRow[i] = carryDiagonal + importance * lowWeight;
			carryDiagonal = importance * diaWeight;

			hdri--;
		}
		hdri++;
		hdri += pitch;
	}
	
	delete carryRow;

	sysEnvTexture->Unmap(D3D10CalcSubresource(0, 0, tDesc.MipLevels));
}

void EnvironmentLightingScene::renderShadowMaps()
{
	DirectionalLight* directionalLights;
	directionalLightBuffer->Map(D3D10_MAP_WRITE_DISCARD, 0, (void**)&directionalLights);
	((unsigned int*)directionalLights)[0] = directionalLightList.size();
	directionalLights = (DirectionalLight*)(((unsigned int*)directionalLights) + 4);
	DirectionalLightList::iterator i = directionalLightList.begin();
	while(i != directionalLightList.end())
	{
		*directionalLights = *i;
		directionalLights++;
		i++;
	}
	directionalLightBuffer->Unmap();

	device->OMSetRenderTargets(1, &shadowMapRTV, shadowMapDSV);
	D3D10_VIEWPORT vp;
	vp.Height = shadowMapHeight;
	vp.Width = shadowMapWidth;
	vp.MinDepth = 0;
	vp.MaxDepth = 1;
	vp.TopLeftX = 0;
	vp.TopLeftY = 0;
	device->RSSetViewports(1, &vp);
	float clearColor[4] = {12, 0, 0, 0};
	device->ClearRenderTargetView(shadowMapRTV, clearColor);
	device->ClearDepthStencilView(shadowMapDSV, D3D10_CLEAR_DEPTH, 1.0f, 0);

	D3DXMATRIX id;
	D3DXMatrixIdentity(&id);

	effects[L"shadow"]->GetTechniqueByName("toDirectionalShadowMap")->GetPassByIndex(0)->Apply(0);

	device->IASetInputLayout(shadowInputLayout);
	device->IASetPrimitiveTopology(D3D10_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
	meshDirectory[L"bunny"]->DrawSubsetInstanced(0, directionalLightList.size(), 0);
}
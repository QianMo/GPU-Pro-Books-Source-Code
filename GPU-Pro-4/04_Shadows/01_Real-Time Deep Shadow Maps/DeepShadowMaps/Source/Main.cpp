#include "Main.h"
#include "MemoryLeakTracker.h"
#include "ModelLoader.h"
#include <fstream>

void CALLBACK OnGUIEventFunc(UINT nEvent, int nControlID, CDXUTControl *pControl, void *pUserContext);

// Initialize everything
CoreResult Main::init(HWND hwnd, bool bFullScreen)
{
	CoreLog::InitWinDebug();
	//CoreLog::Init(L"Debug");

	swapChain = NULL;
	dsm = NULL;
	for(int effect = 0; effect < NUM_EFFECTS; effect++)
		hairEffects[effect].hairEffect = NULL;

	this->hwnd = hwnd;
	leftMouseButtonDown = false;
	rightMouseButtonDown = false;
	leftMouseButtonTrigger = false;
	deferedPosInLight = NULL;
	deferedPosInWorld = NULL;
	deferedNormal = NULL;
	deferedColor = NULL;
	deferedTempColor = NULL;
	deferedPosInLightRTV = NULL;
	deferedPosInWorldRTV = NULL;
	deferedNormalRTV = NULL;
	deferedColorRTV = NULL;
	deferedTempColorRTV = NULL;
	deferedPosInLightSRV = NULL;
	deferedPosInWorldSRV = NULL;
	deferedNormalSRV = NULL;
	deferedColorSRV = NULL;
	deferedTempColorSRV = NULL;
	bufQuad = NULL;
	layoutQuad = NULL;
	deferedRenderTargets = NULL;
	cameraMode = CAMERA_MAIN;
	hairModel = new Model();
	lightHeight = 300.0f;
	lightAngle = 0;
	orbitMode = false;
	currentEffect = EFFECT_DEEPSHADOWMAP;

	if(bFullScreen)
		ShowCursor(false);

	CoreResult result = CreateCore(hwnd, 1, D3D_DRIVER_TYPE_HARDWARE, BACKBUFFER_FORMAT, 60, 1, 1, 0, !bFullScreen, NULL, &core);
	if(result != CORE_OK) return result;

	guiDialog.Init(&dialogResourceManager);

	guiDialog.SetCallback(OnGUIEventFunc, this);

	D3DCOLOR textCol = 0xFF000000;

	guiDialog.AddComboBox(IDC_COMBOBOX_SHADOWMODE, 0, 10, 300, 30, 0, true, &comboBoxShadowMode);
	comboBoxShadowMode->SetTextColor(textCol);

	comboBoxShadowMode->AddItem(L"Shading", NULL);
	comboBoxShadowMode->AddItem(L"Shadow Map", NULL);
	comboBoxShadowMode->AddItem(L"Deep Shadow Map", NULL);
	comboBoxShadowMode->AddItem(L"Exponential Deep Shadow Map", NULL);
	comboBoxShadowMode->SetSelectedByIndex(currentEffect);
	
	
	guiDialog.AddComboBox(IDC_COMBOBOX_CAMERACONTROL, 0, 50, 300, 30, 0, true, &comboBoxCameraControl);
	comboBoxCameraControl->SetTextColor(textCol);

	comboBoxCameraControl->AddItem(L"Control Viewer Camera", NULL);
	comboBoxCameraControl->AddItem(L"Control Light Camera", NULL);

	CDXUTStatic *dxutStaticTemp;

	guiDialog.AddStatic(IDC_SLIDER_HAIR_ALPHA_TEXT, L"Hair Transparency", 0, 90, 300, 20, false, &dxutStaticTemp);
	dxutStaticTemp->SetTextColor(textCol);
	guiDialog.AddSlider(IDC_SLIDER_HAIR_ALPHA, 0, 110, 280, 20, 1, 100, 50, false, &sliderHairAlpha);

	guiDialog.AddButton(IDC_BUTTON_HAIR_COLOR, L"Hair Color", 40, 140, 180, 30, 0, false, &buttonHairColor);
	setDXUTFontColor(textCol, buttonHairColor);

	guiDialog.AddStatic(IDC_ROT_BUTTON_TEXT, L"Head Rotation", 0, 180, 300, 20, false, &dxutStaticTemp);
	dxutStaticTemp->SetTextColor(textCol);
	guiDialog.AddButton(IDC_BUTTON_ROT_UP, L"^", 130, 200, 30, 30, 0, false, &buttonRotUp);
	setDXUTFontColor(textCol, buttonRotUp);
	guiDialog.AddButton(IDC_BUTTON_ROT_LEFT, L"<", 90, 230, 30, 30, 0, false, &buttonRotLeft);
	setDXUTFontColor(textCol, buttonRotLeft);
	guiDialog.AddButton(IDC_BUTTON_ROT_RIGHT, L">", 170, 230, 30, 30, 0, false, &buttonRotRight);
	setDXUTFontColor(textCol, buttonRotRight);
	guiDialog.AddButton(IDC_BUTTON_ROT_DOWN, L"v", 130, 260, 30, 30, 0, false, &buttonRotDown);
	setDXUTFontColor(textCol, buttonRotDown);

	guiDialog.AddCheckBox(IDC_CHECKBOX_ORBIT, L"Orbit Mode (Light Source)", 0, 300, 200, 20, false, 0, false, &cbOrbit);
	setDXUTFontColor(textCol, cbOrbit);

	guiDialog.AddButton(IDC_BUTTON_ORBIT_LIGHT_ROT_PLUS, L"-", 150, 340, 20, 20, 0, false, &buttonOrbitLightRotMinus);
	setDXUTFontColor(textCol, buttonOrbitLightRotMinus);
	buttonOrbitLightRotMinus->SetEnabled(false);
	
	guiDialog.AddButton(IDC_BUTTON_ORBIT_LIGHT_ROT_PLUS, L"+", 180, 340, 20, 20, 0, false, &buttonOrbitLightRotPlus);
	setDXUTFontColor(textCol, buttonOrbitLightRotPlus);
	buttonOrbitLightRotPlus->SetEnabled(false);

	guiDialog.AddStatic(IDC_ORBIT_LIGHT_ROT_BUTTON_TEXT, L"Light Rotation", 0, 340, 300, 20, false, &dxutStaticTemp);
	dxutStaticTemp->SetTextColor(textCol);

	guiDialog.AddButton(IDC_BUTTON_ORBIT_LIGHT_HEIGHT_PLUS, L"-", 150, 370, 20, 20, 0, false, &buttonOrbitLightHeightMinus);
	setDXUTFontColor(textCol, buttonOrbitLightHeightMinus);
	buttonOrbitLightHeightMinus->SetEnabled(false);
	
	guiDialog.AddButton(IDC_BUTTON_ORBIT_LIGHT_HEIGHT_PLUS, L"+", 180, 370, 20, 20, 0, false, &buttonOrbitLightHeightPlus);
	setDXUTFontColor(textCol, buttonOrbitLightHeightPlus);
	buttonOrbitLightHeightPlus->SetEnabled(false);

	guiDialog.AddStatic(IDC_ORBIT_LIGHT_HEIGHT_BUTTON_TEXT, L"Light Height", 0, 370, 300, 20, false, &dxutStaticTemp);
	dxutStaticTemp->SetTextColor(textCol);

	dialogResourceManager.OnD3D11CreateDevice(core->GetDevice(), core->GetImmediateDeviceContext());

	CoreTexture2D *backBuffer = core->GetBackBuffer();

	DXGI_SURFACE_DESC backBufferSurfaceDesc;
	backBufferSurfaceDesc.Height = backBuffer->GetHeight();
	backBufferSurfaceDesc.Width = backBuffer->GetWidth();
	backBufferSurfaceDesc.SampleDesc.Count = backBuffer->GetSampleCount();
	backBufferSurfaceDesc.SampleDesc.Quality = backBuffer->GetSampleQuality();
	backBufferSurfaceDesc.Format = backBuffer->GetFormat();

	dialogResourceManager.OnD3D11ResizedSwapChain(core->GetDevice(), &backBufferSurfaceDesc);

	guiDialog.SetLocation(backBuffer->GetWidth() - 320, 0);
	
	swapChain = core->GetSwapChain();

	D3D11_VIEWPORT vp;
	UINT n = 1;
	vp.Width = (float)backBuffer->GetWidth();
	vp.Height = (float)backBuffer->GetHeight();
	vp.MinDepth = 0.0f;
	vp.MaxDepth = 1.0f;
	vp.TopLeftX = 0;
	vp.TopLeftY = 0;
	core->GetImmediateDeviceContext()->RSSetViewports(1, &vp);		

	camera.SetProjectionPerspective(PI / 4, (float)backBuffer->GetWidth() / backBuffer->GetHeight(), NEAR_PLANE, FAR_PLANE);
	camera.SetView(CoreVector3 (100, 100, 100), CoreVector3(0, 0, 0), CoreVector3(0.0f,1.0f,0.0f));

	CoreResult cr;
	
	cr = LoadHairModel(core, std::wstring(L"HairModels\\straight.hair"), hairModel);
	if(cr != CORE_OK)
		return cr;

	sliderHairAlpha->SetValue((int)(hairModel->GetDiffuseColor().a * 100));

	cr = LoadXModel(core, std::wstring(L"Models\\woman.x"), &headModels, &numHeadModels);
	if(cr != CORE_OK)
		return cr;

	for(int effect = 0; effect < NUM_EFFECTS; effect++)
	{
		std::ifstream effectFile;
		effectFile.open(EffectPaths[effect] + L"\\Hair.hlsl");
		IncludeHandler *ih = new IncludeHandler(effect);
		ID3D10Blob *errors = NULL;
#ifdef _DEBUG
		cr = LoadEffectFromStream(core, effectFile, ih, D3D10_SHADER_DEBUG | D3D10_SHADER_SKIP_OPTIMIZATION, 0, &errors, &hairEffects[effect].hairEffect);
#else
		cr = LoadEffectFromStream(core, effectFile, ih, 0, 0, &errors, &hairEffects[effect].hairEffect);
#endif
		effectFile.close();
		SAFE_DELETE(ih);
		if(cr != CORE_OK && errors)
		{
			CoreLog::Information((char *)errors->GetBufferPointer());
			MessageBoxA(NULL, (char *)errors->GetBufferPointer(), NULL, 0);
			errors->Release();
			return cr;
		}
		else
			if(cr != CORE_OK) return cr;

		if(errors)
			errors->Release();
	
		hairEffects[effect].hairRenderTechnique = hairEffects[effect].hairEffect->GetTechniqueByName("Render");
		hairEffects[effect].hairRenderDeferedTechnique = hairEffects[effect].hairEffect->GetTechniqueByName("RenderDefered");
		hairEffects[effect].hairRenderAATechnique = hairEffects[effect].hairEffect->GetTechniqueByName("RenderAA");
		hairEffects[effect].hairRenderDeferedModelTechnique = hairEffects[effect].hairEffect->GetTechniqueByName("RenderDeferedModel");
		hairModel->CreateInputLayout(hairEffects[effect].hairRenderDeferedTechnique->GetPassByIndex(0), &hairEffects[effect].hairInputLayout);
		headModels[0]->CreateInputLayout(hairEffects[effect].hairRenderDeferedModelTechnique->GetPassByIndex(0), &hairEffects[effect].headInputLayout);	// one layout for all

		hairEffects[effect].hairEffectWorldViewProj = hairEffects[effect].hairEffect->GetVariableByName("WorldViewProj")->AsMatrix();
		hairEffects[effect].hairEffectWorld = hairEffects[effect].hairEffect->GetVariableByName("World")->AsMatrix();
		hairEffects[effect].hairEffectWorldViewProjLight = hairEffects[effect].hairEffect->GetVariableByName("WorldViewProjLight")->AsMatrix();
		hairEffects[effect].hairEffectLightPos = hairEffects[effect].hairEffect->GetVariableByName("LightPos")->AsVector();
		hairEffects[effect].hairEffectCameraPos = hairEffects[effect].hairEffect->GetVariableByName("CameraPos")->AsVector();
		hairEffects[effect].hairEffectColor = hairEffects[effect].hairEffect->GetVariableByName("Color")->AsVector();
		hairEffects[effect].hairEffectTexture = hairEffects[effect].hairEffect->GetVariableByName("Texture")->AsShaderResource();
		hairEffects[effect].hairEffectLinkedListBufWPRO = hairEffects[effect].hairEffect->GetVariableByName("LinkedListBufWPRO")->AsShaderResource();
		hairEffects[effect].hairEffectNeighborsBufRO = hairEffects[effect].hairEffect->GetVariableByName("NeighborsBufRO")->AsShaderResource();
		hairEffects[effect].hairEffectStartElementBuf = hairEffects[effect].hairEffect->GetVariableByName("StartElementBufRO")->AsShaderResource();
		hairEffects[effect].hairEffectDimension = hairEffects[effect].hairEffect->GetVariableByName("Dimension")->AsScalar();
		hairEffects[effect].hairEffectDeferedPosInLight = hairEffects[effect].hairEffect->GetVariableByName("DeferedPosInLight")->AsShaderResource();
		hairEffects[effect].hairEffectDeferedPosInWorld = hairEffects[effect].hairEffect->GetVariableByName("DeferedPosInWorld")->AsShaderResource();
		hairEffects[effect].hairEffectDeferedNormal = hairEffects[effect].hairEffect->GetVariableByName("DeferedNormal")->AsShaderResource();
		hairEffects[effect].hairEffectDeferedColor = hairEffects[effect].hairEffect->GetVariableByName("DeferedColor")->AsShaderResource();
		hairEffects[effect].hairEffectFrameBufferDimension = hairEffects[effect].hairEffect->GetVariableByName("FrameBufferDimension")->AsVector();
	}

	// Set light
	lightCamera.SetProjectionPerspective(PI / 4, 1.0f, NEAR_PLANE_LIGHT, FAR_PLANE_LIGHT);
	lightCamera.SetView(CoreVector3(250.0f, 260.0f, 0.0f), CoreVector3(0, 0, 0), CoreVector3(0.000f, 1.000f, 0.000f));	

	unsigned int size = 512;
	unsigned int memalloc = 50;
	std::fstream fs;
	fs.open("params.txt");
	if(fs.good())
	{
		std::string text;
		std::string line;
		getline (fs, line);
		sscanf(line.c_str(), "DeepShadowMapSize %d", &size);
		getline (fs, line);
		sscanf(line.c_str(), "BufferElementsMemAlloc  %d", &memalloc);
	}
	fs.close();
	dsm = new DeepShadowMap();
	cr = dsm->Init(core, size, size * size * memalloc);
	
	if(cr != CORE_OK)
		return cr;

	// Init defered shading
	cr = createDeferedShadingTextures();
	if(cr != CORE_OK)
		return cr;

	float fbDim[2];
	fbDim[0] = (float)backBuffer->GetWidth();
	fbDim[1] = (float)backBuffer->GetHeight();
	for(int effect = 0; effect < NUM_EFFECTS; effect++)
		hairEffects[effect].hairEffectFrameBufferDimension->SetFloatVector(fbDim);

	// create screen quad
	D3D11_INPUT_ELEMENT_DESC quadlayout[] =
    {
        { "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 },
        { "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 12, D3D11_INPUT_PER_VERTEX_DATA, 0 },
    };
	
	D3DX11_PASS_DESC passDesc;
    hairEffects[0].hairRenderTechnique->GetPassByIndex(0)->GetDesc(&passDesc);		// these have to be the same for all effects
	HRESULT hr = core->GetDevice()->CreateInputLayout(quadlayout, sizeof(quadlayout)/sizeof(D3D11_INPUT_ELEMENT_DESC), passDesc.pIAInputSignature, passDesc.IAInputSignatureSize, &layoutQuad);

	if(FAILED(hr))
	{
		CoreLog::Information(L"Error creating the InputLayout, HRESULT = %x", hr);
		return CORE_MISC_ERROR;
	}

	// Create a screen quad for render to texture operations
    SCREEN_VERTEX svQuad[4];
    svQuad[0].pos = CoreVector3(-1.0f, 1.0f, 0.0f);
    svQuad[0].u   = 0.0f;
	svQuad[0].v   = 0.0f;
    svQuad[1].pos = CoreVector3(1.0f, 1.0f, 0.0f);
    svQuad[1].u   = 1.0f;
	svQuad[1].v   = 0.0f;
    svQuad[2].pos = CoreVector3(-1.0f, -1.0f, 0.0f);
    svQuad[2].u   = 0.0f;
	svQuad[2].v   = 1.0f;
    svQuad[3].pos = CoreVector3(1.0f, -1.0f, 0.0f);
    svQuad[3].u   = 1.0f;
	svQuad[3].v   = 1.0f;

    D3D11_BUFFER_DESC vbdesc =
    {
        4 * sizeof(SCREEN_VERTEX),
        D3D11_USAGE_DEFAULT,
        D3D11_BIND_VERTEX_BUFFER,
        0,
        0
    };
    D3D11_SUBRESOURCE_DATA InitData;
    InitData.pSysMem = svQuad;
    InitData.SysMemPitch = 0;
    InitData.SysMemSlicePitch = 0;
	HRESULT res = core->GetDevice()->CreateBuffer(&vbdesc, &InitData, &bufQuad);
	if(FAILED(res)) 
		return CORE_MISC_ERROR;


	return CORE_OK;	
}

CoreResult Main::createDeferedShadingTextures()
{
	CoreResult cr;
	CoreTexture2D *backBuffer = core->GetBackBuffer();
	D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc;

	srvDesc.Texture2D.MipLevels = 1;
	srvDesc.Texture2D.MostDetailedMip = 0;
	srvDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
	srvDesc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
	
	cr = core->CreateTexture2D(NULL, backBuffer->GetWidth(), backBuffer->GetHeight(), 1, 1, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 0, D3D11_USAGE_DEFAULT, D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET, 1, 0, &deferedPosInLight); 
	if(cr != CORE_OK)
		return cr;
	cr = deferedPosInLight->CreateRenderTargetView(NULL, &deferedPosInLightRTV);
	if(cr != CORE_OK)
		return cr;
	cr = deferedPosInLight->CreateShaderResourceView(&srvDesc, &deferedPosInLightSRV);
	if(cr != CORE_OK)
		return cr;

	cr = core->CreateTexture2D(NULL, backBuffer->GetWidth(), backBuffer->GetHeight(), 1, 1, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 0, D3D11_USAGE_DEFAULT, D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET, 1, 0, &deferedNormal); 
	if(cr != CORE_OK)
		return cr;
	cr = deferedNormal->CreateRenderTargetView(NULL, &deferedNormalRTV);
	if(cr != CORE_OK)
		return cr;
	cr = deferedNormal->CreateShaderResourceView(&srvDesc, &deferedNormalSRV);
	if(cr != CORE_OK)
		return cr;

	cr = core->CreateTexture2D(NULL, backBuffer->GetWidth(), backBuffer->GetHeight(), 1, 1, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 0, D3D11_USAGE_DEFAULT, D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET, 1, 0, &deferedPosInWorld); 
	if(cr != CORE_OK)
		return cr;
	cr = deferedPosInWorld->CreateRenderTargetView(NULL, &deferedPosInWorldRTV);
	if(cr != CORE_OK)
		return cr;
	cr = deferedPosInWorld->CreateShaderResourceView(&srvDesc, &deferedPosInWorldSRV);
	if(cr != CORE_OK)
		return cr;

	cr = core->CreateTexture2D(NULL, backBuffer->GetWidth(), backBuffer->GetHeight(), 1, 1, DXGI_FORMAT_R8G8B8A8_SNORM, 0, 0, D3D11_USAGE_DEFAULT, D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET, 1, 0, &deferedTempColor); 
	if(cr != CORE_OK)
		return cr;
	srvDesc.Format = DXGI_FORMAT_R8G8B8A8_SNORM;
	cr = deferedTempColor->CreateRenderTargetView(NULL, &deferedTempColorRTV);
	if(cr != CORE_OK)
		return cr;
	cr = deferedTempColor->CreateShaderResourceView(&srvDesc, &deferedTempColorSRV);
	if(cr != CORE_OK)
		return cr;

	cr = core->CreateTexture2D(NULL, backBuffer->GetWidth(), backBuffer->GetHeight(), 1, 1, DXGI_FORMAT_R8G8B8A8_SNORM, 0, 0, D3D11_USAGE_DEFAULT, D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET, 1, 0, &deferedColor); 
	if(cr != CORE_OK)
		return cr;
	srvDesc.Format = DXGI_FORMAT_R8G8B8A8_SNORM;
	cr = deferedColor->CreateRenderTargetView(NULL, &deferedColorRTV);
	if(cr != CORE_OK)
		return cr;
	cr = deferedColor->CreateShaderResourceView(&srvDesc, &deferedColorSRV);
	if(cr != CORE_OK)
		return cr;

	deferedRenderTargets = new ID3D11RenderTargetView *[4];
	if(!deferedRenderTargets) return CORE_OUTOFMEM;
	deferedRenderTargets[0] = deferedNormalRTV;
	deferedRenderTargets[1] = deferedPosInWorldRTV;
	deferedRenderTargets[2] = deferedPosInLightRTV;
	deferedRenderTargets[3] = deferedTempColorRTV;

	return CORE_OK;
}

// Release everything
void Main::finalRelease()
{
	SAFE_RELEASE(deferedPosInLightRTV);
	SAFE_RELEASE(deferedPosInWorldRTV);
	SAFE_RELEASE(deferedNormalRTV);
	SAFE_RELEASE(deferedColorRTV);
	SAFE_RELEASE(deferedTempColorRTV);
	SAFE_RELEASE(deferedPosInLightSRV);
	SAFE_RELEASE(deferedPosInWorldSRV);
	SAFE_RELEASE(deferedNormalSRV);
	SAFE_RELEASE(deferedColorSRV);
	SAFE_RELEASE(deferedTempColorSRV);
	SAFE_RELEASE(deferedPosInLight);
	SAFE_RELEASE(deferedPosInWorld);
	SAFE_RELEASE(deferedNormal);
	SAFE_RELEASE(deferedColor);
	SAFE_RELEASE(deferedTempColor);
	SAFE_RELEASE(bufQuad);
	SAFE_RELEASE(layoutQuad);
	SAFE_DELETE(deferedRenderTargets);
	SAFE_RELEASE(dsm);
	SAFE_RELEASE(hairModel);
	for(int model = 0; model < numHeadModels; model++)
		SAFE_RELEASE(headModels[model]);
	SAFE_DELETE(headModels);
	for(int effect = 0; effect < NUM_EFFECTS; effect++)
		SAFE_RELEASE(hairEffects[effect].hairEffect);
	SAFE_RELEASE(swapChain);
	SAFE_RELEASE(core);
	CoreLog::Stop();
	ShowCursor(true);
}

// Animate models
void Main::animate(float timeSinceLastFrame, float timeRunning)
{
	if(orbitMode)
	{
		CoreVector3 pos;
		
		pos.y = lightHeight;
		pos.x = 150.0f * cosf(lightAngle);
		pos.z = 150.0f * sinf(lightAngle);

		lightCamera.SetView(pos, CoreVector3(0, 0, 0), CoreVector3(0.000f, 1.000f, 0.000f));
	}
}

// Render models
void Main::render(float timeSinceLastFrame, float timeRunning)
{	
	CoreCamera *currentCamera;
	if(cameraMode == CAMERA_MAIN)
		currentCamera = &camera;
	else
		currentCamera = &lightCamera;

	CoreVector3 lightPos = lightCamera.GetPosition();
	hairEffects[currentEffect].hairEffectLightPos->SetFloatVector(&lightPos.x);

	dsm->Set(lightCamera, headRotation, hairModel->GetDiffuseColor().a, currentEffect);
	core->GetImmediateDeviceContext()->IASetInputLayout(hairEffects[currentEffect].hairInputLayout);
	hairModel->Draw();
	
	dsm->ChangeAlpha(1.0f, currentEffect);		// no alpha in model
	dsm->ChangeLightCamera(lightCamera, headRotation, currentEffect);
	core->GetImmediateDeviceContext()->IASetInputLayout(hairEffects[currentEffect].headInputLayout);

	for(int model = 0; model < numHeadModels; model++)
		headModels[model]->Draw();

	dsm->Unset(currentEffect);
	dsm->SortLists(currentEffect);
	dsm->LinkLists(currentEffect);
	
	core->ClearRenderTargetView(deferedPosInLightRTV, CoreColor(0.0f, 0.0f, 0.0f, 0.0f));
	core->ClearRenderTargetView(deferedPosInWorldRTV, CoreColor(0.0f, 0.0f, 0.0f, 0.0f));
	core->ClearRenderTargetView(deferedNormalRTV, CoreColor(0.0f, 0.0f, 0.0f, 0.0f));
	
	currentCamera->WorldViewProjectionToEffectVariable(hairEffects[currentEffect].hairEffectWorldViewProj, headRotation);
	hairEffects[currentEffect].hairEffectWorld->SetMatrix(headRotation.arr);
	
	CoreVector3 cameraPos = currentCamera->GetPosition();
	hairEffects[currentEffect].hairEffectCameraPos->SetFloatVector(&cameraPos.x);

	lightCamera.WorldViewProjectionToEffectVariable(hairEffects[currentEffect].hairEffectWorldViewProjLight, headRotation);

	core->GetImmediateDeviceContext()->IASetInputLayout(hairEffects[currentEffect].hairInputLayout);
	
	// draw everything to the textures
	core->SetRenderTargets(4, deferedRenderTargets, core->GetDepthStencilView());
	dsm->SetShaderForRealRendering(hairEffects[currentEffect].hairEffectDimension, hairEffects[currentEffect].hairEffectLinkedListBufWPRO, hairEffects[currentEffect].hairEffectNeighborsBufRO, hairEffects[currentEffect].hairEffectStartElementBuf);
	hairModel->SetMaterial(hairEffects[currentEffect].hairEffectColor, hairEffects[currentEffect].hairEffectTexture);
	hairEffects[currentEffect].hairRenderDeferedTechnique->GetPassByIndex(0)->Apply(0, core->GetImmediateDeviceContext(), true);
	hairModel->Draw();

	core->GetImmediateDeviceContext()->IASetInputLayout(hairEffects[currentEffect].headInputLayout);
	for(int model = 0; model < numHeadModels; model++)
	{
		headModels[model]->SetMaterial(hairEffects[currentEffect].hairEffectColor, hairEffects[currentEffect].hairEffectTexture);
		hairEffects[currentEffect].hairRenderDeferedModelTechnique->GetPassByIndex(0)->Apply(0, core->GetImmediateDeviceContext(), true);
		headModels[model]->Draw();
	}

	// make final output before AA
	core->SetRenderTargets(1, &deferedColorRTV, NULL);
	UINT strides = sizeof(SCREEN_VERTEX);
    UINT offsets = 0;
	core->GetImmediateDeviceContext()->IASetVertexBuffers(0, 1, &bufQuad, &strides, &offsets);
	core->GetImmediateDeviceContext()->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
	core->GetImmediateDeviceContext()->IASetInputLayout(layoutQuad);

	hairEffects[currentEffect].hairEffectDeferedPosInLight->SetResource(deferedPosInLightSRV);
	hairEffects[currentEffect].hairEffectDeferedPosInWorld->SetResource(deferedPosInWorldSRV);
	hairEffects[currentEffect].hairEffectDeferedNormal->SetResource(deferedNormalSRV);
	hairEffects[currentEffect].hairEffectDeferedColor->SetResource(deferedTempColorSRV);

	hairEffects[currentEffect].hairRenderTechnique->GetPassByIndex(0)->Apply(0, core->GetImmediateDeviceContext(), true);
	core->GetImmediateDeviceContext()->Draw(4, 0);

	// AA Pass
	core->SetDefaultRenderTarget();
	hairEffects[currentEffect].hairEffectDeferedColor->SetResource(deferedColorSRV);
	
	hairEffects[currentEffect].hairRenderAATechnique->GetPassByIndex(0)->Apply(0, core->GetImmediateDeviceContext(), true);
	core->GetImmediateDeviceContext()->Draw(4, 0);

	// unset resources to avoid debug output
	hairEffects[currentEffect].hairEffectDeferedColor->SetResource(NULL);
	hairEffects[currentEffect].hairEffectDeferedPosInLight->SetResource(NULL);
	hairEffects[currentEffect].hairEffectDeferedPosInWorld->SetResource(NULL);
	hairEffects[currentEffect].hairEffectDeferedNormal->SetResource(NULL);

	dsm->UnsetShaderForRealRendering(hairEffects[currentEffect].hairEffectDimension, hairEffects[currentEffect].hairEffectLinkedListBufWPRO, hairEffects[currentEffect].hairEffectNeighborsBufRO, hairEffects[currentEffect].hairEffectStartElementBuf);
	hairEffects[currentEffect].hairRenderTechnique->GetPassByIndex(0)->Apply(0, core->GetImmediateDeviceContext(), true);

	guiDialog.OnRender(timeSinceLastFrame);
}

// Windowsize changed
CoreResult Main::WindowSizeChanged(int width, int height)
{
	camera.SetProjectionPerspective(PI / 4, (float)width / height, NEAR_PLANE, FAR_PLANE);

	D3D11_VIEWPORT vp;
	UINT n = 1;
	vp.Width = (float)width;
	vp.Height = (float)height;
	vp.MinDepth = 0.0f;
	vp.MaxDepth = 1.0f;
	vp.TopLeftX = 0;
	vp.TopLeftY = 0;
	core->GetImmediateDeviceContext()->RSSetViewports(1, &vp);

	CoreResult res = core->ResizeSwapChain(width, height, BACKBUFFER_FORMAT);
	if(res != CORE_OK)
		return res;

	SAFE_RELEASE(deferedPosInLightRTV);
	SAFE_RELEASE(deferedPosInWorldRTV);
	SAFE_RELEASE(deferedNormalRTV);
	SAFE_RELEASE(deferedColorRTV);
	SAFE_RELEASE(deferedTempColorRTV);
	SAFE_RELEASE(deferedPosInLightSRV);
	SAFE_RELEASE(deferedPosInWorldSRV);
	SAFE_RELEASE(deferedNormalSRV);
	SAFE_RELEASE(deferedColorSRV);
	SAFE_RELEASE(deferedTempColorSRV);
	SAFE_RELEASE(deferedPosInLight);
	SAFE_RELEASE(deferedPosInWorld);
	SAFE_RELEASE(deferedNormal);
	SAFE_RELEASE(deferedColor);
	SAFE_RELEASE(deferedTempColor);
	SAFE_DELETE(deferedRenderTargets);
	
	res = createDeferedShadingTextures();

	float fbDim[2];
	fbDim[0] = (float)width;
	fbDim[1] = (float)height;
	for(int effect = 0; effect < NUM_EFFECTS; effect++)
		hairEffects[effect].hairEffectFrameBufferDimension->SetFloatVector(fbDim);

	DXGI_SURFACE_DESC backBufferSurfaceDesc;
	backBufferSurfaceDesc.Height = core->GetBackBuffer()->GetHeight();
	backBufferSurfaceDesc.Width = core->GetBackBuffer()->GetWidth();
	backBufferSurfaceDesc.SampleDesc.Count = core->GetBackBuffer()->GetSampleCount();
	backBufferSurfaceDesc.SampleDesc.Quality = core->GetBackBuffer()->GetSampleQuality();
	backBufferSurfaceDesc.Format = core->GetBackBuffer()->GetFormat();

	dialogResourceManager.OnD3D11ResizedSwapChain(core->GetDevice(), &backBufferSurfaceDesc);

	guiDialog.SetLocation(width - 320, 0);

	return res;
}

// Draw a new Frame
bool Main::NewFrame(float timeSinceLastFrame, float timeRunning, bool windowHasFocus)
{
	//core->ClearRenderTargetView(CoreColor(0.025f, 0.025f, 0.025f, 1.0f));
	core->ClearDepthStencilView(D3D11_CLEAR_DEPTH, 1.0f, 0);	

	if (!processInput(timeSinceLastFrame, timeRunning, windowHasFocus))
		return false; // ESCAPE pressed, quit application
	
	animate(timeSinceLastFrame, timeRunning);

	render(timeSinceLastFrame, timeRunning);

	swapChain->Present(0, 0);
	return true;
}

// Process input
bool Main::processInput(float timeSinceLastFrame, float timeRunning, bool windowHasFocus)
{
	CoreCamera *currentCamera;
	if(cameraMode == CAMERA_MAIN)
		currentCamera = &camera;
	else
		currentCamera = &lightCamera;

	if(windowHasFocus)
	{
		if(GetKeyState(VK_ESCAPE) < 0)
			return false;

		if(GetKeyState('W') < 0)
			currentCamera->GoForward(4.0f);
		
		if(GetKeyState('S') < 0)
			currentCamera->GoBackward(4.0f);
		
		if(GetKeyState('A') < 0)
			currentCamera->GoLeft(4.0f);
		
		if(GetKeyState('D') < 0)
			currentCamera->GoRight(4.0f);

		if(GetKeyState('R') < 0)
			currentCamera->GoUp(4.0f);

		if(GetKeyState('F') < 0)
			currentCamera->GoDown(4.0f);

		if(GetKeyState('1') < 0)
		{
			cameraMode = CAMERA_MAIN;
			comboBoxCameraControl->SetSelectedByIndex(0);
		}
		
		if(GetKeyState('2') < 0)
		{
			cameraMode = CAMERA_LIGHT;
			comboBoxCameraControl->SetSelectedByIndex(1);
		}

		if(buttonRotUp->GetPressed())
			headRotation.RotateZLeftThis(2.0f * timeSinceLastFrame);

		if(buttonRotDown->GetPressed())
			headRotation.RotateZLeftThis(-2.0f * timeSinceLastFrame);
	
		if(buttonRotLeft->GetPressed())
			headRotation.RotateYLeftThis(2.0f * timeSinceLastFrame);
	
		if(buttonRotRight->GetPressed())
			headRotation.RotateYLeftThis(-2.0f * timeSinceLastFrame);

		if(buttonOrbitLightRotPlus->GetPressed())
			lightAngle += timeSinceLastFrame;

		if(buttonOrbitLightRotMinus->GetPressed())
			lightAngle -= timeSinceLastFrame;

		if(buttonOrbitLightHeightPlus->GetPressed())
			lightHeight += 50.0f * timeSinceLastFrame;

		if(buttonOrbitLightHeightMinus->GetPressed())
			lightHeight -= 50.0f * timeSinceLastFrame;
	}

	// Mouse look
	// When left mouse button down, hide cursor and rotate camera depending on mouse movement
	POINT currentPos;
	::GetCursorPos(&currentPos);
	float dx = (currentPos.x - mousePos.x) * 0.001f;
	float dy = (currentPos.y - mousePos.y) * 0.001f;

	if(leftMouseButtonTrigger && leftMouseButtonDown)
	{
		int val = 0;
		while(val >= 0)			// there seems to be a bug in the dxut gui stuff
			val = ShowCursor(false);
		SetCapture(hwnd);
	}
	else if(leftMouseButtonTrigger && !leftMouseButtonDown)
	{
		int val = -1;
		while(val < 0)			// there seems to be a bug in the dxut gui stuff
			val = ShowCursor(true);
		ReleaseCapture();
	}
	if (leftMouseButtonDown)
	{
		currentCamera->MouseLook(dx, dy);
		::SetCursorPos(mousePos.x, mousePos.y);
	}
	else
	{
		mousePos.x = currentPos.x;
		mousePos.y = currentPos.y;
	}
	leftMouseButtonTrigger = false;

	return true;
}

void Main::SetMouseButton (bool rightMousebutton, bool buttonDown)
{
	if (rightMousebutton)
		rightMouseButtonDown = buttonDown;
	else 
	{
		leftMouseButtonTrigger = true;
		leftMouseButtonDown = buttonDown;		
	}
}

// Create a Main object
CoreResult CreateMain(HWND hwnd, Main **outMain, bool bFullScreen)
{
	*outMain = new Main();
	CoreResult result = (*outMain)->init(hwnd, bFullScreen);

	if(result != CORE_OK)
	{
		(*outMain)->Release();
		(*outMain) = NULL;
		return result;
	}

	return CORE_OK;
}

void Main::OnGUIEvent(UINT nEvent, int nControlID, CDXUTControl *pControl)
{
	switch(nControlID)
	{
	case IDC_COMBOBOX_SHADOWMODE:
		currentEffect = comboBoxShadowMode->GetSelectedIndex();
		break;
	case IDC_COMBOBOX_CAMERACONTROL:
		cameraMode = (CameraMode)comboBoxCameraControl->GetSelectedIndex();
		break;
	case IDC_SLIDER_HAIR_ALPHA:
	{
		CoreColor newColor = hairModel->GetDiffuseColor();
		newColor.a = sliderHairAlpha->GetValue() / 100.0f;
		hairModel->SetDiffuseColor(newColor);
		break;	
	}
	case IDC_CHECKBOX_ORBIT:
		orbitMode = !orbitMode;
		buttonOrbitLightRotPlus->SetEnabled(orbitMode);
		buttonOrbitLightRotMinus->SetEnabled(orbitMode);
		buttonOrbitLightHeightPlus->SetEnabled(orbitMode);
		buttonOrbitLightHeightMinus->SetEnabled(orbitMode);
		break;
	case IDC_BUTTON_HAIR_COLOR:
	{
		CHOOSECOLOR cc;
		COLORREF custColors[16];			// yes, we want random colors ;)
		memset(&cc, 0, sizeof(cc));
		cc.lStructSize	= sizeof (CHOOSECOLOR);
		cc.hwndOwner	= hwnd;
		cc.rgbResult	= hairModel->GetDiffuseColor().ToBGRA();
		cc.Flags		= CC_FULLOPEN | CC_RGBINIT;
		cc.lpCustColors = custColors;
		
		if(ChooseColor(&cc))
		{
			CoreColor tempColor(cc.rgbResult);
			CoreColor newColor;
			newColor.r = tempColor.b;
			newColor.g = tempColor.g;
			newColor.b = tempColor.r;
			newColor.a = hairModel->GetDiffuseColor().a;
			hairModel->SetDiffuseColor(newColor);
		}
		break;
	}
	};
}


void CALLBACK OnGUIEventFunc(UINT nEvent, int nControlID, CDXUTControl *pControl, void *pUserContext)
{
	Main *main = (Main *)pUserContext;
	main->OnGUIEvent(nEvent, nControlID, pControl);
}

bool Main::MsgProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
	if(dialogResourceManager.MsgProc(hWnd, message, wParam, lParam))
		return true;

	if(guiDialog.MsgProc(hWnd, message, wParam, lParam))
		return true;

	return false;
}

void Main::setDXUTFontColor(D3DCOLOR fontCol, CDXUTControl *control)
{
	for(int state = 0; state < MAX_CONTROL_STATES; state++) 
		if(state != DXUT_STATE_DISABLED)
			for(int element = 0; element < control->m_Elements.GetSize(); element++)
				if(control->GetElement(element)) 
					control->GetElement(element)->FontColor.States[state] = fontCol;
}
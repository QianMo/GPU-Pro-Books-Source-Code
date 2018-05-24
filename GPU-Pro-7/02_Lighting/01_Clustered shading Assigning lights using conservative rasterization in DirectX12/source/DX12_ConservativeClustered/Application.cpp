#include "Application.h"
#include "d3dx12.h"
#include <iostream>
#include <fstream>
#include <ctime>

SharedContext shared_context;

using namespace Log;

Application::Application(char* p_args)
	: m_moveLights(false), m_running(true), m_showShapes(false), m_showSpotLights(true), m_showPointLights(true), m_numSpotShapes(0), m_numPointShapes(15),
	time_shell(0.0f), time_fill(0.0f), time_geometry(0.0f), time_shading(0.0f), time_spot_shell(0.0f), time_spot_fill(0.0f), m_fps(0), time_tot(0.0f),
	time_tot_LA(0.0f), m_useLowLODSphere(true), m_useLowLODCone(true), m_firstFrame(true), m_completeClusterSnap(true), m_useOldShellPipe(false),
	m_showLightDensity(false), m_depthDist(DEPTH_DIST::EXPONENTIAL), m_activeSnapType(LIGHT_TYPE::POINT), m_activeLightSnapIndex(14), m_numPointLights(Constants::NR_POINTLIGHTS), 
	m_numSpotLights(Constants::NR_SPOTLIGHTS), m_nrclusts(0)
{
	
	Log::InitLogColors();

	PRINT(LogLevel::INIT_PRINT, "---STARTING APP---");

	Init();

	SetupAntTweakBar();

	PRINT(LogLevel::SUCCESS, "DX12 set up correctly! Running main loop now with settings:");
	PRINT(LogLevel::HELP_PRINT, "WINDOW_WIDTH:  %d",  Constants::WINDOW_WIDTH);
	PRINT(LogLevel::HELP_PRINT, "WINDOW_HEIGHT: %d", Constants::WINDOW_HEIGHT);
	PRINT(LogLevel::HELP_PRINT, "FARZ:  %f", Constants::FARZ);
	PRINT(LogLevel::HELP_PRINT, "NEARZ: %f", Constants::NEARZ);
	PRINT(LogLevel::HELP_PRINT, "NR_X_CLUSTS:  %d", Constants::NR_X_CLUSTS);
	PRINT(LogLevel::HELP_PRINT, "NR_Y_CLUSTS:  %d", Constants::NR_Y_CLUSTS);
	PRINT(LogLevel::HELP_PRINT, "NR_Z_CLUSTS:  %d", Constants::NR_Z_CLUSTS);
	PRINT(LogLevel::HELP_PRINT, "NR_OF_CLUSTS: %d USING MEM: %d bytes", Constants::NR_OF_CLUSTS, Constants::NR_OF_CLUSTS * sizeof(uint32));
	PRINT(LogLevel::HELP_PRINT, "NR_SPOTLIGHTS:  %d USING MEM: %d bytes", Constants::NR_SPOTLIGHTS, Constants::NR_SPOTLIGHTS * sizeof(SpotLight));
	PRINT(LogLevel::HELP_PRINT, "NR_POINTLIGHTS: %d USING MEM: %d bytes", Constants::NR_POINTLIGHTS, Constants::NR_POINTLIGHTS * sizeof(PointLight));
	PRINT(LogLevel::HELP_PRINT, "MAX_LIGHTS_PER_CLUSTER: %d", Constants::MAX_LIGHTS_PER_CLUSTER);
	PRINT(LogLevel::HELP_PRINT, "LIGHT_INDEX_LIST_COUNT: %d USING MEM: %d bytes", Constants::LIGHT_INDEX_LIST_COUNT, Constants::LIGHT_INDEX_LIST_COUNT * sizeof(LinkedLightID));
	PRINT(LogLevel::HELP_PRINT, "--DESCRIPTOR HEAP INFO--");
	PRINT(LogLevel::HELP_PRINT, "Descpriptor heap CSU: %d / %d descriptors used", shared_context.gfx_device->GetDescHeapCBV_SRV()->GetSize(), shared_context.gfx_device->GetDescHeapCBV_SRV()->GetCapacity());
	PRINT(LogLevel::HELP_PRINT, "Descpriptor heap sampler: %d / %d descriptors used", shared_context.gfx_device->GetDescHeapSampler()->GetSize(), shared_context.gfx_device->GetDescHeapSampler()->GetCapacity());
	PRINT(LogLevel::HELP_PRINT, "Descpriptor heap RTV: %d / %d descriptors used", shared_context.gfx_device->GetDescHeapRTV()->GetSize(), shared_context.gfx_device->GetDescHeapRTV()->GetCapacity());
	PRINT(LogLevel::HELP_PRINT, "Descpriptor heap DSV: %d / %d descriptors used", shared_context.gfx_device->GetDescHeapDSV()->GetSize(), shared_context.gfx_device->GetDescHeapDSV()->GetCapacity());

	Run();

	//clean up tweak bar
	TwTerminate();

	Release();
}

Application::~Application()
{
	delete shared_context.gfx_device;
}

void Application::Init()
{
	HRESULT hr = S_OK;
	shared_context.gfx_device = new KGraphicsDevice();
	hr = shared_context.gfx_device->Init(Constants::WINDOW_WIDTH, Constants::WINDOW_HEIGHT);

	m_camera = std::make_unique<Camera>();
	m_camera->SetLens(DirectX::XM_PI / 4.0f, Constants::NEARZ, Constants::FARZ, shared_context.gfx_device->GetWindowWidth(), shared_context.gfx_device->GetWindowHeight());

	KShader vertexShader(L"../assets/shaders/FullScreenLighting.vertex", "main", "vs_5_0");
	KShader pixelShader(L"../assets/shaders/FullScreenLighting.pixel", "main", "ps_5_0", Constants::SHADER_DEFINES);
	KShader pixelShaderColor(L"../assets/shaders/FullScreenLighting.pixel", "main", "ps_5_0", Constants::SHADER_DEFINES_COLOR);
	KShader pixelShaderLinear(L"../assets/shaders/FullScreenLighting.pixel", "main", "ps_5_0", Constants::SHADER_DEFINES_LINEAR);
	KShader pixelShaderLinearColor(L"../assets/shaders/FullScreenLighting.pixel", "main", "ps_5_0", Constants::SHADER_DEFINES_LINEAR_COLOR);

	RootDescriptorRange root_desc_range[] = 
	{
		{3, ROOT_DESCRIPTOR_TYPE::RANGE_SRV, SHADER_VISIBILITY::PIXEL},
		{1, ROOT_DESCRIPTOR_TYPE::RANGE_SRV, SHADER_VISIBILITY::PIXEL},
		{2, ROOT_DESCRIPTOR_TYPE::RANGE_SRV, SHADER_VISIBILITY::PIXEL},
		{1, ROOT_DESCRIPTOR_TYPE::RANGE_SRV, SHADER_VISIBILITY::PIXEL},
		{1, ROOT_DESCRIPTOR_TYPE::RANGE_UAV, SHADER_VISIBILITY::PIXEL},
	};
	
	g_RootSignature = shared_context.gfx_device->CreateRootSignature(5, root_desc_range);

	D3D12_GRAPHICS_PIPELINE_STATE_DESC PSODesc;
	ZeroMemory(&PSODesc, sizeof(PSODesc));
	PSODesc.InputLayout = vertexShader.GetInputLayout();
	PSODesc.pRootSignature = g_RootSignature;
	PSODesc.VS = { reinterpret_cast<BYTE*>(vertexShader.GetBufferPointer()), vertexShader.GetBufferSize() };
	PSODesc.PS = { reinterpret_cast<BYTE*>(pixelShader.GetBufferPointer()), pixelShader.GetBufferSize() };
	PSODesc.RasterizerState.CullMode = D3D12_CULL_MODE_BACK;
	PSODesc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
	PSODesc.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
	PSODesc.SampleMask = UINT_MAX;
	PSODesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
	PSODesc.NumRenderTargets = 1;
	PSODesc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
	PSODesc.SampleDesc.Count = 1;

	hr = shared_context.gfx_device->GetDevice()->CreateGraphicsPipelineState(&PSODesc, IID_PPV_ARGS(&g_PSO));
	if (FAILED(hr))
		PRINT(LogLevel::FATAL_ERROR, "Failed to create PSO");

	PSODesc.PS = { reinterpret_cast<BYTE*>(pixelShaderLinear.GetBufferPointer()), pixelShaderLinear.GetBufferSize() };
	hr = shared_context.gfx_device->GetDevice()->CreateGraphicsPipelineState(&PSODesc, IID_PPV_ARGS(&m_PSOLinear));
	if (FAILED(hr))
		PRINT(LogLevel::FATAL_ERROR, "Failed to create PSOLinear");

	PSODesc.PS = { reinterpret_cast<BYTE*>(pixelShaderColor.GetBufferPointer()), pixelShaderColor.GetBufferSize() };
	hr = shared_context.gfx_device->GetDevice()->CreateGraphicsPipelineState(&PSODesc, IID_PPV_ARGS(&m_PSOColor));
	if (FAILED(hr))
		PRINT(LogLevel::FATAL_ERROR, "Failed to create m_PSOColor");

	PSODesc.PS = { reinterpret_cast<BYTE*>(pixelShaderLinearColor.GetBufferPointer()), pixelShaderLinearColor.GetBufferSize() };
	hr = shared_context.gfx_device->GetDevice()->CreateGraphicsPipelineState(&PSODesc, IID_PPV_ARGS(&m_PSOLinearColor));
	if (FAILED(hr))
		PRINT(LogLevel::FATAL_ERROR, "Failed to create m_PSOLinearColor");

	//Create sampler
	m_wrapSampler = shared_context.gfx_device->CreateSampler(D3D12_FILTER_ANISOTROPIC, D3D12_TEXTURE_ADDRESS_MODE_WRAP);

	m_cbCamera = shared_context.gfx_device->CreateBuffer(1, sizeof(m_camera->GetCamData()), KBufferType::CONSTANT, D3D12_HEAP_TYPE_UPLOAD);
	memcpy(m_cbCamera.mem, &m_camera->GetCamData(), sizeof(m_camera->GetCamData()));

	uint32 light_type = 0;
	m_cbLightTypePoint = shared_context.gfx_device->CreateBuffer(1, sizeof(uint32), KBufferType::NONE, D3D12_HEAP_TYPE_UPLOAD);
	memcpy(m_cbLightTypePoint.mem, &light_type, sizeof(uint32));
	light_type = 1;
	m_cbLightTypeSpot = shared_context.gfx_device->CreateBuffer(1, sizeof(uint32), KBufferType::NONE, D3D12_HEAP_TYPE_UPLOAD);
	memcpy(m_cbLightTypeSpot.mem, &light_type, sizeof(uint32));
	
	g_inputManager = std::make_unique<InputManager>();
	m_lightManager = std::make_unique<LightManager>();
	m_gbuffer = std::make_unique<GBuffer>();
	m_clusterRenderer = std::make_unique<ClusterRenderer>();
	m_lightShapeRenderer = std::make_unique<LightShapeRenderer>();
	
	//Load assets to VRAM 
	hr = shared_context.gfx_device->GetDevice()->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, shared_context.gfx_device->GetCommandAllocator(), g_PSO, IID_PPV_ARGS(&g_CommandList));
	if (FAILED(hr))
		PRINT(LogLevel::FATAL_ERROR, "Failed to create GFX commandlist");

	OBJLoader loader;
	g_SponzaModel	 = loader.LoadBIN("../assets/models/sponza.bin", g_CommandList);
	m_SphereShape	 = loader.LoadLightShape("../assets/models/testSphere.obj", g_CommandList);
	m_ConeShape		 = loader.LoadLightShape("../assets/models/testCone.obj", g_CommandList);
	m_SphereShapeLow = loader.LoadLightShape("../assets/models/testSphereLow.obj", g_CommandList);
	m_ConeShapeLow	 = loader.LoadLightShape("../assets/models/testConeLow.obj", g_CommandList);

	m_laRaster = std::make_unique<LightAssignmentRaster>(g_CommandList);

	g_CommandList->Close();

	ID3D12CommandList* commandLists[] = { g_CommandList };

	shared_context.gfx_device->GetCommandQueue()->ExecuteCommandLists(1, commandLists);

	shared_context.gfx_device->WaitForGPU();

	g_SponzaModel->CleanupUploadData();
}

void Application::Run()
{
	uint32 curtime = SDL_GetTicks();
	uint32 prevtime = 0;
	uint32 frames = 0;
	float collectedTime = 0.0f;

	while (m_running)
	{
		prevtime = curtime;
		curtime = SDL_GetTicks();
		++frames;
		float dt = (curtime - prevtime) / 1000.0f;
		collectedTime += dt;
		if (collectedTime >= 1.0f)
		{
			collectedTime -= 1.0f;
			m_fps = frames;

			frames = 0;
		}

		HandleEvents();
		Update(dt);
		
		Draw();
		g_inputManager->Reset();
	}
}

void Application::Update(float dt)
{
	//Handle mouse, keyboard
	if (g_inputManager->GetKeyState(SDL_SCANCODE_ESCAPE) == KeyState::DOWN_EDGE)
		m_running = false;
	if (g_inputManager->GetKeyState(SDL_SCANCODE_A) == KeyState::DOWN)
		m_camera->StrafeLeft(dt);
	if (g_inputManager->GetKeyState(SDL_SCANCODE_D) == KeyState::DOWN)
		m_camera->StrafeRight(dt);
	if (g_inputManager->GetKeyState(SDL_SCANCODE_W) == KeyState::DOWN)
		m_camera->MoveForward(dt);
	if (g_inputManager->GetKeyState(SDL_SCANCODE_S) == KeyState::DOWN)
		m_camera->MoveBackward(dt);
	if (g_inputManager->GetKeyState(SDL_SCANCODE_LSHIFT) == KeyState::DOWN)
		m_camera->MoveDown(dt);
	if (g_inputManager->GetKeyState(SDL_SCANCODE_SPACE) == KeyState::DOWN)
		m_camera->MoveUp(dt);
	if (g_inputManager->GetKeyState(MouseButton::LEFT) == KeyState::DOWN)
	{
		float pitch = g_inputManager->GetDeltaMousPos().y / 200.0f;
		float yaw = -g_inputManager->GetDeltaMousPos().x / 200.0f;

		m_camera->Pitch(pitch);
		m_camera->Yaw(yaw);
	}
	//Handle gamepad
	if (g_inputManager->GetKeyState(GamepadButton::START) == KeyState::DOWN_EDGE)
		m_running = false;
	if (g_inputManager->GetKeyState(GamepadButton::X) == KeyState::DOWN_EDGE)
		m_moveLights = !m_moveLights;
	if (g_inputManager->GetKeyState(GamepadButton::A) == KeyState::DOWN_EDGE)
		m_showShapes = !m_showShapes;
	if (g_inputManager->GetKeyState(GamepadButton::LSTICK) == KeyState::DOWN_EDGE)
		m_showPointLights = !m_showPointLights;
	if (g_inputManager->GetKeyState(GamepadButton::RSTICK) == KeyState::DOWN_EDGE)
		m_showSpotLights = !m_showSpotLights;
	if (g_inputManager->GetKeyState(GamepadButton::DUP) == KeyState::DOWN_EDGE)
		m_showLightDensity = !m_showLightDensity;
	if (g_inputManager->GetKeyState(GamepadButton::DLEFT) == KeyState::DOWN_EDGE)
		m_depthDist = (DEPTH_DIST)!m_depthDist;

	float throttle = 1.0f + g_inputManager->GetTriggers().right * 10.0f;
	float brake = 1.0f - g_inputManager->GetTriggers().left;

	float posx = g_inputManager->GetThumbSticks().leftX;
	float posy = g_inputManager->GetThumbSticks().leftY;

	if (posx > 0)
		m_camera->StrafeRight(dt, posx * throttle * brake);
	else if (posx < 0)
		m_camera->StrafeLeft(dt, -posx * throttle * brake);

	if (posy < 0)
		m_camera->MoveForward(dt, -posy * throttle * brake);
	else if (posy > 0)
		m_camera->MoveBackward(dt, posy * throttle * brake);

	if (g_inputManager->GetKeyState(GamepadButton::LSHOULDER) == KeyState::DOWN)
		m_camera->MoveDown(dt, (1.0f * throttle) * brake);
	if (g_inputManager->GetKeyState(GamepadButton::RSHOULDER) == KeyState::DOWN)
		m_camera->MoveUp(dt, (1.0f * throttle) * brake);

	float camx = g_inputManager->GetThumbSticks().rightX;
	float camy = g_inputManager->GetThumbSticks().rightY;

	m_camera->Pitch(camy * dt * 3.0f);
	m_camera->Yaw(-camx * dt * 3.0f);

	m_camera->Update();
	memcpy(m_cbCamera.mem, &m_camera->GetCamData(), sizeof(m_camera->GetCamData()));

	if (m_moveLights)
		m_lightManager->Update(dt);
}

void Application::Draw()
{
	int32 swapIndex = shared_context.gfx_device->GetSwapIndex();

	//RENDER TIME
	//Reset and populate command list for this frame
	shared_context.gfx_device->GetCommandAllocator()->Reset();
	g_CommandList->Reset(shared_context.gfx_device->GetCommandAllocator(), g_PSO);

	//Set descriptor heaps
	ID3D12DescriptorHeap* descHeaps[] = { shared_context.gfx_device->GetDescHeapCBV_SRV()->GetHeap(), shared_context.gfx_device->GetDescHeapSampler()->GetHeap() };
	g_CommandList->SetDescriptorHeaps(2, descHeaps);

	//////////////////////////////////////////////////////////////////////////
	//GPU LA
	//////////////////////////////////////////////////////////////////////////

	D3D12_VIEWPORT vp = { 0.0f, 0.0f, (float)Constants::NR_X_CLUSTS, (float)Constants::NR_Y_CLUSTS, 0.0f, 1.0f };
	D3D12_RECT sr = { 0, 0, Constants::NR_X_CLUSTS, Constants::NR_Y_CLUSTS };
	g_CommandList->RSSetViewports(1, &vp);
	g_CommandList->RSSetScissorRects(1, &sr);

	//Clear UAVs (clear UAV function doesn't work atm)
	m_laRaster->ClearUAVs(g_CommandList);

	//////////////////////////////////////////////////////////////////////////
	//Point light shell pass
	if(m_useOldShellPipe)
	{
		if (m_depthDist)
			g_CommandList->SetPipelineState(m_laRaster->GetPSOPointOld());
		else
			g_CommandList->SetPipelineState(m_laRaster->GetPSOPointLinearOld());
	}
	else
	{
		if(m_depthDist)
			g_CommandList->SetPipelineState(m_laRaster->GetPSOPoint());
		else
			g_CommandList->SetPipelineState(m_laRaster->GetPSOPointLinear());
	}

	g_CommandList->SetGraphicsRootSignature(m_laRaster->GetRootSig());

	g_CommandList->SetGraphicsRootDescriptorTable(0, m_cbCamera.cbv.gpu_handle);
	g_CommandList->SetGraphicsRootDescriptorTable(1, m_lightManager->GetPointLightSRV().gpu_handle);

	shared_context.gfx_device->TransitionResource(g_CommandList, m_laRaster->GetColorRT(swapIndex)->GetResource(), D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET);
	g_CommandList->ClearRenderTargetView(m_laRaster->GetColorRT(swapIndex)->GetRTVCPUHandle(), COLOR::WHITE, 0, nullptr);
	g_CommandList->OMSetRenderTargets(1, &m_laRaster->GetColorRT(swapIndex)->GetRTVCPUHandle(), true, nullptr);
	shared_context.gfx_device->SetTimeStampQuery(0 + 2 * swapIndex, g_CommandList);
	if(m_showPointLights)
	{
		if (m_useLowLODSphere)
			m_SphereShapeLow->DrawIndexedInstanced(m_numPointLights, g_CommandList);
		else
			m_SphereShape->DrawIndexedInstanced(m_numPointLights, g_CommandList);

	}
	shared_context.gfx_device->SetTimeStampQuery(1 + 2 * swapIndex, g_CommandList);
	shared_context.gfx_device->TransitionResource(g_CommandList, m_laRaster->GetColorRT(swapIndex)->GetResource(), D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);

	//Point light fill pass
	g_CommandList->SetPipelineState(m_laRaster->GetComputePSO());
	g_CommandList->SetComputeRootSignature(m_laRaster->GetComputeRootSig());

	g_CommandList->SetComputeRootDescriptorTable(0, m_laRaster->GetColorRT(swapIndex)->GetSRVGPUHandle());
	g_CommandList->SetComputeRootDescriptorTable(1, m_laRaster->GetSobUAVGPUHandle());
	g_CommandList->SetComputeRootConstantBufferView(2, m_cbLightTypePoint.resource->GetGPUVirtualAddress());

	shared_context.gfx_device->SetTimeStampQuery(4 + 2 * swapIndex, g_CommandList);
	g_CommandList->Dispatch(Constants::NR_X_CLUSTS / 24, Constants::NR_Y_CLUSTS / 12, m_numPointLights);
	shared_context.gfx_device->SetTimeStampQuery(5 + 2 * swapIndex, g_CommandList);
	//////////////////////////////////////////////////////////////////////////

	//////////////////////////////////////////////////////////////////////////
	//Spot light shell pass
	if (m_useOldShellPipe)
	{
		if (m_depthDist)
			g_CommandList->SetPipelineState(m_laRaster->GetPSOSpotOld());
		else
			g_CommandList->SetPipelineState(m_laRaster->GetPSOSpotLinearOld());
	}
	else
	{
		if(m_depthDist)
			g_CommandList->SetPipelineState(m_laRaster->GetPSOSpot());
		else
			g_CommandList->SetPipelineState(m_laRaster->GetPSOSpotLinear());
	}

	g_CommandList->SetGraphicsRootSignature(m_laRaster->GetRootSig());

	g_CommandList->SetGraphicsRootDescriptorTable(0, m_cbCamera.cbv.gpu_handle);
	g_CommandList->SetGraphicsRootDescriptorTable(1, m_lightManager->GetSpotLightSRV().gpu_handle); //Light data

	shared_context.gfx_device->TransitionResource(g_CommandList, m_laRaster->GetSpotRT(swapIndex)->GetResource(), D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET);
	g_CommandList->ClearRenderTargetView(m_laRaster->GetSpotRT(swapIndex)->GetRTVCPUHandle(), COLOR::WHITE, 0, nullptr);
	g_CommandList->OMSetRenderTargets(1, &m_laRaster->GetSpotRT(swapIndex)->GetRTVCPUHandle(), true, nullptr);
	shared_context.gfx_device->SetTimeStampQuery(16 + 2 * swapIndex, g_CommandList);
	if(m_showSpotLights)
	{
		if(m_useLowLODCone)
			m_ConeShapeLow->DrawIndexedInstanced(m_numSpotLights, g_CommandList);
		else
			m_ConeShape->DrawIndexedInstanced(m_numSpotLights, g_CommandList);
	}
	shared_context.gfx_device->SetTimeStampQuery(17 + 2 * swapIndex, g_CommandList);
	shared_context.gfx_device->TransitionResource(g_CommandList, m_laRaster->GetSpotRT(swapIndex)->GetResource(), D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);

	//Spot light fill pass
	g_CommandList->SetPipelineState(m_laRaster->GetComputePSO());
	g_CommandList->SetComputeRootSignature(m_laRaster->GetComputeRootSig());

	g_CommandList->SetComputeRootDescriptorTable(0, m_laRaster->GetSpotRT(swapIndex)->GetSRVGPUHandle());
	g_CommandList->SetComputeRootDescriptorTable(1, m_laRaster->GetSobUAVGPUHandle());
	g_CommandList->SetComputeRootConstantBufferView(2, m_cbLightTypeSpot.resource->GetGPUVirtualAddress());

	shared_context.gfx_device->SetTimeStampQuery(20 + 2 * swapIndex, g_CommandList);
	g_CommandList->Dispatch(Constants::NR_X_CLUSTS / 24, Constants::NR_Y_CLUSTS / 12, m_numSpotLights);
	shared_context.gfx_device->SetTimeStampQuery(21 + 2 * swapIndex, g_CommandList);
	//////////////////////////////////////////////////////////////////////////



	//////////////////////////////////////////////////////////////////////////
	//DRAW 
	//////////////////////////////////////////////////////////////////////////

	//////////////////////////////////////////////////////////////////////////
	//GBuffer pass
	g_CommandList->RSSetViewports(1, &shared_context.gfx_device->GetViewPort());
	g_CommandList->RSSetScissorRects(1, &shared_context.gfx_device->GetScissorRect());

	g_CommandList->SetPipelineState(m_gbuffer->GetPSO());
	g_CommandList->SetGraphicsRootSignature(m_gbuffer->GetRootSig());

	g_CommandList->SetGraphicsRootDescriptorTable(0, m_cbCamera.cbv.gpu_handle);
	g_CommandList->SetGraphicsRootDescriptorTable(2, m_wrapSampler.gpu_handle);

	ID3D12Resource* transRes[] = { m_gbuffer->GetColorRT(swapIndex)->GetResource(), m_gbuffer->GetNormalRT(swapIndex)->GetResource(), m_gbuffer->GetPositionRT(swapIndex)->GetResource() };
	shared_context.gfx_device->TransitionResources(3, g_CommandList, transRes, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET);
	g_CommandList->ClearRenderTargetView(m_gbuffer->GetColorRT(swapIndex)->GetRTVCPUHandle(), COLOR::BLACK, 0, nullptr);
	g_CommandList->ClearRenderTargetView(m_gbuffer->GetNormalRT(swapIndex)->GetRTVCPUHandle(), COLOR::BLACK, 0, nullptr);
	g_CommandList->ClearRenderTargetView(m_gbuffer->GetPositionRT(swapIndex)->GetRTVCPUHandle(), COLOR::BLACK, 0, nullptr);
	g_CommandList->ClearDepthStencilView(m_gbuffer->GetDepthTarget(swapIndex)->GetDSVCPUHandle(), D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);
	g_CommandList->OMSetRenderTargets(3, &m_gbuffer->GetColorRT(swapIndex)->GetRTVCPUHandle(), true, &m_gbuffer->GetDepthTarget(swapIndex)->GetDSVCPUHandle());
	g_SponzaModel->Apply(g_CommandList);
	shared_context.gfx_device->SetTimeStampQuery(8 + 2 * swapIndex, g_CommandList);
	g_SponzaModel->DrawIndexed(g_CommandList);
	shared_context.gfx_device->SetTimeStampQuery(9 + 2 * swapIndex, g_CommandList);
	shared_context.gfx_device->TransitionResources(3, g_CommandList, transRes, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
	//////////////////////////////////////////////////////////////////////////

	//////////////////////////////////////////////////////////////////////////
	//Shading pass
	if(m_depthDist)
	{
		if(m_showLightDensity)
			g_CommandList->SetPipelineState(m_PSOColor);
		else
			g_CommandList->SetPipelineState(g_PSO);
	}
	else
	{
		if (m_showLightDensity)
			g_CommandList->SetPipelineState(m_PSOLinearColor);
		else
			g_CommandList->SetPipelineState(m_PSOLinear);
	}

	g_CommandList->SetGraphicsRootSignature(g_RootSignature);

	g_CommandList->SetGraphicsRootDescriptorTable(0, m_gbuffer->GetColorRT(swapIndex)->GetSRVGPUHandle());
	g_CommandList->SetGraphicsRootDescriptorTable(1, m_gbuffer->GetDepthTarget(swapIndex)->GetSRVGPUHandle());
	g_CommandList->SetGraphicsRootDescriptorTable(2, m_lightManager->GetPointLightSRV().gpu_handle);
	g_CommandList->SetGraphicsRootDescriptorTable(3, m_laRaster->GetLLLSRVGPUHandle());
	g_CommandList->SetGraphicsRootDescriptorTable(4, m_laRaster->GetSobUAVGPUHandle());
	
	shared_context.gfx_device->TransitionResource(g_CommandList, shared_context.gfx_device->GetRTResource(), D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET);
	shared_context.gfx_device->TransitionResource(g_CommandList, m_gbuffer->GetDepthTarget(swapIndex)->GetResource(), D3D12_RESOURCE_STATE_DEPTH_WRITE, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_DEPTH_READ);
	g_CommandList->OMSetRenderTargets(1, &shared_context.gfx_device->GetRTDescHandle(), true, nullptr);
	g_CommandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
	shared_context.gfx_device->SetTimeStampQuery(12 + 2 * swapIndex, g_CommandList);
	g_CommandList->DrawInstanced(3, 1, 0, 0);
	shared_context.gfx_device->SetTimeStampQuery(13 + 2 * swapIndex, g_CommandList);

	shared_context.gfx_device->TransitionResource(g_CommandList, m_gbuffer->GetDepthTarget(swapIndex)->GetResource(), D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_DEPTH_READ, D3D12_RESOURCE_STATE_DEPTH_WRITE);
	//////////////////////////////////////////////////////////////////////////
		
	//////////////////////////////////////////////////////////////////////////
	//Draw debug lines for clusters
	g_CommandList->SetPipelineState(m_clusterRenderer->GetPSO());
	g_CommandList->SetGraphicsRootSignature(m_clusterRenderer->GetRootSig());

	g_CommandList->SetGraphicsRootDescriptorTable(0, m_cbCamera.cbv.gpu_handle);
	g_CommandList->SetGraphicsRootDescriptorTable(1, m_clusterRenderer->GetGPUHandle());
	g_CommandList->OMSetRenderTargets(1, &shared_context.gfx_device->GetRTDescHandle(), true, &m_gbuffer->GetDepthTarget(swapIndex)->GetDSVCPUHandle());
	g_CommandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_LINELIST);
	g_CommandList->DrawInstanced(m_clusterRenderer->GetNumPoints(), 1, 0, 0);
	//////////////////////////////////////////////////////////////////////////

	//////////////////////////////////////////////////////////////////////////
	//Draw debug light shapes
	if(m_showShapes)
	{
		
		g_CommandList->SetGraphicsRootSignature(m_lightShapeRenderer->GetRootSig());
		g_CommandList->SetPipelineState(m_lightShapeRenderer->GetPSO());

		//Set root tables
		g_CommandList->SetGraphicsRootDescriptorTable(0, m_cbCamera.cbv.gpu_handle);
		g_CommandList->SetGraphicsRootDescriptorTable(1, m_lightManager->GetPointLightSRV().gpu_handle);
		g_CommandList->OMSetRenderTargets(1, &shared_context.gfx_device->GetRTDescHandle(), true, &m_gbuffer->GetDepthTarget(swapIndex)->GetDSVCPUHandle());
		g_CommandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
		
		if (m_useLowLODSphere)
			m_SphereShapeLow->DrawIndexedInstanced(m_numPointShapes, g_CommandList);
		else
			m_SphereShape->DrawIndexedInstanced(m_numPointShapes, g_CommandList);

		g_CommandList->SetPipelineState(m_lightShapeRenderer->GetSpotPSO());
		g_CommandList->SetGraphicsRootDescriptorTable(1, m_lightManager->GetSpotLightSRV().gpu_handle);

		if (m_useLowLODCone)
			m_ConeShapeLow->DrawIndexedInstanced(m_numSpotShapes, g_CommandList);
		else
			m_ConeShape->DrawIndexedInstanced(m_numSpotShapes, g_CommandList);
	}
	//////////////////////////////////////////////////////////////////////////

	//////////////////////////////////////////////////////////////////////////
	//DEBUG STUFF
	//////////////////////////////////////////////////////////////////////////
	m_laRaster->ReadBackDebugData(g_CommandList, swapIndex);

	if (g_inputManager->GetKeyState(GamepadButton::Y) == KeyState::DOWN_EDGE)
	{
		ShapeClusterSnapShot();
	}
	else if (g_inputManager->GetKeyState(GamepadButton::DRIGHT) == KeyState::DOWN_EDGE)
	{
		ShadeClusterSnapShot();
	}

	if ((!m_firstFrame && m_completeClusterSnap) || g_inputManager->GetKeyState(GamepadButton::B) == KeyState::DOWN_EDGE)
	{
		ClusterSnapShotCount();
	}
	m_firstFrame = false;
	//////////////////////////////////////////////////////////////////////////
	//Draw GUI
	//Set backbuffer for AntTweakBar to render
	g_CommandList->OMSetRenderTargets(1, &shared_context.gfx_device->GetRTDescHandle(), true, nullptr);
	
	//Process AntTweakBarEvents. Bit of a hack to work with DX12.
	for(auto event : m_atbSDLEvents)
		TwEventSDL((void*)&event, 2, 0);
	m_atbSDLEvents.clear();
	TwRefreshBar(antbar);
	TwDraw();

	shared_context.gfx_device->TransitionResource(g_CommandList, shared_context.gfx_device->GetRTResource(), D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PRESENT);
	//////////////////////////////////////////////////////////////////////////

	g_CommandList->Close();

	//Execute commandlist
	ID3D12CommandList* commandLists[] = { g_CommandList };
	shared_context.gfx_device->GetCommandQueue()->ExecuteCommandLists(1, commandLists);

	//Temp timing code
	uint64 before = shared_context.gfx_device->QueryTimeStamp(0 + 2 * !swapIndex);
	uint64 after = shared_context.gfx_device->QueryTimeStamp(1 + 2 * !swapIndex);
	uint64 diff = after - before;
	time_shell = (float)((diff / (double)shared_context.gfx_device->GetFreq()) * 1000.0);

	before = shared_context.gfx_device->QueryTimeStamp(4 + 2 * !swapIndex);
	after = shared_context.gfx_device->QueryTimeStamp(5 + 2 * !swapIndex);
	diff = after - before;
	time_fill = (float)((diff / (double)shared_context.gfx_device->GetFreq()) * 1000.0);

	before = shared_context.gfx_device->QueryTimeStamp(8 + 2 * !swapIndex);
	after = shared_context.gfx_device->QueryTimeStamp(9 + 2 * !swapIndex);
	diff = after - before;
	time_geometry = (float)((diff / (double)shared_context.gfx_device->GetFreq()) * 1000.0);

	before = shared_context.gfx_device->QueryTimeStamp(12 + 2 * !swapIndex);
	after = shared_context.gfx_device->QueryTimeStamp(13 + 2 * !swapIndex);
	diff = after - before;
	time_shading = (float)((diff / (double)shared_context.gfx_device->GetFreq()) * 1000.0);

	before = shared_context.gfx_device->QueryTimeStamp(16 + 2 * !swapIndex);
	after = shared_context.gfx_device->QueryTimeStamp(17 + 2 * !swapIndex);
	diff = after - before;
	time_spot_shell = (float)((diff / (double)shared_context.gfx_device->GetFreq()) * 1000.0);

	before = shared_context.gfx_device->QueryTimeStamp(20 + 2 * !swapIndex);
	after = shared_context.gfx_device->QueryTimeStamp(21 + 2 * !swapIndex);
	diff = after - before;
	time_spot_fill = (float)((diff / (double)shared_context.gfx_device->GetFreq()) * 1000.0);

	time_tot_LA = time_shell + time_fill + time_spot_shell + time_spot_fill;
	time_tot = time_tot_LA + time_shading;

	shared_context.gfx_device->Present();
}

void Application::Release()
{
	g_RootSignature->Release();
	g_PSO->Release();
	m_PSOLinear->Release();
	m_PSOColor->Release();
	m_PSOLinearColor->Release();
	m_cbCamera.resource->Release();
	m_cbLightTypePoint.resource->Release();
	m_cbLightTypeSpot.resource->Release();
	g_CommandList->Release();
}

void Application::HandleEvents()
{
	SDL_Event event;
	while (SDL_PollEvent(&event))
	{
		switch (event.type)
		{
		case SDL_QUIT:
			m_running = false;
			return;
		}
		//Store sdl events for proccessing in a command buffer later
		m_atbSDLEvents.push_back(event);
		g_inputManager->Update(event);
	}
}


void Application::ShadeClusterSnapShot()
{
	uint32* datta = nullptr;
	m_laRaster->GetStartOffsetReadResource()->Map(0, &CD3DX12_RANGE(0, 0), (void**)&datta);
	LinkedLightID* lightData = nullptr;
	m_laRaster->GetLinkedLightListReadResource()->Map(0, &CD3DX12_RANGE(0, 0), (void**)&lightData);
	m_clusterRenderer->BuildWorldSpacePositions(m_camera.get(), m_depthDist == DEPTH_DIST::EXPONENTIAL);
	uint32 num_clusters = 0;
	uint32 max_lights_in_clusts = 0;
	for (int i = 0; i < Constants::NR_OF_CLUSTS; ++i)
	{
		uint32 shade_bits = (datta[i] >> 30);

		if (shade_bits == 1)
		{
			uint32 x = i % Constants::NR_X_CLUSTS;
			uint32 y = (i / Constants::NR_X_CLUSTS) % Constants::NR_Y_CLUSTS;
			uint32 z = i / (Constants::NR_X_CLUSTS * Constants::NR_Y_CLUSTS);

			m_clusterRenderer->AddCluster(x, y, z);
			++num_clusters;

			//Count lights in clusters
			uint32 light_index = (datta[i] & 0x3FFFFFFF);
			uint32 light_count = 0;
			
			while (light_index != 0x3FFFFFFF)
			{
				light_index = lightData[light_index].link;
				++light_count;
			}

			if (light_count > max_lights_in_clusts)
				max_lights_in_clusts = light_count;

		}
	}
	PRINT(LogLevel::DEBUG_PRINT, "Num shaded clusters: %d", num_clusters);
	PRINT(LogLevel::DEBUG_PRINT, "Max lights in clusters: %d", max_lights_in_clusts);
	m_clusterRenderer->UploadClusters();
	m_laRaster->GetStartOffsetReadResource()->Unmap(0, nullptr);
	m_laRaster->GetLinkedLightListReadResource()->Unmap(0, nullptr);
}

void Application::ShapeClusterSnapShot()
{
	if ((m_activeLightSnapIndex < m_numPointShapes && m_activeSnapType == LIGHT_TYPE::POINT) || (m_activeLightSnapIndex < m_numSpotShapes && m_activeSnapType == LIGHT_TYPE::SPOT))
	{
		int32* datta = nullptr;
		m_laRaster->GetStartOffsetReadResource()->Map(0, &CD3DX12_RANGE(0, 0), (void**)&datta);
		LinkedLightID* lightData = nullptr;
		m_laRaster->GetLinkedLightListReadResource()->Map(0, &CD3DX12_RANGE(0, 0), (void**)&lightData);
		m_clusterRenderer->BuildWorldSpacePositions(m_camera.get(), m_depthDist == DEPTH_DIST::EXPONENTIAL);
		uint32 num_clusters = 0;
		for (int i = 0; i < Constants::NR_OF_CLUSTS; ++i)
		{
			uint32 light_index = (datta[i] & 0x3FFFFFFF);
			LinkedLightID linked_light;

			while (light_index != 0x3FFFFFFF)
			{
				linked_light = lightData[light_index];
				if ((linked_light.lightID & 0xFFFFFF) == m_activeLightSnapIndex && (linked_light.lightID >> 24) == m_activeSnapType)
				{
					uint32 x = i % Constants::NR_X_CLUSTS;
					uint32 y = (i / Constants::NR_X_CLUSTS) % Constants::NR_Y_CLUSTS;
					uint32 z = i / (Constants::NR_X_CLUSTS * Constants::NR_Y_CLUSTS);

					m_clusterRenderer->AddCluster(x, y, z);
					++num_clusters;
					break;
				}
				light_index = linked_light.link;
			}
		}
		PRINT(LogLevel::DEBUG_PRINT, "Num clusters: %d", num_clusters);
		m_clusterRenderer->UploadClusters();
		m_laRaster->GetStartOffsetReadResource()->Unmap(0, nullptr);
		m_laRaster->GetLinkedLightListReadResource()->Unmap(0, nullptr);
	}
}

void Application::ClusterSnapShotCount()
{
	m_completeClusterSnap = false;
	int32* datta = nullptr;
	m_laRaster->GetUAVCounterReadResource()->Map(0, &CD3DX12_RANGE(0, 0), (void**)&datta);
	PRINT(LogLevel::DEBUG_PRINT, "Total cluster assignments: %d", *datta);
	m_nrclusts = *datta;
	m_laRaster->GetUAVCounterReadResource()->Unmap(0, nullptr);
}

void TW_CALL ShadeClusterSnapShotCB(void* app)
{
	Application* application = static_cast<Application*>(app);
	application->ShadeClusterSnapShot();
}

void TW_CALL ShapeClusterSnapShotCB(void* app)
{
	Application* application = static_cast<Application*>(app);
	application->ShapeClusterSnapShot();
}

void TW_CALL ClusterSnapShotCountCB(void* app)
{
	Application* application = static_cast<Application*>(app);
	application->ClusterSnapShotCount();
}

void Application::SetupAntTweakBar()
{
	//Set up Tweak Bar
	TwInit(TW_DIRECT3D12, (void*)shared_context.gfx_device->GetDevice(), (void*)g_CommandList);
	TwWindowSize(shared_context.gfx_device->GetWindowWidth(), shared_context.gfx_device->GetWindowHeight());
	antbar = TwNewBar("ClusteredDemo");
	TwDefine(" ClusteredDemo size='260 520' ");

	//Lights group
	TwAddVarRW(antbar, "pointLights", TW_TYPE_BOOLCPP, &m_showPointLights, " label='Point lights' group='Lights'");
	TwAddVarRW(antbar, "spotLights", TW_TYPE_BOOLCPP, &m_showSpotLights, " label='Spot lights' group='Lights'");
	TwAddVarRW(antbar, "move lights", TW_TYPE_BOOLCPP, &m_moveLights, " label='Light movement' group='Lights'");
	
	char numlights[128];
	sprintf_s(numlights, "min=0 max=%d group='Lights'", Constants::NR_POINTLIGHTS);
	TwAddVarRW(antbar, "Num point lights", TW_TYPE_UINT32, &m_numPointLights, numlights);

	sprintf_s(numlights, "min=0 max=%d group='Lights'", Constants::NR_SPOTLIGHTS);
	TwAddVarRW(antbar, "Num spot lights", TW_TYPE_UINT32, &m_numSpotLights, numlights);

	//Light shapes group
	TwAddVarRW(antbar, "show shapes", TW_TYPE_BOOLCPP, &m_showShapes, " label='Show shapes' group='Light shapes'");
	TwAddVarRW(antbar, "sphereLOD", TW_TYPE_BOOLCPP, &m_useLowLODSphere, " label='Low sphere LOD' group='Light shapes'");
	TwAddVarRW(antbar, "coneLOD", TW_TYPE_BOOLCPP, &m_useLowLODCone, " label='Low cone LOD' group='Light shapes'");
	
	char props[128];
	sprintf_s(props, "min=0 max=%d group='Light shapes'", Constants::NR_SPOTLIGHTS);
	TwAddVarRW(antbar, "Num spot shapes", TW_TYPE_UINT32, &m_numSpotShapes, props);

	sprintf_s(props, "min=0 max=%d group='Light shapes'", Constants::NR_POINTLIGHTS);
	TwAddVarRW(antbar, "Num point shapes", TW_TYPE_UINT32, &m_numPointShapes, props);

	//Clustering group
	{
		TwEnumVal enumval[] = { { EXPONENTIAL, "Exponential" }, { LINEAR, "Linear" } };
		TwType type = TwDefineEnum("DEPTH_DIST", enumval, ARRAYSIZE(enumval));
		TwAddVarRW(antbar, "Depth dist", type, &m_depthDist, " label='Depth dist' group='Clustering' ");
	}

	TwAddVarRW(antbar, "oldpipe", TW_TYPE_BOOLCPP, &m_useOldShellPipe, " label='Use approx. pipe' group='Clustering'");
	TwAddVarRW(antbar, "lightDens", TW_TYPE_BOOLCPP, &m_showLightDensity, " label='Density map' group='Clustering'");
	TwAddButton(antbar, "CSSCCB", ClusterSnapShotCountCB, this, " label='Cluster assign count' key=c help='Print number of total cluster assignments' group='Clustering'");
	TwAddButton(antbar, "SCSSCB", ShadeClusterSnapShotCB, this, " label='Shade cluster snap' key=f help='Display a snap shot of clusters used for shading' group='Clustering'");

	//Snap shot group
	{
		TwEnumVal enumval[] = { { POINT, "Point" }, { SPOT, "Spot" } };
		TwType type = TwDefineEnum("LIGHT_TYPE", enumval, ARRAYSIZE(enumval));
		TwAddVarRW(antbar, "Snap type", type, &m_activeSnapType, " label='Light type' group='Snap shot' ");
	}
	sprintf_s(props, "min=0 max=%d group='Snap shot'", (Constants::NR_POINTLIGHTS - 1 < Constants::NR_SPOTLIGHTS - 1) ? Constants::NR_SPOTLIGHTS - 1 : Constants::NR_POINTLIGHTS - 1 );
	TwAddVarRW(antbar, "Shape index", TW_TYPE_UINT32, &m_activeLightSnapIndex, props);
	TwAddButton(antbar, "ShapeCSSCB", ShapeClusterSnapShotCB, this, " label='Shape cluster snap' key=v help='Display a snap shot of a clustered light shape' group='Snap shot'");

	//Benchmark group
	TwAddVarRO(antbar, "time_shell", TW_TYPE_FLOAT, &time_shell, " label='Point shell time' group='Benchmark'");
	TwAddVarRO(antbar, "time_fill", TW_TYPE_FLOAT, &time_fill, " label='Point fill time' group='Benchmark'");
	TwAddVarRO(antbar, "time_sp_shell", TW_TYPE_FLOAT, &time_spot_shell, " label='Spot shell time' group='Benchmark'");
	TwAddVarRO(antbar, "time_sp_fill", TW_TYPE_FLOAT, &time_spot_fill, " label='Spot fill time' group='Benchmark'");
	TwAddVarRO(antbar, "time_geometry", TW_TYPE_FLOAT, &time_geometry, " label='GBuffer time' group='Benchmark'");
	TwAddVarRO(antbar, "time_shading", TW_TYPE_FLOAT, &time_shading, " label='Shading time' group='Benchmark'");
	TwAddVarRO(antbar, "time_tot_LA", TW_TYPE_FLOAT, &time_tot_LA, " label='Total LA' group='Benchmark'");
	TwAddVarRO(antbar, "time_tot", TW_TYPE_FLOAT, &time_tot, " label='Total GPU time' group='Benchmark'");
	TwAddVarRO(antbar, "m_fps", TW_TYPE_UINT32, &m_fps, " label='FPS' group='Benchmark'");
}



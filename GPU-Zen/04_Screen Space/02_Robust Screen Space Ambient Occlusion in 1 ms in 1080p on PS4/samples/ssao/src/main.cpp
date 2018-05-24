#include <main.h>
#include <namespaces.h>


struct MeshConstantBuffer
{
	Matrix worldTransform;
	Matrix viewProjTransform;
};


bool fullScreen = false;
int screenWidth = 1280;
int screenHeight = 720;

int viewMode = 0; // 0 - diffuse with SSAO, 1 - diffuse, 2 - SSAO, 3 - raw SSAO
int ssaoVariant = 0;
float ssaoRadius_world = 1.0f;
float ssaoMaxRadius_screen = 0.1f;
float ssaoContrast = 4.0f;


Application application;

Profiler profiler;

RenderTarget gbufferDiffuseRT;
RenderTarget gbufferNormalRT;
RenderTarget depth16RT_x4;
RenderTarget ssaoRT_x4;
RenderTarget ssaoBlurXRT_x4;
RenderTarget ssaoBlurRT_x4;
RenderTarget ssaoUpsampleRT;
RenderTarget compositeRT;
DepthStencilTarget depthStencilTarget;

ID3D11InputLayout* meshIL = nullptr;

ID3D11Buffer* meshVS_CB = nullptr;

ID3D11VertexShader* meshVS = nullptr;

ID3D11PixelShader* meshGBufferPS = nullptr;
ID3D11PixelShader* downsamplePS = nullptr;
ID3D11PixelShader* ssaoPS[4] = { nullptr, nullptr, nullptr, nullptr };
ID3D11PixelShader* ssaoBlurPS = nullptr;
ID3D11PixelShader* ssaoUpsamplePS = nullptr;
ID3D11PixelShader* compositePS = nullptr;

ID3D11SamplerState* pointClampSamplerState = nullptr;
ID3D11SamplerState* linearClampSamplerState = nullptr;
ID3D11SamplerState* anisotropicWrapSamplerState = nullptr;

ID3D11RasterizerState* rasterizerState = nullptr;

NUtils::Scene sponzaScene;
map<string, NUtils::Texture> textures;
MeshConstantBuffer meshConstantBuffer;

Camera camera;


void Log(const string& msg)
{
	cout << msg << endl;
}


void CreateShaders()
{
	gpuUtils.Create("../../../../");

	ASSERT_FUNCTION(CreateVertexShader("../../data/mesh_vs.hlsl", meshVS));

	ASSERT_FUNCTION(CreatePixelShader("../../data/mesh_gbuffer_ps.hlsl", meshGBufferPS));
	ASSERT_FUNCTION(CreatePixelShader("../../data/downsample_ps.hlsl", downsamplePS));
	ASSERT_FUNCTION(CreatePixelShader("../../data/ssao_ps.hlsl", "VARIANT=1", ssaoPS[0]));
	ASSERT_FUNCTION(CreatePixelShader("../../data/ssao_ps.hlsl", "VARIANT=2", ssaoPS[1]));
	ASSERT_FUNCTION(CreatePixelShader("../../data/ssao_ps.hlsl", "VARIANT=3", ssaoPS[2]));
	ASSERT_FUNCTION(CreatePixelShader("../../data/ssao_ps.hlsl", "VARIANT=4", ssaoPS[3]));
	ASSERT_FUNCTION(CreatePixelShader("../../data/ssao_blur_ps.hlsl", ssaoBlurPS));
	ASSERT_FUNCTION(CreatePixelShader("../../data/ssao_upsample_ps.hlsl", ssaoUpsamplePS));
	ASSERT_FUNCTION(CreatePixelShader("../../data/composite_ps.hlsl", compositePS));
}


void DestroyShaders()
{
	DestroyPixelShader(meshGBufferPS);
	DestroyPixelShader(downsamplePS);
	for (int i = 0; i < ARRAY_SIZE(ssaoPS); i++)
		DestroyPixelShader(ssaoPS[i]);
	DestroyPixelShader(ssaoBlurPS);
	DestroyPixelShader(ssaoUpsamplePS);
	DestroyPixelShader(compositePS);

	DestroyVertexShader(meshVS);

	gpuUtils.Destroy();
}


bool Create()
{
	CreateD3D11(screenWidth, screenHeight);

	profiler.Create();
	profiler.AddQuery("All");

	DXGI_FORMAT floatFormat = DXGI_FORMAT_R11G11B10_FLOAT;
	CreateRenderTarget(screenWidth, screenHeight, DXGI_FORMAT_R8G8B8A8_UNORM, gbufferDiffuseRT);
	CreateRenderTarget(screenWidth, screenHeight, DXGI_FORMAT_R8G8B8A8_UNORM, gbufferNormalRT);
	CreateRenderTarget(screenWidth/2, screenHeight/2, DXGI_FORMAT_R16_FLOAT, depth16RT_x4);
	CreateRenderTarget(screenWidth/2, screenHeight/2, DXGI_FORMAT_R8G8B8A8_UNORM, ssaoRT_x4);
	CreateRenderTarget(screenWidth/2, screenHeight/2, DXGI_FORMAT_R8G8B8A8_UNORM, ssaoBlurXRT_x4);
	CreateRenderTarget(screenWidth/2, screenHeight/2, DXGI_FORMAT_R8G8B8A8_UNORM, ssaoBlurRT_x4);
	CreateRenderTarget(screenWidth, screenHeight, DXGI_FORMAT_R8G8B8A8_UNORM, ssaoUpsampleRT);
	CreateRenderTarget(screenWidth, screenHeight, floatFormat, compositeRT);
	CreateDepthStencilTarget(screenWidth, screenHeight, depthStencilTarget);

	D3D11_INPUT_ELEMENT_DESC inputLayoutElements[] =
	{
		{ "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 },
		{ "NORMAL", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 12, D3D11_INPUT_PER_VERTEX_DATA, 0 },
		{ "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 24, D3D11_INPUT_PER_VERTEX_DATA, 0 },
	};
	CreateInputLayout("../../data/mesh_vs.hlsl", inputLayoutElements, ARRAY_SIZE(inputLayoutElements), meshIL);

	CreateConstantBuffer(sizeof(meshConstantBuffer), meshVS_CB);

	CreateShaders();

	CreateSamplerState(pointClampSamplerState, SamplerFilter::Point, SamplerAddressing::Clamp, SamplerComparisonFunction::None);
	CreateSamplerState(linearClampSamplerState, SamplerFilter::Linear, SamplerAddressing::Clamp, SamplerComparisonFunction::None);
	CreateSamplerState(anisotropicWrapSamplerState, SamplerFilter::Anisotropic, SamplerAddressing::Wrap, SamplerComparisonFunction::None);

	CreateRasterizerState(rasterizerState);

	sponzaScene.CreateFromFile("../../../common/data/sponza/sponza.obj", &textures);

	camera.UpdateFixed(VectorCustom(0.0f, 50.0f, 50.0f), VectorCustom(0.0f, 0.0f, 0.0f));

	return true;
}


void Destroy()
{
	for (auto it = textures.begin(); it != textures.end(); it++)
		it->second.Destroy();
	sponzaScene.Destroy();

	DestroyRasterizerState(rasterizerState);

	DestroySamplerState(pointClampSamplerState);
	DestroySamplerState(linearClampSamplerState);
	DestroySamplerState(anisotropicWrapSamplerState);

	DestroyShaders();

	DestroyBuffer(meshVS_CB);

	DestroyInputLayout(meshIL);

	DestroyRenderTarget(gbufferDiffuseRT);
	DestroyRenderTarget(gbufferNormalRT);
	DestroyRenderTarget(depth16RT_x4);
	DestroyRenderTarget(ssaoRT_x4);
	DestroyRenderTarget(ssaoBlurXRT_x4);
	DestroyRenderTarget(ssaoBlurRT_x4);
	DestroyRenderTarget(ssaoUpsampleRT);
	DestroyRenderTarget(compositeRT);
	DestroyDepthStencilTarget(depthStencilTarget);

	profiler.Destroy();

	DestroyD3D11();
}


void DumpVogelDiskSamples()
{
	vector<Vector2> samples = VogelDiskSamples(16);

	for (int i = 0; i < 16; i++)
	{
		string s = "float2(" + ToString(samples[i].x) + "f, " + ToString(samples[i].y) + "f),\n";
		OutputDebugStringA(s.c_str());
	}
}


void DumpAlchemySpiralSamples()
{
	vector<Vector2> samples = AlchemySpiralSamples(16, 7);

	for (int i = 0; i < 16; i++)
	{
		string s = "float2(" + ToString(samples[i].x) + "f, " + ToString(samples[i].y) + "f),\n";
		OutputDebugStringA(s.c_str());
	}
}


void SSAO(const Matrix& viewTransform, const Matrix& projTransform, int variant, float radius_world, float maxRadius_screen, float contrast)
{
	ScopedProfilerQuery query(profiler, "All");

	SetViewport(screenWidth/2, screenHeight/2);

	// downsample
	{
		ScopedProfilerQuery query(profiler, "Downsample");

		deviceContext->OMSetRenderTargets(1, &depth16RT_x4.rtv, nullptr);

		deviceContext->VSSetShader(gpuUtils.screenQuadVS, nullptr, 0);
		deviceContext->PSSetShader(downsamplePS, nullptr, 0);
		deviceContext->PSSetShaderResources(0, 1, &depthStencilTarget.srv);

		deviceContext->IASetIndexBuffer(gpuUtils.screenQuadIB, DXGI_FORMAT_R16_UINT, 0);

		struct Params
		{
			Vector2 pixelSize;
			Vector2 projParams;
		} params;
		params.pixelSize = VectorCustom(2.0f/(float)screenWidth, 2.0f/(float)screenHeight);
		params.projParams = VectorCustom(projTransform.m[2][2], projTransform.m[3][2]);
		deviceContext->UpdateSubresource(gpuUtils.oneVectorCB, 0, nullptr, &params, 0, 0);
		deviceContext->PSSetConstantBuffers(0, 1, &gpuUtils.oneVectorCB);

		deviceContext->DrawIndexed(6, 0, 0);
	}

	// computation
	{
		ScopedProfilerQuery query(profiler, "Computation");

		deviceContext->OMSetRenderTargets(1, &ssaoRT_x4.rtv, nullptr);

		deviceContext->VSSetShader(gpuUtils.screenQuadVS, nullptr, 0);
		deviceContext->PSSetShader(ssaoPS[variant], nullptr, 0);
		deviceContext->PSSetShaderResources(0, 1, &depth16RT_x4.srv);
		deviceContext->PSSetShaderResources(1, 1, &gbufferNormalRT.srv);

		deviceContext->IASetIndexBuffer(gpuUtils.screenQuadIB, DXGI_FORMAT_R16_UINT, 0);

		struct Params
		{
			Vector2 pixelSize;
			Vector2 nearPlaneSize_normalized;
			Matrix viewTransform;
			float aspect;
			float radius_world;
			float maxRadius_screen;
			float contrast;
		} params;
		params.pixelSize = VectorCustom(2.0f/(float)screenWidth, 2.0f/(float)screenHeight);
		params.nearPlaneSize_normalized = PlaneSize(1.0f, Pi/3.0f, (float)screenWidth/(float)screenHeight);
		params.viewTransform = viewTransform;
		params.aspect = (float)screenWidth / (float)screenHeight;
		params.radius_world = radius_world;
		params.maxRadius_screen = maxRadius_screen;
		params.contrast = contrast;
		deviceContext->UpdateSubresource(gpuUtils.sixVectorsCB, 0, nullptr, &params, 0, 0);
		deviceContext->PSSetConstantBuffers(0, 1, &gpuUtils.sixVectorsCB);

		deviceContext->DrawIndexed(6, 0, 0);
	}

	// blur X
	{
		ScopedProfilerQuery query(profiler, "Blur X");

		deviceContext->OMSetRenderTargets(1, &ssaoBlurXRT_x4.rtv, nullptr);

		deviceContext->VSSetShader(gpuUtils.screenQuadVS, nullptr, 0);
		deviceContext->PSSetShader(ssaoBlurPS, nullptr, 0);
		deviceContext->PSSetShaderResources(0, 1, &depth16RT_x4.srv);
		deviceContext->PSSetShaderResources(1, 1, &ssaoRT_x4.srv);

		deviceContext->IASetIndexBuffer(gpuUtils.screenQuadIB, DXGI_FORMAT_R16_UINT, 0);

		struct Params
		{
			Vector2 pixelOffset;
			Vector2 padding;
		} params;
		params.pixelOffset = VectorCustom(2.0f/(float)screenWidth, 0.0f);
		deviceContext->UpdateSubresource(gpuUtils.oneVectorCB, 0, nullptr, &params, 0, 0);
		deviceContext->PSSetConstantBuffers(0, 1, &gpuUtils.oneVectorCB);

		deviceContext->DrawIndexed(6, 0, 0);
	}

	// blur Y
	{
		ScopedProfilerQuery query(profiler, "Blur Y");

		deviceContext->OMSetRenderTargets(1, &ssaoBlurRT_x4.rtv, nullptr);

		deviceContext->VSSetShader(gpuUtils.screenQuadVS, nullptr, 0);
		deviceContext->PSSetShader(ssaoBlurPS, nullptr, 0);
		deviceContext->PSSetShaderResources(0, 1, &depth16RT_x4.srv);
		deviceContext->PSSetShaderResources(1, 1, &ssaoBlurXRT_x4.srv);

		deviceContext->IASetIndexBuffer(gpuUtils.screenQuadIB, DXGI_FORMAT_R16_UINT, 0);

		struct Params
		{
			Vector2 pixelOffset;
			Vector2 padding;
		} params;
		params.pixelOffset = VectorCustom(0.0f, 2.0f/(float)screenHeight);
		deviceContext->UpdateSubresource(gpuUtils.oneVectorCB, 0, nullptr, &params, 0, 0);
		deviceContext->PSSetConstantBuffers(0, 1, &gpuUtils.oneVectorCB);

		deviceContext->DrawIndexed(6, 0, 0);
	}

	SetViewport(screenWidth, screenHeight);

	// upsample
	{
		ScopedProfilerQuery query(profiler, "Upsample");

		deviceContext->OMSetRenderTargets(1, &ssaoUpsampleRT.rtv, nullptr);

		deviceContext->VSSetShader(gpuUtils.screenQuadVS, nullptr, 0);
		deviceContext->PSSetShader(ssaoUpsamplePS, nullptr, 0);
		deviceContext->PSSetShaderResources(0, 1, &depthStencilTarget.srv);
		deviceContext->PSSetShaderResources(1, 1, &depth16RT_x4.srv);
		deviceContext->PSSetShaderResources(2, 1, &ssaoBlurRT_x4.srv);

		deviceContext->IASetIndexBuffer(gpuUtils.screenQuadIB, DXGI_FORMAT_R16_UINT, 0);

		struct Params
		{
			Vector2 pixelSize;
			Vector2 projParams;
		} params;
		params.pixelSize = VectorCustom(1.0f/(float)screenWidth, 1.0f/(float)screenHeight);
		params.projParams = VectorCustom(projTransform.m[2][2], projTransform.m[3][2]);
		deviceContext->UpdateSubresource(gpuUtils.oneVectorCB, 0, nullptr, &params, 0, 0);
		deviceContext->PSSetConstantBuffers(0, 1, &gpuUtils.oneVectorCB);

		deviceContext->DrawIndexed(6, 0, 0);
	}
}


bool Run()
{
	float lastFrameTime = application.LastFrameTime();

	//

	Vector3 eye;

	float speed = 0.025f * lastFrameTime;
	if (application.IsKeyDown(Keys::LShift))
		speed *= 8.0f;

	eye = camera.eye;
	if (application.IsKeyDown(Keys::W))
		eye = eye + speed*camera.forwardVector;
	if (application.IsKeyDown(Keys::S))
		eye = eye - speed*camera.forwardVector;
	if (application.IsKeyDown(Keys::A))
		eye = eye - speed*camera.rightVector;
	if (application.IsKeyDown(Keys::D))
		eye = eye + speed*camera.rightVector;

	camera.horizontalAngle -= application.MouseRelX() / 1000.0f;
	camera.verticalAngle -= application.MouseRelY() / 1000.0f;

	camera.UpdateFree(eye);

	//

	if (application.IsKeyDown(Keys::F1))
		viewMode = 0;
	if (application.IsKeyDown(Keys::F2))
		viewMode = 1;
	if (application.IsKeyDown(Keys::F3))
		viewMode = 2;
	if (application.IsKeyDown(Keys::F4))
		viewMode = 3;

	if (application.IsKeyDown(Keys::F5))
		ssaoVariant = 0;
	if (application.IsKeyDown(Keys::F6))
		ssaoVariant = 1;
	if (application.IsKeyDown(Keys::F7))
		ssaoVariant = 2;
	if (application.IsKeyDown(Keys::F8))
		ssaoVariant = 3;

	if (application.IsKeyDown(Keys::Insert))
		ssaoRadius_world += 0.001f * lastFrameTime;
	if (application.IsKeyDown(Keys::Delete))
		ssaoRadius_world -= 0.001f * lastFrameTime;

	if (application.IsKeyDown(Keys::Home))
		ssaoMaxRadius_screen += 0.001f * lastFrameTime;
	if (application.IsKeyDown(Keys::End))
		ssaoMaxRadius_screen -= 0.001f * lastFrameTime;

	if (application.IsKeyDown(Keys::PageUp))
		ssaoContrast += 0.005f * lastFrameTime;
	if (application.IsKeyDown(Keys::PageDown))
		ssaoContrast -= 0.005f * lastFrameTime;

	//

	profiler.StartFrame();

	float backgroundColor[] = { 0.2f, 0.2f, 0.2f, 1.0f };

	deviceContext->RSSetState(rasterizerState);
	deviceContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

	Matrix viewTransform = MatrixLookAtRH(camera.eye, camera.at, camera.up);
	Matrix projTransform = MatrixPerspectiveFovRH(ZRange::ZeroToOne, Pi/3.0f, (float)screenWidth/(float)screenHeight, 1.0f, 1000.0f);

	SetViewport(screenWidth, screenHeight);

	deviceContext->PSSetSamplers(0, 1, &pointClampSamplerState);
	deviceContext->PSSetSamplers(1, 1, &linearClampSamplerState);
	deviceContext->PSSetSamplers(2, 1, &anisotropicWrapSamplerState);

	// gbuffer
	{
		ID3D11RenderTargetView* rtvs[] = { gbufferDiffuseRT.rtv, gbufferNormalRT.rtv };
		deviceContext->OMSetRenderTargets(2, rtvs, depthStencilTarget.dsv);
		deviceContext->ClearDepthStencilView(depthStencilTarget.dsv, D3D11_CLEAR_DEPTH, 1.0f, 0);

		deviceContext->VSSetShader(meshVS, nullptr, 0);
		deviceContext->PSSetShader(meshGBufferPS, nullptr, 0);

		deviceContext->IASetInputLayout(meshIL);

		meshConstantBuffer.worldTransform = MatrixScale(0.1f, 0.1f, 0.1f);
		meshConstantBuffer.viewProjTransform = viewTransform * projTransform;
		deviceContext->UpdateSubresource(meshVS_CB, 0, nullptr, &meshConstantBuffer, 0, 0);
		deviceContext->VSSetConstantBuffers(0, 1, &meshVS_CB);

		for (uint i = 0; i < sponzaScene.meshes.size(); i++)
		{
			UINT stride = sizeof(NUtils::Vertex);
			UINT offset = 0;
			deviceContext->IASetVertexBuffers(0, 1, &sponzaScene.meshes[i].vb, &stride, &offset);
			deviceContext->IASetIndexBuffer(sponzaScene.meshes[i].ib, DXGI_FORMAT_R16_UINT, 0);

			auto texture = textures.find(sponzaScene.meshes[i].textureFileName);
			if (texture != textures.end())
				deviceContext->PSSetShaderResources(0, 1, &texture->second.texture.srv);

			deviceContext->DrawIndexed(sponzaScene.meshes[i].indicesCount, 0, 0);
		}
	}

	// SSAO
	{
		SSAO(viewTransform, projTransform, ssaoVariant, ssaoRadius_world, ssaoMaxRadius_screen, ssaoContrast);
	}

	SetViewport(screenWidth, screenHeight);

	// composite diffuse with SSAO
	{
		deviceContext->OMSetRenderTargets(1, &compositeRT.rtv, nullptr);

		deviceContext->VSSetShader(gpuUtils.screenQuadVS, nullptr, 0);
		deviceContext->PSSetShader(compositePS, nullptr, 0);
		deviceContext->PSSetShaderResources(0, 1, &gbufferDiffuseRT.srv);
		deviceContext->PSSetShaderResources(1, 1, &ssaoUpsampleRT.srv);

		deviceContext->IASetIndexBuffer(gpuUtils.screenQuadIB, DXGI_FORMAT_R16_UINT, 0);

		deviceContext->DrawIndexed(6, 0, 0);
	}

	// render full-screen quad
	{
		deviceContext->OMSetRenderTargets(1, &backBufferRTV, nullptr);

		deviceContext->VSSetShader(gpuUtils.screenQuadVS, nullptr, 0);
		deviceContext->PSSetShader(gpuUtils.copyTexturePS, nullptr, 0);
		if (viewMode == 0)
			deviceContext->PSSetShaderResources(0, 1, &compositeRT.srv);
		else if (viewMode == 1)
			deviceContext->PSSetShaderResources(0, 1, &gbufferDiffuseRT.srv);
		else if (viewMode == 2)
			deviceContext->PSSetShaderResources(0, 1, &ssaoUpsampleRT.srv);
		else if (viewMode == 3)
			deviceContext->PSSetShaderResources(0, 1, &ssaoRT_x4.srv);

		deviceContext->IASetIndexBuffer(gpuUtils.screenQuadIB, DXGI_FORMAT_R16_UINT, 0);

		deviceContext->DrawIndexed(6, 0, 0);
	}

	ID3D11ShaderResourceView* nullSRVS[] = { nullptr, nullptr, nullptr, nullptr, nullptr, nullptr };
	deviceContext->PSSetShaderResources(0, 6, nullSRVS);

	swapChain->Present(0, 0);

	profiler.EndFrame();
	profiler.StopProfiling();
	if (!profiler.isProfiling)
		profiler.StartProfiling();

	//

	if (application.IsKeyDown(Keys::F11))
	{
		DestroyShaders();
		CreateShaders();
	}

	if (application.IsKeyDown(Keys::Escape))
		return false;

	//

	return true;
}


void KeyDownFunction(Keys key)
{
	if (key == Keys::Space)
		profiler.StartProfiling();
}


void LoadConfigFile()
{
	File file;
	string temp;

	if (file.Open("config.txt", File::OpenMode::ReadText))
	{
		file.ReadText(temp);
		file.ReadText(fullScreen);
		file.ReadText(temp);
		file.ReadText(screenWidth);
		file.ReadText(temp);
		file.ReadText(screenHeight);

		file.Close();
	}
	else
	{
		file.Open("config.txt", File::OpenMode::WriteText);
		
		file.WriteText("FullScreen 0\n");
		file.WriteText("ScreenWidth 1280\n");
		file.WriteText("ScreenHeight 720\n");

		file.Close();
	}
}


int main()
{
	LoadConfigFile();

	if (screenWidth == 0 || screenHeight == 0)
		NSystem::ScreenSize(screenWidth, screenHeight);

	NImage::Initialize();
	NImage::SetFreeImageCustomOutputMessageFunction(Log);

	if (!application.Create(screenWidth, screenHeight, fullScreen))
		return 1;
	application.SetKeyDownFunction(KeyDownFunction);
	application.ShowCursor(false);

	Create();
	application.Run(Run);
	Destroy();

	application.Destroy();

	NImage::Deinitialize();

	return 0;
}

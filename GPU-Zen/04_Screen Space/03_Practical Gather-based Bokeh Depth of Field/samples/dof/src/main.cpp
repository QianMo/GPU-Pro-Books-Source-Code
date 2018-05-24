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

float dofFocalPlaneDistance = 40.0f;
float dofFocusTransitionDistance = 20.0f;
float dofStrength = 1.0f;


Application application;

Profiler profiler;

RenderTarget colorRT;
RenderTarget cocRT;
RenderTarget cocRT_x4;
RenderTarget cocMaxXRT_x4;
RenderTarget cocMaxRT_x4;
RenderTarget cocNearBlurXRT_x4;
RenderTarget cocNearBlurRT_x4;
RenderTarget colorRT_x4;
RenderTarget colorMulCoCFarRT_x4;
RenderTarget dofNearRT_x4;
RenderTarget dofFarRT_x4;
RenderTarget dofNearFillRT_x4;
RenderTarget dofFarFillRT_x4;
RenderTarget dofCompositeRT;
DepthStencilTarget depthStencilTarget;

ID3D11InputLayout* meshIL = nullptr;

ID3D11Buffer* meshVS_CB = nullptr;

ID3D11VertexShader* meshVS = nullptr;

ID3D11PixelShader* meshPS = nullptr;
ID3D11PixelShader* downsamplePS = nullptr;
ID3D11PixelShader* dofCOCPS = nullptr;
ID3D11PixelShader* dofPS = nullptr;
ID3D11PixelShader* dofFillPS = nullptr;
ID3D11PixelShader* dofCompositePS = nullptr;

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

	ASSERT_FUNCTION(CreatePixelShader("../../data/mesh_ps.hlsl", meshPS));
	ASSERT_FUNCTION(CreatePixelShader("../../data/downsample_ps.hlsl", downsamplePS));
	ASSERT_FUNCTION(CreatePixelShader("../../data/dof_coc_ps.hlsl", dofCOCPS));
	ASSERT_FUNCTION(CreatePixelShader("../../data/dof_ps.hlsl", dofPS));
	ASSERT_FUNCTION(CreatePixelShader("../../data/dof_fill_ps.hlsl", dofFillPS));
	ASSERT_FUNCTION(CreatePixelShader("../../data/dof_composite_ps.hlsl", dofCompositePS));
}


void DestroyShaders()
{
	DestroyPixelShader(meshPS);
	DestroyPixelShader(downsamplePS);
	DestroyPixelShader(dofCOCPS);
	DestroyPixelShader(dofPS);
	DestroyPixelShader(dofFillPS);
	DestroyPixelShader(dofCompositePS);

	DestroyVertexShader(meshVS);

	gpuUtils.Destroy();
}


bool Create()
{
	CreateD3D11(screenWidth, screenHeight);

	profiler.Create();

	DXGI_FORMAT colorBufferFormat = DXGI_FORMAT_R11G11B10_FLOAT;
	CreateRenderTarget(screenWidth, screenHeight, colorBufferFormat, colorRT);
	CreateRenderTarget(screenWidth, screenHeight, DXGI_FORMAT_R8G8_UNORM, cocRT);
	CreateRenderTarget(screenWidth/2, screenHeight/2, DXGI_FORMAT_R8G8_UNORM, cocRT_x4);
	CreateRenderTarget(screenWidth/2, screenHeight/2, DXGI_FORMAT_R8_UNORM, cocMaxXRT_x4);
	CreateRenderTarget(screenWidth/2, screenHeight/2, DXGI_FORMAT_R8_UNORM, cocMaxRT_x4);
	CreateRenderTarget(screenWidth/2, screenHeight/2, DXGI_FORMAT_R8_UNORM, cocNearBlurXRT_x4);
	CreateRenderTarget(screenWidth/2, screenHeight/2, DXGI_FORMAT_R8_UNORM, cocNearBlurRT_x4);
	CreateRenderTarget(screenWidth/2, screenHeight/2, colorBufferFormat, colorRT_x4);
	CreateRenderTarget(screenWidth/2, screenHeight/2, colorBufferFormat, colorMulCoCFarRT_x4);
	CreateRenderTarget(screenWidth/2, screenHeight/2, colorBufferFormat, dofNearRT_x4);
	CreateRenderTarget(screenWidth/2, screenHeight/2, colorBufferFormat, dofFarRT_x4);
	CreateRenderTarget(screenWidth/2, screenHeight/2, colorBufferFormat, dofNearFillRT_x4);
	CreateRenderTarget(screenWidth/2, screenHeight/2, colorBufferFormat, dofFarFillRT_x4);
	CreateRenderTarget(screenWidth, screenHeight, colorBufferFormat, dofCompositeRT);
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

	DestroyRenderTarget(colorRT);
	DestroyRenderTarget(cocRT);
	DestroyRenderTarget(cocRT_x4);
	DestroyRenderTarget(cocMaxXRT_x4);
	DestroyRenderTarget(cocMaxRT_x4);
	DestroyRenderTarget(cocNearBlurXRT_x4);
	DestroyRenderTarget(cocNearBlurRT_x4);
	DestroyRenderTarget(colorRT_x4);
	DestroyRenderTarget(colorMulCoCFarRT_x4);
	DestroyRenderTarget(dofNearRT_x4);
	DestroyRenderTarget(dofFarRT_x4);
	DestroyRenderTarget(dofNearFillRT_x4);
	DestroyRenderTarget(dofFarFillRT_x4);
	DestroyRenderTarget(dofCompositeRT);
	DestroyDepthStencilTarget(depthStencilTarget);

	profiler.Destroy();

	DestroyD3D11();
}


void DOF(float focalPlaneDistance, float focusTransitionRange, float strength, const Matrix& projTransform)
{
	ScopedProfilerQuery query(profiler, "All");

	float kernelScale = 1.0;
	float compositeBlend = 1.0;

	if (strength >= 0.25f)
	{
		kernelScale = strength;
		compositeBlend = 1.0f;
	}
	else
	{
		kernelScale = 0.25f;
		compositeBlend = 4.0f * strength;
	}

	// circle of confusion
	{
		ScopedProfilerQuery query(profiler, "CoC generation");

		deviceContext->OMSetRenderTargets(1, &cocRT.rtv, nullptr);

		deviceContext->VSSetShader(gpuUtils.screenQuadVS, nullptr, 0);
		deviceContext->PSSetShader(dofCOCPS, nullptr, 0);
		deviceContext->PSSetShaderResources(0, 1, &depthStencilTarget.srv);

		deviceContext->IASetIndexBuffer(gpuUtils.screenQuadIB, DXGI_FORMAT_R16_UINT, 0);

		float nearBegin = focalPlaneDistance - focusTransitionRange;
		if (nearBegin < 0.0f)
			nearBegin = 0.0f;
		float nearEnd = focalPlaneDistance;
		float farBegin = focalPlaneDistance;
		float farEnd = focalPlaneDistance + focusTransitionRange;

		float buffer[8] =
		{
			nearBegin, nearEnd, farBegin, farEnd,
			projTransform.m[2][2], projTransform.m[3][2], 0.0f, 0.0f
		};
		deviceContext->UpdateSubresource(gpuUtils.twoVectorsCB, 0, nullptr, &buffer, 0, 0);
		deviceContext->PSSetConstantBuffers(0, 1, &gpuUtils.twoVectorsCB);

		deviceContext->DrawIndexed(6, 0, 0);
	}

	SetViewport(screenWidth/2, screenHeight/2);

	// downsample
	{
		ScopedProfilerQuery query(profiler, "Downsample");

		ID3D11RenderTargetView* downsampleRTVs[] = { colorRT_x4.rtv, colorMulCoCFarRT_x4.rtv, cocRT_x4.rtv };
		deviceContext->OMSetRenderTargets(3, downsampleRTVs, nullptr);

		deviceContext->VSSetShader(gpuUtils.screenQuadVS, nullptr, 0);
		deviceContext->PSSetShader(downsamplePS, nullptr, 0);
		deviceContext->PSSetShaderResources(0, 1, &colorRT.srv);
		deviceContext->PSSetShaderResources(1, 1, &cocRT.srv);

		deviceContext->IASetIndexBuffer(gpuUtils.screenQuadIB, DXGI_FORMAT_R16_UINT, 0);

		Vector4 pixelSize = VectorCustom(2.0f/(float)screenWidth, 2.0f/(float)screenHeight, 0.0f, 0.0f);
		deviceContext->UpdateSubresource(gpuUtils.oneVectorCB, 0, nullptr, &pixelSize, 0, 0);
		deviceContext->PSSetConstantBuffers(0, 1, &gpuUtils.oneVectorCB);

		deviceContext->DrawIndexed(6, 0, 0);
	}

	// near coc max X
	{
		ScopedProfilerQuery query(profiler, "Near CoC Max X");
		gpuUtils.MaxX(cocMaxXRT_x4, cocRT_x4, 1, VectorCustom(2.0f/(float)screenWidth, 2.0f/(float)screenHeight), -6, 6);
	}

	// near coc max Y
	{
		ScopedProfilerQuery query(profiler, "Near CoC Max Y");
		gpuUtils.MaxY(cocMaxRT_x4, cocMaxXRT_x4, 1, VectorCustom(2.0f/(float)screenWidth, 2.0f/(float)screenHeight), -6, 6);
	}

	// near coc blur X
	{
		ScopedProfilerQuery query(profiler, "Near CoC Blur X");
		gpuUtils.BlurX(cocNearBlurXRT_x4, cocMaxRT_x4, 1, VectorCustom(2.0f/(float)screenWidth, 2.0f/(float)screenHeight), -6, 6);
	}

	// near coc blur Y
	{
		ScopedProfilerQuery query(profiler, "Near CoC Blur Y");
		gpuUtils.BlurY(cocNearBlurRT_x4, cocNearBlurXRT_x4, 1, VectorCustom(2.0f/(float)screenWidth, 2.0f/(float)screenHeight), -6, 6);
	}

	// computation
	{
		ScopedProfilerQuery query(profiler, "Computation");

		ID3D11RenderTargetView* rtvs[] = { dofNearRT_x4.rtv, dofFarRT_x4.rtv };
		deviceContext->OMSetRenderTargets(2, rtvs, nullptr);

		deviceContext->VSSetShader(gpuUtils.screenQuadVS, nullptr, 0);
		deviceContext->PSSetShader(dofPS, nullptr, 0);
		deviceContext->PSSetShaderResources(0, 1, &cocRT_x4.srv);
		deviceContext->PSSetShaderResources(1, 1, &cocNearBlurRT_x4.srv);
		deviceContext->PSSetShaderResources(2, 1, &colorRT_x4.srv);
		deviceContext->PSSetShaderResources(3, 1, &colorMulCoCFarRT_x4.srv);

		deviceContext->IASetIndexBuffer(gpuUtils.screenQuadIB, DXGI_FORMAT_R16_UINT, 0);

		Vector4 pixelSize_dofKernelScale = VectorCustom(2.0f/(float)screenWidth, 2.0f/(float)screenHeight, kernelScale, 0.0f);
		deviceContext->UpdateSubresource(gpuUtils.oneVectorCB, 0, nullptr, &pixelSize_dofKernelScale, 0, 0);
		deviceContext->PSSetConstantBuffers(0, 1, &gpuUtils.oneVectorCB);

		deviceContext->DrawIndexed(6, 0, 0);
	}

	// fill
	{
		ScopedProfilerQuery query(profiler, "Fill");

		ID3D11RenderTargetView* rtvs[] = { dofNearFillRT_x4.rtv, dofFarFillRT_x4.rtv };
		deviceContext->OMSetRenderTargets(2, rtvs, nullptr);

		deviceContext->VSSetShader(gpuUtils.screenQuadVS, nullptr, 0);
		deviceContext->PSSetShader(dofFillPS, nullptr, 0);
		deviceContext->PSSetShaderResources(0, 1, &cocRT_x4.srv);
		deviceContext->PSSetShaderResources(1, 1, &cocNearBlurRT_x4.srv);
		deviceContext->PSSetShaderResources(2, 1, &dofNearRT_x4.srv);
		deviceContext->PSSetShaderResources(3, 1, &dofFarRT_x4.srv);

		deviceContext->IASetIndexBuffer(gpuUtils.screenQuadIB, DXGI_FORMAT_R16_UINT, 0);

		Vector4 pixelSize = VectorCustom(2.0f/(float)screenWidth, 2.0f/(float)screenHeight, 0.0f, 0.0f);
		deviceContext->UpdateSubresource(gpuUtils.oneVectorCB, 0, nullptr, &pixelSize, 0, 0);
		deviceContext->PSSetConstantBuffers(0, 1, &gpuUtils.oneVectorCB);

		deviceContext->DrawIndexed(6, 0, 0);
	}

	SetViewport(screenWidth, screenHeight);

	// composite
	{
		ScopedProfilerQuery query(profiler, "Composite");

		deviceContext->OMSetRenderTargets(1, &dofCompositeRT.rtv, nullptr);

		deviceContext->VSSetShader(gpuUtils.screenQuadVS, nullptr, 0);
		deviceContext->PSSetShader(dofCompositePS, nullptr, 0);
		deviceContext->PSSetShaderResources(0, 1, &colorRT.srv);
		deviceContext->PSSetShaderResources(1, 1, &cocRT.srv);
		deviceContext->PSSetShaderResources(2, 1, &cocRT_x4.srv);
		deviceContext->PSSetShaderResources(3, 1, &cocNearBlurRT_x4.srv);
		deviceContext->PSSetShaderResources(4, 1, &dofNearFillRT_x4.srv);
		deviceContext->PSSetShaderResources(5, 1, &dofFarFillRT_x4.srv);

		deviceContext->IASetIndexBuffer(gpuUtils.screenQuadIB, DXGI_FORMAT_R16_UINT, 0);

		Vector4 pixelSize_dofCompositeBlend = VectorCustom(1.0f/(float)screenWidth, 1.0f/(float)screenHeight, compositeBlend, 0.0f);
		deviceContext->UpdateSubresource(gpuUtils.oneVectorCB, 0, nullptr, &pixelSize_dofCompositeBlend, 0, 0);
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

	if (application.IsKeyDown(Keys::Insert))
		dofFocalPlaneDistance += 0.025f * lastFrameTime;
	if (application.IsKeyDown(Keys::Delete))
		dofFocalPlaneDistance -= 0.025f * lastFrameTime;

	if (application.IsKeyDown(Keys::Home))
		dofFocusTransitionDistance += 0.025f * lastFrameTime;
	if (application.IsKeyDown(Keys::End))
		dofFocusTransitionDistance -= 0.025f * lastFrameTime;

	if (application.IsKeyDown(Keys::PageUp))
		dofStrength += 0.001f * lastFrameTime;
	if (application.IsKeyDown(Keys::PageDown))
		dofStrength -= 0.001f * lastFrameTime;
	dofStrength = Saturate(dofStrength);

	//

	profiler.StartFrame();

	float backgroundColor[] = { 0.2f, 0.2f, 0.2f, 1.0f };

	deviceContext->RSSetState(rasterizerState);
	deviceContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

	Matrix projTransform = MatrixPerspectiveFovRH(ZRange::ZeroToOne, Pi/3.0f, (float)screenWidth/(float)screenHeight, 1.0f, 1000.0f);

	SetViewport(screenWidth, screenHeight);

	deviceContext->PSSetSamplers(0, 1, &pointClampSamplerState);
	deviceContext->PSSetSamplers(1, 1, &linearClampSamplerState);
	deviceContext->PSSetSamplers(2, 1, &anisotropicWrapSamplerState);

	// render scene
	{
		deviceContext->OMSetRenderTargets(1, &colorRT.rtv, depthStencilTarget.dsv);
		deviceContext->ClearRenderTargetView(colorRT.rtv, backgroundColor);
		deviceContext->ClearDepthStencilView(depthStencilTarget.dsv, D3D11_CLEAR_DEPTH, 1.0f, 0);

		deviceContext->VSSetShader(meshVS, nullptr, 0);
		deviceContext->PSSetShader(meshPS, nullptr, 0);

		deviceContext->IASetInputLayout(meshIL);

		meshConstantBuffer.worldTransform = MatrixScale(0.1f, 0.1f, 0.1f);
		meshConstantBuffer.viewProjTransform =
			MatrixLookAtRH(camera.eye, camera.at, camera.up) *
			projTransform;
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

	// DoF
	{
		DOF(dofFocalPlaneDistance, dofFocusTransitionDistance, dofStrength, projTransform);
	}

	// render full-screen quad
	{
		deviceContext->OMSetRenderTargets(1, &backBufferRTV, nullptr);

		deviceContext->VSSetShader(gpuUtils.screenQuadVS, nullptr, 0);
		deviceContext->PSSetShader(gpuUtils.copyTexturePS, nullptr, 0);
		deviceContext->PSSetShaderResources(0, 1, &dofCompositeRT.srv);

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

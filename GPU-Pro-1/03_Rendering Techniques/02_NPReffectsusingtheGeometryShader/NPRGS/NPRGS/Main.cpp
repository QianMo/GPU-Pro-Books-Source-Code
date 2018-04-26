/**
 *	Pedro Hermosilla
 *	
 *	Moving Group - UPC
 *	Main.cpp
 */

#include <windows.h>
#include <d3d10.h>
#include <d3dx10.h>
#include <D3DX10Math.h>

#include "Mesh.h"

HINSTANCE								hInstance = NULL;
HWND									hWnd = NULL;
int										WIN_WIDTH;
int										WIN_HEIGHT;
bool									salir = false;
bool									renderSil = true;
bool									firstTexture = true;
bool									helpPannel = true;
bool									mesh1 = true;
bool									about = false;

ID3D10Device*							d3dDevice = NULL;
IDXGISwapChain*							dxSwapChain = NULL;
ID3D10RenderTargetView*					backbufferView = NULL;
ID3D10Texture2D*						depthBuffer = NULL;
ID3D10DepthStencilView*					depthView = NULL;
ID3DX10Font*							d3dxFont = NULL;
ID3DX10Sprite*							d3dxFontSprite = NULL;

ID3D10Effect*							silEffect = NULL;
ID3D10EffectTechnique*					silTechnique = NULL;
ID3D10InputLayout*						silLayout = NULL;
ID3D10InputLayout*						silLayout2 = NULL;
ID3D10Texture2D*						silTexture = NULL;
ID3D10ShaderResourceView*				silTextView = NULL;
ID3D10Texture2D*						silTexture2 = NULL;
ID3D10ShaderResourceView*				silTextView2 = NULL;

ID3D10Effect*							pencilEffect = NULL;
ID3D10EffectTechnique*					pencilTechnique = NULL;
ID3D10InputLayout*						pencilLayout = NULL;
ID3D10InputLayout*						pencilLayout2 = NULL;
ID3D10Texture2D*						pencilTexture = NULL;
ID3D10ShaderResourceView*				pencilTextView = NULL;
ID3D10Texture2D*						pencilTexture2 = NULL;
ID3D10ShaderResourceView*				pencilTextView2 = NULL;


Mesh*									currentMesh1 = NULL;
Mesh*									currentMesh2 = NULL;

ID3D10EffectMatrixVariable*				worldViewSilParam = NULL;
ID3D10EffectMatrixVariable*				projSilParam = NULL;
ID3D10EffectVectorVariable*				aabbSilParam = NULL;
ID3D10EffectScalarVariable*				edgeSizeSilParam = NULL;
ID3D10EffectScalarVariable*				lengthSilParam = NULL;
ID3D10EffectScalarVariable*				scaleSilParam = NULL;
ID3D10EffectShaderResourceVariable*		textureSilParam = NULL;

ID3D10EffectMatrixVariable*				worldViewProjPencilParam = NULL;
ID3D10EffectMatrixVariable*				worldPencilParam = NULL;
ID3D10EffectVectorVariable*				lightDirPencilParam = NULL;
ID3D10EffectScalarVariable*				widthPencilParam = NULL;
ID3D10EffectScalarVariable*				heightPencilParam = NULL;
ID3D10EffectScalarVariable*				textureRepeatParam = NULL;
ID3D10EffectScalarVariable*				OParam = NULL;
ID3D10EffectScalarVariable*				SParam = NULL;
ID3D10EffectShaderResourceVariable*		texturePencilParam = NULL;

D3DXMATRIX				worldMatrix;
D3DXMATRIX				viewMatrix;
D3DXMATRIX				projMatrix;

float auxXRotation = 0.0f;
float auxYRotation = 3.1416f;
float texRepeate = 4.0f;
float edgeSilSize = 0.01f;
float lengthFactor = 0.0f;
float OParamFloat = 7.0f;
float SParamFloat = 0.5f;

LRESULT CALLBACK	WndProc(HWND, UINT, WPARAM, LPARAM);
bool				InitWindow();
bool				InitDirectX();
bool				InitResources();
void				UpdateMatrixs();
void				clearApp();
void				Render();
void				DrawTextString(int x, int y, const WCHAR* strOutput);
void				ParseKeyboard();

int main()
{
	WIN_WIDTH = 800;
	WIN_HEIGHT = 800;

	hInstance = (HINSTANCE)GetModuleHandle(NULL);

	if(!InitWindow())
	{
		clearApp();
		return -1;
	}

	if(!InitDirectX())
	{
		clearApp();
		return -1;
	}

	if(!InitResources())
	{
		clearApp();
		return -1;
	}

	MSG msg;

	salir = false;
	DWORD time;
	DWORD lastTime = GetTickCount();
	int frameId = 0;

	while (!salir)
	{
		time = GetTickCount();
		while(((double)time-lastTime)/1000.0 < 1/60.0){
			time = GetTickCount();
		}
		lastTime = time;

		if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE) == TRUE)
		{
			TranslateMessage(&msg);
			DispatchMessage(&msg);
		}

		if(frameId%5 == 0)
			ParseKeyboard();

		Render();

		frameId = (frameId+1)%60;

	}

	clearApp();

	return (int) msg.wParam;
}

LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
	static POINTS lastPoint;
	switch (message)
	{
		case WM_DESTROY:
			salir = true;
			PostQuitMessage(0);
			break;
		case WM_MOUSEMOVE:
			if (wParam & MK_LBUTTON) 
            {
				POINTS p = MAKEPOINTS(lParam);
				auxXRotation += ((float)(lastPoint.y - p.y))/500.0f;
				auxYRotation += ((float)(lastPoint.x - p.x))/500.0f;
			}
			lastPoint = MAKEPOINTS(lParam);
			break;
		default:
			return DefWindowProc(hWnd, message, wParam, lParam);
	}
	return 0;
}

void ParseKeyboard()
{
	if(GetFocus() == hWnd)
	{
		if(GetAsyncKeyState(72))//H
		{
			helpPannel = !helpPannel;
			if(helpPannel)
				about = false;
		}

		if(GetAsyncKeyState(82))//R
			renderSil = !renderSil;

		if(GetAsyncKeyState(84))//T
			firstTexture = !firstTexture;

		if(GetAsyncKeyState(77))//M
			mesh1 = !mesh1;

		if(GetAsyncKeyState(66))//B
		{
			about = !about;
			helpPannel = false;
		}

		if(GetAsyncKeyState(81))//Q
		{
			edgeSilSize += 0.001f;
			edgeSilSize = (edgeSilSize > 0.03f)?0.03f:edgeSilSize;
		}

		if(GetAsyncKeyState(87))//W
		{
			edgeSilSize -= 0.001f;
			edgeSilSize = (edgeSilSize < 0.01f)?0.01f:edgeSilSize;
		}

		if(GetAsyncKeyState(65))//A
		{
			texRepeate += 0.1f;
			texRepeate = (texRepeate > 5.0f)?5.0f:texRepeate;
		}

		if(GetAsyncKeyState(83))//S
		{
			texRepeate -= 0.1f;
			texRepeate = (texRepeate < 1.0f)?1.0f:texRepeate;
		}

		if(GetAsyncKeyState(90))//Z
		{
			lengthFactor += 0.01f;
			lengthFactor = (lengthFactor > 1.0f)?1.0f:lengthFactor;
		}

		if(GetAsyncKeyState(88))//X
		{
			lengthFactor -= 0.01f;
			lengthFactor = (lengthFactor < 0.0f)?0.0f:lengthFactor;
		}

		if(GetAsyncKeyState(68))//D
		{
			OParamFloat += 0.5f;
			OParamFloat = (OParamFloat > 10.0f)?10.0f:OParamFloat;
		}

		if(GetAsyncKeyState(70))//F
		{
			OParamFloat -= 0.5f;
			OParamFloat = (OParamFloat < 2.0f)?2.0f:OParamFloat;
		}

		if(GetAsyncKeyState(67))//C
		{
			SParamFloat += 0.05f;
			SParamFloat = (SParamFloat > 1.0f)?1.0f:SParamFloat;
		}

		if(GetAsyncKeyState(86))//V
		{
			SParamFloat -= 0.05f;
			SParamFloat = (SParamFloat < 0.0f)?0.0f:SParamFloat;
		}

		if(GetAsyncKeyState(VK_ESCAPE))
			salir = true;
	}
}

bool InitWindow()
{
	WNDCLASSEX wcex;

	ZeroMemory(&wcex,sizeof(WNDCLASSEX));

	wcex.cbSize = sizeof(WNDCLASSEX);
	wcex.style			= CS_HREDRAW | CS_VREDRAW;
	wcex.lpfnWndProc	= (WNDPROC)WndProc;
	wcex.hInstance		= hInstance;
	wcex.hCursor		= NULL;
	wcex.hbrBackground = (HBRUSH)COLOR_WINDOW;
	wcex.lpszClassName	= L"MyWindowNPR";

	RegisterClassEx(&wcex);

	hWnd = CreateWindowEx(NULL,L"MyWindowNPR", L"GPUSilhouetteEdges ((c)Pedro Hermosilla)", WS_OVERLAPPEDWINDOW,
      0, 0, WIN_WIDTH, WIN_HEIGHT, NULL, NULL, hInstance, NULL);

	if (!hWnd)
		return false;

	ShowWindow(hWnd, SW_SHOW);
	UpdateWindow(hWnd);

	return true;
}

bool InitDirectX()
{
	DXGI_SWAP_CHAIN_DESC sd;
    ZeroMemory( &sd, sizeof(sd) );
	sd.Windowed = FALSE;
    sd.BufferCount = 1;
    sd.BufferDesc.Width = WIN_WIDTH;
    sd.BufferDesc.Height = WIN_HEIGHT;
    sd.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    sd.BufferDesc.RefreshRate.Numerator = 60;
    sd.BufferDesc.RefreshRate.Denominator = 1;
    sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    sd.OutputWindow = hWnd;
    sd.SampleDesc.Count = 1;
    sd.SampleDesc.Quality = 0;
    sd.Windowed = TRUE;

    if( FAILED( D3D10CreateDeviceAndSwapChain
		( NULL, D3D10_DRIVER_TYPE_HARDWARE, NULL,
        0, D3D10_SDK_VERSION, &sd, &dxSwapChain, &d3dDevice ) ) )
    {
        return false;
    }

	D3D10_VIEWPORT vp;
    vp.Width = WIN_WIDTH;
    vp.Height = WIN_HEIGHT;
    vp.MinDepth = 0.0f;
    vp.MaxDepth = 1.0f;
    vp.TopLeftX = 0;
    vp.TopLeftY = 0;
    d3dDevice->RSSetViewports( 1, &vp );

    ID3D10Texture2D* backBuffer;
    if(FAILED(dxSwapChain->GetBuffer( 0, __uuidof( ID3D10Texture2D ), (LPVOID*)&backBuffer)))
        return false;
    HRESULT hr = d3dDevice->CreateRenderTargetView( backBuffer, NULL, &backbufferView );
    backBuffer->Release();
    if( FAILED( hr ) )
        return false;

	// Create depth stencil texture
    D3D10_TEXTURE2D_DESC descDepth;
    descDepth.Width = WIN_WIDTH;
    descDepth.Height = WIN_HEIGHT;
    descDepth.MipLevels = 1;
    descDepth.ArraySize = 1;
    descDepth.Format = DXGI_FORMAT_D32_FLOAT;
    descDepth.SampleDesc.Count = 1;
    descDepth.SampleDesc.Quality = 0;
    descDepth.Usage = D3D10_USAGE_DEFAULT;
    descDepth.BindFlags = D3D10_BIND_DEPTH_STENCIL;
    descDepth.CPUAccessFlags = 0;
    descDepth.MiscFlags = 0;
    hr = d3dDevice->CreateTexture2D( &descDepth, NULL, &depthBuffer );
    if( FAILED(hr) )
        return false;

    // Create the depth stencil view
    D3D10_DEPTH_STENCIL_VIEW_DESC descDSV;
    descDSV.Format = descDepth.Format;
    descDSV.ViewDimension = D3D10_DSV_DIMENSION_TEXTURE2D;
    descDSV.Texture2D.MipSlice = 0;
    hr = d3dDevice->CreateDepthStencilView( depthBuffer, &descDSV, &depthView );
    if( FAILED(hr) )
        return false;

    d3dDevice->OMSetRenderTargets( 1, &backbufferView, depthView );

	D3DX10_FONT_DESC fontDesc = {
		24, 0, 200, 1,
		false,
		DEFAULT_CHARSET,
		OUT_TT_PRECIS,
		CLIP_DEFAULT_PRECIS,
		DEFAULT_PITCH,
		L"Times New Roman"
	};
	hr = D3DX10CreateFontIndirect(d3dDevice, &fontDesc, &d3dxFont);

	if( FAILED(hr) )
        return false;

	hr = D3DX10CreateSprite(d3dDevice,1,&d3dxFontSprite);
	
	if( FAILED(hr))
		return false;

	return true;
}

bool InitResources()
{
	currentMesh1 = new Mesh(d3dDevice);
	currentMesh1->load("bunny.obj");

	currentMesh2 = new Mesh(d3dDevice);
	currentMesh2->load("toad.obj");

	HRESULT hr = D3DX10CreateEffectFromFile(L"Silhouette.fx",NULL,NULL,"fx_4_0",0,0,d3dDevice,NULL,NULL,&silEffect,NULL,NULL);
	if( FAILED( hr ) )
        return false;

	silTechnique = silEffect->GetTechniqueByName("Silhouette");
	
	D3D10_PASS_DESC silPassDesc;
	silTechnique->GetPassByIndex(0)->GetDesc(&silPassDesc);
	hr = d3dDevice->CreateInputLayout(&currentMesh1->getVertexDescription()[0],currentMesh1->getVertexDescription().size(),
		silPassDesc.pIAInputSignature,silPassDesc.IAInputSignatureSize,&silLayout);

	if( FAILED( hr ) )
        return false;

	hr = d3dDevice->CreateInputLayout(&currentMesh2->getVertexDescription()[0],currentMesh2->getVertexDescription().size(),
		silPassDesc.pIAInputSignature,silPassDesc.IAInputSignatureSize,&silLayout2);

	if( FAILED( hr ) )
        return false;

	worldViewSilParam = silEffect->GetVariableByName( "WorldView" )->AsMatrix();
	projSilParam = silEffect->GetVariableByName( "projMatrix" )->AsMatrix();
	aabbSilParam = silEffect->GetVariableByName( "aabbPos" )->AsVector();
	edgeSizeSilParam = silEffect->GetVariableByName( "edgeSize" )->AsScalar();
	lengthSilParam = silEffect->GetVariableByName( "lengthPer" )->AsScalar();
	scaleSilParam = silEffect->GetVariableByName( "scale" )->AsScalar();
	textureSilParam = silEffect->GetVariableByName( "texureDiff" )->AsShaderResource();

	//Load the first texture.
	//Create texture info load
	D3DX10_IMAGE_LOAD_INFO loadInfo;
	ZeroMemory( &loadInfo, sizeof(D3DX10_IMAGE_LOAD_INFO) );
	loadInfo.MipLevels = 1;
	loadInfo.Usage = D3D10_USAGE_DEFAULT;
    loadInfo.BindFlags = D3D10_BIND_SHADER_RESOURCE;
	loadInfo.Format = DXGI_FORMAT_R8G8B8A8_TYPELESS;

	//Load texture
	if(FAILED(D3DX10CreateTextureFromFile( d3dDevice, L"Pincel.dds", &loadInfo
		, NULL, (ID3D10Resource**)&silTexture, NULL )))
		return false;

	//Create resource view description.
	D3D10_SHADER_RESOURCE_VIEW_DESC srvDesc;
	srvDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
	srvDesc.ViewDimension = D3D10_SRV_DIMENSION_TEXTURE2D;
	srvDesc.Texture2D.MostDetailedMip = 0;
	srvDesc.Texture2D.MipLevels = 1;

	//Create resource view.
	if(FAILED(d3dDevice->CreateShaderResourceView((ID3D10Resource*)silTexture
		,&srvDesc,&silTextView)))
		return false;

	//Load the second texture.
	//Create texture info load
	ZeroMemory( &loadInfo, sizeof(D3DX10_IMAGE_LOAD_INFO) );
	loadInfo.MipLevels = 1;
	loadInfo.Usage = D3D10_USAGE_DEFAULT;
    loadInfo.BindFlags = D3D10_BIND_SHADER_RESOURCE;
	loadInfo.Format = DXGI_FORMAT_R8G8B8A8_TYPELESS;

	//Load texture
	if(FAILED(D3DX10CreateTextureFromFile( d3dDevice, L"Pincel2.dds", &loadInfo
		, NULL, (ID3D10Resource**)&silTexture2, NULL )))
		return false;

	srvDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
	srvDesc.ViewDimension = D3D10_SRV_DIMENSION_TEXTURE2D;
	srvDesc.Texture2D.MostDetailedMip = 0;
	srvDesc.Texture2D.MipLevels = 1;

	//Create resource view.
	if(FAILED(d3dDevice->CreateShaderResourceView((ID3D10Resource*)silTexture2
		,&srvDesc,&silTextView2)))
		return false;

	hr = D3DX10CreateEffectFromFile(L"Pencil.fx",NULL,NULL,"fx_4_0",0,0,d3dDevice,NULL,NULL,&pencilEffect,NULL,NULL);
	if( FAILED( hr ) )
        return false;

	pencilTechnique = pencilEffect->GetTechniqueByName("Pencil");

	D3D10_PASS_DESC pencilPassDesc;
	pencilTechnique->GetPassByIndex(0)->GetDesc(&pencilPassDesc);
	hr = d3dDevice->CreateInputLayout(&currentMesh1->getVertexDescription()[0],currentMesh1->getVertexDescription().size(),
		pencilPassDesc.pIAInputSignature,pencilPassDesc.IAInputSignatureSize,&pencilLayout);

	if( FAILED( hr ) )
        return false;

	hr = d3dDevice->CreateInputLayout(&currentMesh2->getVertexDescription()[0],currentMesh2->getVertexDescription().size(),
		pencilPassDesc.pIAInputSignature,pencilPassDesc.IAInputSignatureSize,&pencilLayout2);

	if( FAILED( hr ) )
        return false;

	worldViewProjPencilParam = pencilEffect->GetVariableByName( "WorldViewProj" )->AsMatrix();
	worldPencilParam = pencilEffect->GetVariableByName( "world" )->AsMatrix();
	lightDirPencilParam = pencilEffect->GetVariableByName( "lightDir" )->AsVector();
	widthPencilParam = pencilEffect->GetVariableByName( "width" )->AsScalar();
	heightPencilParam = pencilEffect->GetVariableByName( "height" )->AsScalar();
	textureRepeatParam = pencilEffect->GetVariableByName( "texRepeat" )->AsScalar();
	OParam = pencilEffect->GetVariableByName( "OParam" )->AsScalar();
	SParam = pencilEffect->GetVariableByName( "SParam" )->AsScalar();
	texturePencilParam = pencilEffect->GetVariableByName( "texPencil" )->AsShaderResource();

	//Load the first texture.
	//Create texture info load
	ZeroMemory( &loadInfo, sizeof(D3DX10_IMAGE_LOAD_INFO) );
	loadInfo.MipLevels = 1;
	loadInfo.Usage = D3D10_USAGE_DEFAULT;
    loadInfo.BindFlags = D3D10_BIND_SHADER_RESOURCE;
	loadInfo.Format = DXGI_FORMAT_R8G8B8A8_TYPELESS;

	//Load texture
	if(FAILED(D3DX10CreateTextureFromFile( d3dDevice, L"hatch1.jpg", &loadInfo
		, NULL, (ID3D10Resource**)&pencilTexture, NULL )))
		return false;

	//Create resource view description.
	srvDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
	srvDesc.ViewDimension = D3D10_SRV_DIMENSION_TEXTURE2D;
	srvDesc.Texture2D.MostDetailedMip = 0;
	srvDesc.Texture2D.MipLevels = 1;

	//Create resource view.
	if(FAILED(d3dDevice->CreateShaderResourceView((ID3D10Resource*)pencilTexture
		,&srvDesc,&pencilTextView)))
		return false;

	//Load the second texture.
	//Create texture info load
	ZeroMemory( &loadInfo, sizeof(D3DX10_IMAGE_LOAD_INFO) );
	loadInfo.MipLevels = 1;
	loadInfo.Usage = D3D10_USAGE_DEFAULT;
    loadInfo.BindFlags = D3D10_BIND_SHADER_RESOURCE;
	loadInfo.Format = DXGI_FORMAT_R8G8B8A8_TYPELESS;

	//Load texture
	if(FAILED(D3DX10CreateTextureFromFile( d3dDevice, L"hatch2.jpg", &loadInfo
		, NULL, (ID3D10Resource**)&pencilTexture2, NULL )))
		return false;

	//Create resource view description.
	srvDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
	srvDesc.ViewDimension = D3D10_SRV_DIMENSION_TEXTURE2D;
	srvDesc.Texture2D.MostDetailedMip = 0;
	srvDesc.Texture2D.MipLevels = 1;

	//Create resource view.
	if(FAILED(d3dDevice->CreateShaderResourceView((ID3D10Resource*)pencilTexture2
		,&srvDesc,&pencilTextView2)))
		return false;

	return true;
}

void UpdateMatrixs()
{
	if(helpPannel){
		D3DXMatrixLookAtLH( &viewMatrix,
			&D3DXVECTOR3(-0.5f, 0.5f, -5.0f), 
			&D3DXVECTOR3(-0.5f, 0.5f, 0.0f),
			&D3DXVECTOR3(0.0f, 1.0f, 0.0f));

		D3DXMatrixPerspectiveFovLH(&projMatrix,D3DXToRadian(45),WIN_WIDTH/WIN_HEIGHT,1.0f,20.0f);

		D3DXMATRIX auxMatrix;
		D3DXMatrixRotationY(&auxMatrix,auxYRotation);
		D3DXMatrixRotationX(&worldMatrix,auxXRotation);
		worldMatrix = auxMatrix*worldMatrix;
	}else{
		D3DXMatrixLookAtLH( &viewMatrix,
			&D3DXVECTOR3(0.0f, 0.0f, -4.0f), 
			&D3DXVECTOR3(0.0f, 0.0f, 0.0f),
			&D3DXVECTOR3(0.0f, 1.0f, 0.0f));

		D3DXMatrixPerspectiveFovLH(&projMatrix,D3DXToRadian(45),WIN_WIDTH/WIN_HEIGHT,1.0f,20.0f);

		D3DXMATRIX auxMatrix;
		D3DXMatrixRotationY(&auxMatrix,auxYRotation);
		D3DXMatrixRotationX(&worldMatrix,auxXRotation);
		worldMatrix = auxMatrix*worldMatrix;
	}
}

void clearApp()
{
	if(d3dxFontSprite)
		d3dxFontSprite->Release();
	if(d3dxFont)
		d3dxFont->Release();
	if(pencilTextView2)
		pencilTextView2->Release();
	if(pencilTexture2)
		pencilTexture2->Release();
	if(silTextView2)
		silTextView2->Release();
	if(silTexture2)
		silTexture2->Release();
	if(pencilTextView)
		pencilTextView->Release();
	if(pencilTexture)
		pencilTexture->Release();
	if(silTextView)
		silTextView->Release();
	if(silTexture)
		silTexture->Release();
	if(silLayout)
		silLayout->Release();
	if(pencilLayout)
		pencilLayout->Release();
	if(silLayout2)
		silLayout2->Release();
	if(pencilLayout2)
		pencilLayout2->Release();
	if(currentMesh1)
		delete currentMesh1;
	if(currentMesh2)
		delete currentMesh2;
	if(silEffect)
		silEffect->Release();
	if(pencilEffect)
		pencilEffect->Release();
	if(depthBuffer)
		depthBuffer->Release();
	if(depthView)
		depthView->Release();
	if(backbufferView)
		backbufferView->Release();
	if(dxSwapChain)
		dxSwapChain->Release();
	if(d3dDevice)
		d3dDevice->Release();
}

void DrawTextString(int x, int y, const WCHAR* strOutput)
{
	LPCWSTR text = LPCWSTR(strOutput);
	d3dxFontSprite->Begin(D3DX10_SPRITE_SAVE_STATE);
	RECT rect = {x, y, 500, 800};
	D3DXCOLOR textColor;
	textColor.r = 0.2f;
	textColor.g = 0.2f;
	textColor.b = 0.2f;
	textColor.a = 1.0f;
	d3dxFont->DrawText(d3dxFontSprite, text, -1, &rect, DT_LEFT, textColor);
	d3dxFontSprite->End();
}

void Render()
{
	UpdateMatrixs();

	float ClearColor[4] = { 0.6f, 0.6f, 0.6f, 1.0f }; 
    d3dDevice->ClearRenderTargetView( backbufferView, ClearColor );
	d3dDevice->ClearDepthStencilView( depthView, D3D10_CLEAR_DEPTH, 1.0f, 0 );

	if(renderSil){
		//SIL
		if(mesh1){
			d3dDevice->IASetInputLayout(silLayout);
			currentMesh1->renderAdy();
		}else{
			d3dDevice->IASetInputLayout(silLayout2);
			currentMesh2->renderAdy();
		}

		D3DXVECTOR4 minPoint;
		D3DXVECTOR4 maxPoint;
		currentMesh1->getaabbMin(minPoint.x,minPoint.y,minPoint.z);
		currentMesh1->getaabbMax(maxPoint.x,maxPoint.y,maxPoint.z);
		D3DXVECTOR4 aabbCenter = (minPoint + maxPoint) / 2.0f;
		D3DXVECTOR4 aabbCenterTransf;
		D3DXMATRIX worldViewProj = worldMatrix*viewMatrix*projMatrix;
		D3DXVec4Transform(&aabbCenterTransf,&aabbCenter,&worldViewProj);
		aabbCenterTransf.x = aabbCenterTransf.x / aabbCenterTransf.w;
		aabbCenterTransf.y = aabbCenterTransf.y / aabbCenterTransf.w;
		aabbCenterTransf.z = aabbCenterTransf.z / aabbCenterTransf.w;

		worldViewSilParam->SetMatrix((float*)&(worldMatrix*viewMatrix));
		projSilParam->SetMatrix((float*)&projMatrix);
		aabbSilParam->SetFloatVector((float*)&aabbCenterTransf);
		edgeSizeSilParam->SetFloat(edgeSilSize);
		lengthSilParam->SetFloat(lengthFactor);
		scaleSilParam->SetFloat(texRepeate);
		if(firstTexture)
			textureSilParam->SetResource(silTextView);
		else
			textureSilParam->SetResource(silTextView2);

		silTechnique->GetPassByIndex(0)->Apply(0);

		if(mesh1)
			d3dDevice->DrawIndexed(currentMesh1->getNumTriangles()*6,0,0);
		else
			d3dDevice->DrawIndexed(currentMesh2->getNumTriangles()*6,0,0);

	}else{
		//PENCIL
		if(mesh1){
			d3dDevice->IASetInputLayout(pencilLayout);
			currentMesh1->render();
		}else{
			d3dDevice->IASetInputLayout(pencilLayout2);
			currentMesh2->render();
		}

		float aux = 1.0f/sqrt(3.0f);
		D3DXVECTOR4 lightDir;
		lightDir.x = 1.0f*aux;
		lightDir.y = 1.0f*aux;
		lightDir.z = -1.0f*aux;
		lightDir.w = 0.0f;

		worldViewProjPencilParam->SetMatrix((float*)&(worldMatrix*viewMatrix*projMatrix));
		worldPencilParam->SetMatrix((float*)&worldMatrix);
		lightDirPencilParam->SetFloatVector((float*)&lightDir);
		widthPencilParam->SetFloat((float)WIN_WIDTH);
		heightPencilParam->SetFloat((float)WIN_HEIGHT);
		textureRepeatParam->SetFloat(texRepeate);
		OParam->SetFloat(OParamFloat);
		SParam->SetFloat(SParamFloat);
		if(firstTexture)
			texturePencilParam->SetResource(pencilTextView);
		else
			texturePencilParam->SetResource(pencilTextView2);

		pencilTechnique->GetPassByIndex(0)->Apply(0);

		if(mesh1)
			d3dDevice->DrawIndexed(currentMesh1->getNumTriangles()*3,0,0);
		else
			d3dDevice->DrawIndexed(currentMesh2->getNumTriangles()*3,0,0);
	}

	if(helpPannel){
		d3dDevice->PSSetShader(NULL);
		d3dDevice->VSSetShader(NULL);
		d3dDevice->GSSetShader(NULL);

		DrawTextString(10,10,L"ESC - Exit");
		DrawTextString(10,30,L"B - About");
		DrawTextString(10,50,L"H - Toggle Help");
		DrawTextString(10,70,L"Left Button - Rotate");
		DrawTextString(10,90,L"M - Change model");
		DrawTextString(10,110,L"T - Change texture");
		DrawTextString(10,130,L"R - Render Mode (Silhouettes/Pencil)");
		DrawTextString(10,150,L"A/S - Increase/Decrease texture repeat");
		DrawTextString(10,180,L"------Silhouettes------");
		DrawTextString(10,210,L"Q/W - Increase/Decrease edge size");
		DrawTextString(10,230,L"Z/X - Increase/Decrease length factor");
		DrawTextString(10,260,L"--------Pencil---------");
		DrawTextString(10,290,L"D/F - Increase/Decrease O param");
		DrawTextString(10,310,L"C/V - Increase/Decrease S param");
	}else if(about){
		d3dDevice->PSSetShader(NULL);
		d3dDevice->VSSetShader(NULL);
		d3dDevice->GSSetShader(NULL);

		DrawTextString(10,10,L"GPUSilhouetteEdges");
		DrawTextString(10,35,L"Demo program for the paper");
		DrawTextString(10,60,L"\"NPR effects using the Geometry Shader\"");
		DrawTextString(10,85,L"by Pedro Hermosilla and Pere-Pau Vázquez,");
		DrawTextString(10,110,L"GPU Pro 2009, (c) Pedro Hermosilla");
	}

    dxSwapChain->Present(0,0);
}

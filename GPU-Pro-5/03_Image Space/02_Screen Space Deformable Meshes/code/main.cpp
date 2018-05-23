//Copyright(c) 2009 - 2011, yakiimo02
//	All rights reserved.
//
//Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met :
//
//*Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
//
//	* Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and / or other materials provided with the distribution.
//
//	* Neither the name of Yakiimo3D nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
//
//	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// Original per-pixel linked list implementation code from Yakiimo02 was altered by Joao Raza and Gustavo Nunes for GPU Pro 5 'Screen Space Deformable Meshes via CSG with Per-pixel Linked Lists'

#include <fstream>
#include <iostream>
#include <string>
#include <windows.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <directxmath.h>
#include "resource.h"
#include <atlbase.h>
#include <memory>
#include "Camera.h"
#include <vector>

using namespace DirectX;
using namespace std;

#define SAFE_RELEASE(p)      { if (p) { (p)->Release(); (p)=NULL; } }
#define HR_RETURN(x)			 { hr = (x); if( FAILED(hr) ) { return hr; } }

//--------------------------------------------------------------------------------------
// Structures
//--------------------------------------------------------------------------------------
struct SimpleVertex
{
	XMFLOAT3 Pos;
};

struct SCREEN_VERTEX
{
	XMFLOAT4 pos;
};


struct CSGVertex
{
	XMFLOAT4 pos;
	XMFLOAT4 color;
	XMFLOAT3 normal;
	XMFLOAT2 texcoord;		// Unusued in algorithm. Left here for future reference. 
};


struct FragmentData //PixelNodeData
{
	UINT color;
	float    depth;
	UINT frontFace;
};

struct FragmentLink //PixelNode
{
	FragmentData pixelNodeData;
	UINT pNext;
};

struct PS_CB_RenderTargetData 
{
	UINT renderTargetWidth;
	UINT renderTargetHeight;
	UINT padding[2];
};

struct PS_CB_ShaderOperation
{
	unsigned int ShaderSubtractOperation;
};

struct PS_CB_RenderInformation
{
	XMFLOAT3 eyePos;
	bool     invert_normal;
};

struct Camera_CB
{
	XMMATRIX World;
	XMMATRIX ViewProj;
};


enum MESH_RENDER_MODE
{
	MRM_MESH_RENDER_NORMAL = 0,
	MRM_MESH_RENDER_SUBTRACT = 2
};

enum SHADER_SUBTRACT_OPERATION
{
	SSO_MESH_PLUS_BACK_FACING_PIXEL = 0,
	SSO_MESH_PLUS_FRONT_FACING_PIXEL = 1,
	SSO_MESH_MINUS_BACK_FACING_PIXEL = 2,
	SSO_MESH_MINUS_FRONT_FACING_PIXEL = 3
};


//--------------------------------------------------------------------------------------
// Global Variables
//--------------------------------------------------------------------------------------
UINT								g_RenderTargetWidth = 800;
UINT								g_RenderTargetHeight = 600;
HINSTANCE							g_hInst = NULL;
HWND								g_hWnd = NULL;
D3D_DRIVER_TYPE						g_driverType = D3D_DRIVER_TYPE_NULL;
D3D_FEATURE_LEVEL					g_featureLevel = D3D_FEATURE_LEVEL_11_0;
ID3D11Device*						g_pd3dDevice = NULL;
ID3D11DeviceContext*				g_pImmediateContext = NULL;
IDXGISwapChain*						g_pSwapChain = NULL;
ID3D11RenderTargetView*				g_pRenderTargetView = NULL;
ID3D11VertexShader*					g_pVertexShader = NULL;
ID3D11InputLayout*					g_pVertexLayout = NULL;
ID3D11Buffer*						g_pVertexBuffer = NULL;
Camera*								g_Camera = NULL;
XMFLOAT3							g_MeshPosition = XMFLOAT3{ 0.248f, 1.107f, 0.000f } ;
bool								g_isDeformationOn=0;
ID3D11Buffer*						g_pFragmentLink = NULL; 
ID3D11ShaderResourceView*			g_pFragmentLinkSRV = NULL;
ID3D11UnorderedAccessView*			g_pFragmentLinkUAV = NULL;
ID3D11Buffer*						g_pStartOffsetBuffer = NULL; 
ID3D11ShaderResourceView*			g_pStartOffsetSRV = NULL;
ID3D11UnorderedAccessView*			g_pStartOffsetUAV = NULL;
ID3D11DepthStencilState*			g_pDepthStencilState = NULL;
ID3D11RasterizerState*				g_pRasterizerState = NULL;
ID3D11Buffer*						g_ConstantBufferRenderTargetData = NULL;
ID3D11Texture2D*					g_pRenderTargetTexture = NULL;
ID3D11RenderTargetView* 			g_pRenderTargetView_RenderToTexture = NULL;
ID3D11PixelShader*					g_pPixelShaderStoreFragmentsPass = NULL;
ID3D11InputLayout*					g_pQuadLayout = NULL;
ID3D11Buffer*						g_ScreenQuadVertexBuffer = NULL;
ID3D11VertexShader*					g_ScreenQuadVertexShader = NULL;
ID3D11PixelShader*					g_pPSSortFragmentsAndRender = NULL;
ID3D11VertexShader*					g_pSceneVertexShader = NULL;
ID3D11InputLayout*					g_pSceneVertexLayout = NULL;
ID3D11Buffer*						g_pCameraConstantBuffer = NULL;
ID3D11Buffer*						g_pShaderOperationConstantBuffer = NULL;
ID3D11Buffer*						g_pRenderInformationConstantBuffer = NULL;
ID3D11RasterizerState*				g_pFrontFacingRasterizerState = NULL;
ID3D11RasterizerState*				g_pBackFacingRasterizerState = NULL;
ID3D11Buffer*						g_pMPlusVertexBuffer = NULL;
ID3D11Buffer*						g_pMPlusIndexBuffer = NULL;
ID3D11Buffer*						g_pMMinusVertexBuffer = NULL;
ID3D11Buffer*						g_pMMinusIndexBuffer = NULL;
ID3D11ShaderResourceView*			m_pCubeShaderResourceView = NULL;
ID3D11ShaderResourceView*			g_pShaderResourceView = NULL;
ID3D11PixelShader*					g_pPixelShader = NULL;
ID3D11SamplerState*					g_pPSSampler = NULL;
ID3D11DepthStencilState*			g_pDSState = NULL;

//--------------------------------------------------------------------------------------
// Forward declarations
//--------------------------------------------------------------------------------------
HRESULT InitWindow(HINSTANCE hInstance, int nCmdShow);
HRESULT InitDevice();
void CleanupDevice();
LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);
void Render();
void Update(float deltaTime);
HRESULT loadFromObj(ID3D11Device* pDevice, ID3D11Buffer*& vertexBuffer, ID3D11Buffer*& indexBuffer, char* filePath, XMFLOAT4 color);
void OnRender(bool finalPass);
void DrawObject(ID3D11Buffer* vertexBuffer, ID3D11Buffer* indexBuffer, int numIndices, XMMATRIX* pmWorld, const MESH_RENDER_MODE draw_render_mode, bool IsFrameBufferRenderPass, const bool invert_normal);


//--------------------------------------------------------------------------------------
// Entry point to the program. Initializes everything and goes into a message processing 
// loop. Idle time is used to render the scene.
//--------------------------------------------------------------------------------------
int WINAPI wWinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPWSTR lpCmdLine, int nCmdShow)
{
	UNREFERENCED_PARAMETER(hPrevInstance);
	UNREFERENCED_PARAMETER(lpCmdLine);

	if (FAILED(InitWindow(hInstance, nCmdShow)))
		return 0;

	RECT windowRect;
	GetClientRect(g_hWnd, &windowRect);

	//Instantiate Camera aligned
	void* p = _aligned_malloc(sizeof(Camera), 16);
	g_Camera = (Camera*) p;
	new (g_Camera) Camera((float) windowRect.right - (float) windowRect.left, (float) windowRect.bottom - (float) windowRect.top);	
	g_Camera->SetValues(XMVectorSet(-73.47f, 73.08f, -85.83f, 982.77f), 
						XMVectorSet(-72.85f, 72.73f, -85.12f, 983.77f),
						XMVectorSet( 00.22f, 0.93f,   0.25f,  1.00f),
						0.33f, 0.73f );

	if (FAILED(InitDevice()))
	{
		CleanupDevice();
		return 0;
	}

	//Timer Variables
	__int64 countsPerSec;
	QueryPerformanceFrequency((LARGE_INTEGER*) &countsPerSec);
	double secondsPerCount = (1.0 / (double) countsPerSec);
	__int64 previousTime = 0;

	// Main message loop
	MSG msg = { 0 };
	while (WM_QUIT != msg.message)
	{
		__int64 currentTime = 0;
		QueryPerformanceCounter((LARGE_INTEGER*) &currentTime);

		if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE))
		{
			TranslateMessage(&msg);
			DispatchMessage(&msg);
		}
		else
		{
			const double delta = max( (currentTime - previousTime)*secondsPerCount, 0);			
			Update((float) delta);
			Render();
		}

		previousTime = currentTime;
	}

	CleanupDevice();

	g_Camera->~Camera();
	_aligned_free(g_Camera);
	return (int) msg.wParam;
}


//--------------------------------------------------------------------------------------
// Register class and create window
//--------------------------------------------------------------------------------------
HRESULT InitWindow(HINSTANCE hInstance, int nCmdShow)
{
	(void) nCmdShow;
	// Register class
	WNDCLASSEX wcex;
	wcex.cbSize = sizeof(WNDCLASSEX);
	wcex.style = CS_HREDRAW | CS_VREDRAW;
	wcex.lpfnWndProc = WndProc;
	wcex.cbClsExtra = 0;
	wcex.cbWndExtra = 0;
	wcex.hInstance = hInstance;
	wcex.hIcon = LoadIcon(hInstance, (LPCTSTR) IDI_TUTORIAL1);
	wcex.hCursor = LoadCursor(NULL, IDC_ARROW);
	wcex.hbrBackground = (HBRUSH) (COLOR_WINDOW + 1);
	wcex.lpszMenuName = NULL;
	wcex.lpszClassName = L"WindowClass";
	wcex.hIconSm = LoadIcon(wcex.hInstance, (LPCTSTR) IDI_TUTORIAL1);
	if (!RegisterClassEx(&wcex))
		return E_FAIL;

	// Create window
	g_hInst = hInstance;
	RECT rc = { 0, 0, g_RenderTargetWidth, g_RenderTargetHeight };
	AdjustWindowRect(&rc, WS_OVERLAPPEDWINDOW, FALSE);
	g_hWnd = CreateWindow(L"WindowClass", L"Deformable Mesh",
		WS_OVERLAPPEDWINDOW,
		CW_USEDEFAULT, CW_USEDEFAULT, rc.right - rc.left, rc.bottom - rc.top, NULL, NULL, hInstance,
		NULL);
	if (!g_hWnd)
		return E_FAIL;

	ShowWindow(g_hWnd, SW_SHOWNOACTIVATE);

	return S_OK;
}


//--------------------------------------------------------------------------------------
// Helper for compiling shaders with D3DCompile
//
// With VS 11, we could load up prebuilt .cso files instead...
//--------------------------------------------------------------------------------------
HRESULT CompileShaderFromFile(WCHAR* szFileName, LPCSTR szEntryPoint, LPCSTR szShaderModel, ID3DBlob** ppBlobOut)
{
	HRESULT hr = S_OK;

	DWORD dwShaderFlags = D3DCOMPILE_ENABLE_STRICTNESS | D3D10_SHADER_SKIP_OPTIMIZATION;
#if defined( DEBUG ) || defined( _DEBUG )
	// Set the D3DCOMPILE_DEBUG flag to embed debug information in the shaders.
	// Setting this flag improves the shader debugging experience, but still allows 
	// the shaders to be optimized and to run exactly the way they will run in 
	// the release configuration of this program.
	dwShaderFlags |= D3DCOMPILE_DEBUG;
#endif

	ID3DBlob* pErrorBlob;
	hr = D3DCompileFromFile(szFileName, NULL, NULL, szEntryPoint, szShaderModel,
		dwShaderFlags, 0, ppBlobOut, &pErrorBlob);
	if (FAILED(hr))
	{
		if (pErrorBlob != NULL)
		{
			OutputDebugStringA((char*) pErrorBlob->GetBufferPointer());
		}
		if (pErrorBlob) pErrorBlob->Release();
		return hr;
	}
	if (pErrorBlob) pErrorBlob->Release();

	return S_OK;
}

HRESULT CreateViewPort()
{
	RECT rc;
	GetClientRect(g_hWnd, &rc);
	UINT width = rc.right - rc.left;
	UINT height = rc.bottom - rc.top;

	D3D11_VIEWPORT vp;
	vp.Width = (FLOAT) width;
	vp.Height = (FLOAT) height;
	vp.MinDepth = 0.0f;
	vp.MaxDepth = 1.0f;
	vp.TopLeftX = 0;
	vp.TopLeftY = 0;
	g_pImmediateContext->RSSetViewports(1, &vp);
	return S_OK ;
}

HRESULT CreateDepthStencilStateWithWritePermissionOff()
{
	HRESULT hr ;
	D3D11_DEPTH_STENCIL_DESC depthStencilDesc;
	ZeroMemory(&depthStencilDesc, sizeof(D3D11_DEPTH_STENCIL_DESC));
	depthStencilDesc.DepthFunc = D3D11_COMPARISON_LESS;
	depthStencilDesc.StencilEnable = FALSE;
	depthStencilDesc.DepthEnable = TRUE;
	depthStencilDesc.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ZERO;
	hr = g_pd3dDevice->CreateDepthStencilState(&depthStencilDesc, &g_pDepthStencilState);
	if (FAILED(hr)) return hr;

	D3D11_BUFFER_DESC cbDesc;
	cbDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
	cbDesc.MiscFlags = 0;
	cbDesc.Usage = D3D11_USAGE_DYNAMIC;
	cbDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
	cbDesc.ByteWidth = sizeof(PS_CB_RenderTargetData);
	hr = g_pd3dDevice->CreateBuffer(&cbDesc, NULL, &g_ConstantBufferRenderTargetData);
	if (FAILED(hr)) return hr;

	return S_OK ;
}

HRESULT CreateRasterizerState()
{
	D3D11_RASTERIZER_DESC rasterizerDesc;
	ZeroMemory(&rasterizerDesc, sizeof(D3D11_RASTERIZER_DESC));
	rasterizerDesc.DepthClipEnable = TRUE;
	rasterizerDesc.ScissorEnable = FALSE;
	rasterizerDesc.MultisampleEnable = TRUE;
	rasterizerDesc.AntialiasedLineEnable = FALSE;
	rasterizerDesc.DepthBiasClamp = 0.0f;
	rasterizerDesc.SlopeScaledDepthBias = 0.0f;
	rasterizerDesc.FillMode = D3D11_FILL_SOLID;
	rasterizerDesc.CullMode = D3D11_CULL_NONE;
	rasterizerDesc.FrontCounterClockwise = FALSE;
	rasterizerDesc.DepthBias = 0;

	HRESULT hr ;
	hr = g_pd3dDevice->CreateRasterizerState(&rasterizerDesc, &g_pRasterizerState);
	if (FAILED(hr)) return hr;

	return S_OK ;
}

HRESULT CreateRenderToTextureResources()
{
	HRESULT hr ;
	D3D11_TEXTURE2D_DESC tex2dDesc;
	ZeroMemory(&tex2dDesc, sizeof(D3D11_TEXTURE2D_DESC));
	tex2dDesc.Width = g_RenderTargetWidth;
	tex2dDesc.Height = g_RenderTargetHeight;
	tex2dDesc.MipLevels = 1;
	tex2dDesc.ArraySize = 1;
	tex2dDesc.Format = DXGI_FORMAT_R32_FLOAT;
	tex2dDesc.Usage = D3D11_USAGE_DEFAULT;
	tex2dDesc.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
	tex2dDesc.CPUAccessFlags = 0;
	tex2dDesc.MiscFlags = 0;
	DXGI_SAMPLE_DESC sampleDesc;
	ZeroMemory(&sampleDesc, sizeof(DXGI_SAMPLE_DESC));
	sampleDesc.Count = 1;
	sampleDesc.Quality = 0;
	tex2dDesc.SampleDesc = sampleDesc;

	hr = g_pd3dDevice->CreateTexture2D(&tex2dDesc, NULL, &g_pRenderTargetTexture);
	if (FAILED(hr)) return hr;

	D3D11_RENDER_TARGET_VIEW_DESC rtDesc;
	ZeroMemory(&rtDesc, sizeof(D3D11_RENDER_TARGET_VIEW_DESC));
	rtDesc.Format = tex2dDesc.Format;
	rtDesc.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2D;
	rtDesc.Texture2D.MipSlice = 0;

	hr = g_pd3dDevice->CreateRenderTargetView(g_pRenderTargetTexture, &rtDesc, &g_pRenderTargetView_RenderToTexture);
	if (FAILED(hr)) return hr;

	D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc;
	ZeroMemory(&srvDesc, sizeof(D3D11_SHADER_RESOURCE_VIEW_DESC));
	srvDesc.Format = DXGI_FORMAT_R32_FLOAT;
	D3D11_TEX2D_SRV tex2d;
	ZeroMemory(&tex2d, sizeof(D3D11_TEX2D_SRV));
	tex2d.MipLevels = (UINT) - 1;
	tex2d.MostDetailedMip = 0;
	srvDesc.Texture2D = tex2d;
	srvDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;

	hr = g_pd3dDevice->CreateShaderResourceView(g_pRenderTargetTexture, &srvDesc, &g_pShaderResourceView);
	if (FAILED(hr)) return hr;

	return S_OK ;
}

HRESULT CreateD3DDevice()
{
	HRESULT hr = S_OK;

	g_isDeformationOn = false;

	RECT rc;
	GetClientRect(g_hWnd, &rc);
	UINT width = rc.right - rc.left;
	UINT height = rc.bottom - rc.top;

	UINT createDeviceFlags = 0;
#ifdef _DEBUG
	createDeviceFlags |= D3D11_CREATE_DEVICE_DEBUG;
#endif

	D3D_DRIVER_TYPE driverTypes [] =
	{
		D3D_DRIVER_TYPE_HARDWARE,
		D3D_DRIVER_TYPE_WARP,
		D3D_DRIVER_TYPE_REFERENCE,
	};
	UINT numDriverTypes = ARRAYSIZE(driverTypes);

	D3D_FEATURE_LEVEL featureLevels [] =
	{
		D3D_FEATURE_LEVEL_11_0,
	};
	UINT numFeatureLevels = ARRAYSIZE(featureLevels);

	DXGI_SWAP_CHAIN_DESC sd;
	ZeroMemory(&sd, sizeof(sd));
	sd.BufferCount = 1;
	sd.BufferDesc.Width = width;
	sd.BufferDesc.Height = height;
	sd.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
	sd.BufferDesc.RefreshRate.Numerator = 60;
	sd.BufferDesc.RefreshRate.Denominator = 1;
	sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
	sd.OutputWindow = g_hWnd;
	sd.SampleDesc.Count = 1;
	sd.SampleDesc.Quality = 0;
	sd.Windowed = TRUE;

	for (UINT driverTypeIndex = 0; driverTypeIndex < numDriverTypes; driverTypeIndex++)
	{
		g_driverType = driverTypes[driverTypeIndex];
		hr = D3D11CreateDeviceAndSwapChain(NULL, g_driverType, NULL, createDeviceFlags, featureLevels, numFeatureLevels,
			D3D11_SDK_VERSION, &sd, &g_pSwapChain, &g_pd3dDevice, &g_featureLevel, &g_pImmediateContext);
		if (SUCCEEDED(hr))
			break;
	}
	if (FAILED(hr)) return hr;

	return S_OK;
}

HRESULT CreateRenderTargetView()
{
	HRESULT hr ;
	CComPtr<ID3D11Texture2D> pBackBuffer;
	hr = g_pSwapChain->GetBuffer(0, __uuidof(ID3D11Texture2D), (LPVOID*) &pBackBuffer);
	if (FAILED(hr))
		return hr;

	hr = g_pd3dDevice->CreateRenderTargetView(pBackBuffer, NULL, &g_pRenderTargetView);
	if (FAILED(hr)) return hr;

	g_pImmediateContext->OMSetRenderTargets(1, &g_pRenderTargetView, NULL);
	return S_OK ;
}

HRESULT CreateFragmentLinkBuffer( const int NUM_ELEMENTS)
{
	HRESULT hr ;
	D3D11_BUFFER_DESC descBuf;
	memset(&descBuf, 0, sizeof(descBuf));
	descBuf.StructureByteStride = sizeof(FragmentLink);
	descBuf.ByteWidth = NUM_ELEMENTS * descBuf.StructureByteStride;
	descBuf.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
	descBuf.BindFlags = D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;
	hr = g_pd3dDevice->CreateBuffer(&descBuf, NULL, &g_pFragmentLink);
	if (FAILED(hr))
		return hr;

	return S_OK ;
}

HRESULT CreateUAV( const int NUM_ELEMENTS )
{
	HRESULT hr ;
	D3D11_UNORDERED_ACCESS_VIEW_DESC descUAV;
	memset(&descUAV, 0, sizeof(descUAV));
	descUAV.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
	descUAV.Buffer.FirstElement = 0;
	descUAV.Format = DXGI_FORMAT_UNKNOWN;
	descUAV.Buffer.NumElements = NUM_ELEMENTS;
	descUAV.Buffer.Flags = D3D11_BUFFER_UAV_FLAG_COUNTER;
	hr = g_pd3dDevice->CreateUnorderedAccessView(g_pFragmentLink, &descUAV, &g_pFragmentLinkUAV);
	if (FAILED(hr)) return hr;

	return S_OK ;
}

HRESULT CreateShaderResourceView( const int NUM_ELEMENTS)
{
	HRESULT hr ;
	D3D11_SHADER_RESOURCE_VIEW_DESC descSRV;
	descSRV.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
	descSRV.Buffer.FirstElement = 0;
	descSRV.Format = DXGI_FORMAT_UNKNOWN;
	descSRV.Buffer.NumElements = NUM_ELEMENTS;
	hr = g_pd3dDevice->CreateShaderResourceView(g_pFragmentLink, &descSRV, &g_pFragmentLinkSRV);
	if (FAILED(hr)) return hr;

	return S_OK ;
}

HRESULT CreateLinkedListUAV( const int NUM_ELEMENTS2 )
{
	D3D11_BUFFER_DESC descBuf2; 
	HRESULT hr ;
	memset(&descBuf2, 0, sizeof(descBuf2));
	descBuf2.StructureByteStride = sizeof(unsigned int) ;
	descBuf2.ByteWidth = NUM_ELEMENTS2 * descBuf2.StructureByteStride;
	descBuf2.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS;
	descBuf2.BindFlags = D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;
	hr = g_pd3dDevice->CreateBuffer(&descBuf2, NULL, &g_pStartOffsetBuffer);
	if (FAILED(hr)) return hr;

	return S_OK ;
}

HRESULT CreateOffsetUAV( const int NUM_ELEMENTS2 )
{
	HRESULT hr ;
	D3D11_UNORDERED_ACCESS_VIEW_DESC descUAV2; 
	memset(&descUAV2, 0, sizeof(descUAV2));
	descUAV2.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
	descUAV2.Buffer.FirstElement = 0;
	descUAV2.Format = DXGI_FORMAT_R32_TYPELESS;
	descUAV2.Buffer.NumElements = NUM_ELEMENTS2;
	descUAV2.Buffer.Flags = D3D11_BUFFER_UAV_FLAG_RAW;
	hr = g_pd3dDevice->CreateUnorderedAccessView(g_pStartOffsetBuffer, &descUAV2, &g_pStartOffsetUAV);
	if (FAILED(hr)) return hr;

	return S_OK ;
}

HRESULT CreateSRVForUAV( const int NUM_ELEMENTS2 )
{
	HRESULT hr ;
	D3D11_SHADER_RESOURCE_VIEW_DESC descSRV2; 
	descSRV2.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
	descSRV2.Buffer.FirstElement = 0;
	descSRV2.Format = DXGI_FORMAT_R32_UINT;
	descSRV2.Buffer.NumElements = NUM_ELEMENTS2;
	hr = g_pd3dDevice->CreateShaderResourceView(g_pStartOffsetBuffer, &descSRV2, &g_pStartOffsetSRV);
	if (FAILED(hr)) return hr;	

	ID3DBlob* pBlob = NULL;
	hr = CompileShaderFromFile(L"StoreFragments.hlsl", "StoreFragmentsPS", "ps_5_0", &pBlob);
	if (FAILED(hr)) return hr;
	hr = g_pd3dDevice->CreatePixelShader(pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &g_pPixelShaderStoreFragmentsPass);
	if (FAILED(hr)) return hr;
	SAFE_RELEASE(pBlob);

	return S_OK ;
}

HRESULT CreateInputLayout()
{
	HRESULT hr ;
	ID3DBlob* pBlobVS = NULL;
	hr = CompileShaderFromFile(L"Scene.hlsl", "SceneVS", "vs_5_0", &pBlobVS);
	HR_RETURN(g_pd3dDevice->CreateVertexShader(pBlobVS->GetBufferPointer(), pBlobVS->GetBufferSize(), NULL, &g_pSceneVertexShader));

	// Create the input layout
	const D3D11_INPUT_ELEMENT_DESC vertexLayout [] =
	{
		{ "POSITION", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 0,							  D3D11_INPUT_PER_VERTEX_DATA, 0 },
		{ "COLOR",    0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, D3D11_APPEND_ALIGNED_ELEMENT, D3D11_INPUT_PER_VERTEX_DATA, 0 },
		{ "NORMAL",   0, DXGI_FORMAT_R32G32B32_FLOAT,    0, D3D11_APPEND_ALIGNED_ELEMENT, D3D11_INPUT_PER_VERTEX_DATA, 0 },
		{ "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT,       0, D3D11_APPEND_ALIGNED_ELEMENT, D3D11_INPUT_PER_VERTEX_DATA, 0 }		
	};
	HR_RETURN(g_pd3dDevice->CreateInputLayout(vertexLayout, 4, pBlobVS->GetBufferPointer(), pBlobVS->GetBufferSize(), &g_pSceneVertexLayout));
	SAFE_RELEASE(pBlobVS);	
	return hr;	
}

HRESULT CreateFinalPassPixelShader()
{
	HRESULT hr ;
	// Create final pass PS
	ID3DBlob* pBlobPS = NULL;
	HR_RETURN(CompileShaderFromFile(L"FinalPass.hlsl", "PS", "ps_5_0", &pBlobPS));
	HR_RETURN(g_pd3dDevice->CreatePixelShader(pBlobPS->GetBufferPointer(), pBlobPS->GetBufferSize(), NULL, &g_pPixelShader));
	SAFE_RELEASE(pBlobPS);

	//Sampler for final pass PS
	D3D11_SAMPLER_DESC samplerDesc;
	samplerDesc.Filter = D3D11_FILTER_MIN_POINT_MAG_LINEAR_MIP_POINT;
	samplerDesc.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
	samplerDesc.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
	samplerDesc.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
	samplerDesc.MaxAnisotropy = 0;
	samplerDesc.MipLODBias = 0;
	HR_RETURN(g_pd3dDevice->CreateSamplerState(&samplerDesc, &g_pPSSampler));

	//DSState for final pass
	D3D11_DEPTH_STENCIL_DESC depDesc;
	depDesc.DepthEnable = TRUE;
	depDesc.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ALL;
	depDesc.DepthFunc = D3D11_COMPARISON_LESS;
	depDesc.StencilEnable = FALSE;

	HR_RETURN(g_pd3dDevice->CreateDepthStencilState(&depDesc, &g_pDSState));

	return S_OK ;
}

HRESULT SetupConstantBuffer()
{
	HRESULT hr ;
	D3D11_BUFFER_DESC Desc;
	Desc.Usage = D3D11_USAGE_DYNAMIC;
	Desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
	Desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
	Desc.MiscFlags = 0;
	Desc.ByteWidth = sizeof(Camera_CB);
	HR_RETURN(g_pd3dDevice->CreateBuffer(&Desc, NULL, &g_pCameraConstantBuffer));

	Desc.Usage = D3D11_USAGE_DYNAMIC;
	Desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
	Desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
	Desc.MiscFlags = 0;
	Desc.ByteWidth = 16;
	HR_RETURN(g_pd3dDevice->CreateBuffer(&Desc, NULL, &g_pShaderOperationConstantBuffer));

	HR_RETURN(g_pd3dDevice->CreateBuffer(&Desc, NULL, &g_pRenderInformationConstantBuffer));

	return S_OK ;
}

HRESULT FillVertexAndIndexBuffer()
{
	HRESULT hr ;
	CSGVertex pVertex[8];
	pVertex[0].pos = XMFLOAT4(0, 0, 0, 1.0f);
	pVertex[1].pos = XMFLOAT4(5.0f, 0, 0, 1.0f);
	pVertex[2].pos = XMFLOAT4(5.0f, 0, 5.0f, 1.0f);
	pVertex[3].pos = XMFLOAT4(0, 0, 5.0f, 1.0f);

	pVertex[4].pos = XMFLOAT4(0, 5.0f, 0, 1.0f);
	pVertex[5].pos = XMFLOAT4(5.0f, 5.0f, 0, 1.0f);
	pVertex[6].pos = XMFLOAT4(5.0f, 5.0f, 5.0f, 1.0f);
	pVertex[7].pos = XMFLOAT4(0, 5.0f, 5.0f, 1.0f);

	pVertex[0].color = XMFLOAT4(1.0f, 0.0f, 0.0f, 0.5f);
	pVertex[1].color = XMFLOAT4(1.0f, 0.0f, 0.0f, 0.5f);
	pVertex[2].color = XMFLOAT4(1.0f, 0.0f, 0.0f, 0.5f);
	pVertex[3].color = XMFLOAT4(1.0f, 0.0f, 0.0f, 0.5f);

	pVertex[4].color = XMFLOAT4(1.0f, 1.0f, 0.0f, 0.5f);
	pVertex[5].color = XMFLOAT4(1.0f, 1.0f, 0.0f, 0.5f);
	pVertex[6].color = XMFLOAT4(1.0f, 0.0f, 0.0f, 0.5f);
	pVertex[7].color = XMFLOAT4(1.0f, 0.0f, 0.0f, 0.5f);

	const float nv = 0.5745;
	pVertex[0].normal = XMFLOAT3(-nv, -nv, -nv);
	pVertex[1].normal = XMFLOAT3( nv,  -nv, -nv);
	pVertex[2].normal = XMFLOAT3( nv,  -nv, nv);
	pVertex[3].normal = XMFLOAT3(-nv, -nv, nv);

	pVertex[4].normal = XMFLOAT3(-nv, nv, -nv);
	pVertex[5].normal = XMFLOAT3( nv, nv, -nv);
	pVertex[6].normal = XMFLOAT3( nv, nv, nv);
	pVertex[7].normal = XMFLOAT3(-nv, nv, nv);

	//fill index buffer
	int indexBuffer[36];

	indexBuffer[0] = 4;
	indexBuffer[1] = 5;
	indexBuffer[2] = 0;
	indexBuffer[3] = 0;
	indexBuffer[4] = 5;
	indexBuffer[5] = 1;

	indexBuffer[6] = 5;
	indexBuffer[7] = 6;
	indexBuffer[8] = 2;
	indexBuffer[9] = 1;
	indexBuffer[10] = 5;
	indexBuffer[11] = 2;

	indexBuffer[12] = 6;
	indexBuffer[13] = 3;
	indexBuffer[14] = 2;
	indexBuffer[15] = 6;
	indexBuffer[16] = 7;
	indexBuffer[17] = 3;

	indexBuffer[18] = 7;
	indexBuffer[19] = 4;
	indexBuffer[20] = 3;
	indexBuffer[21] = 4;
	indexBuffer[22] = 0;
	indexBuffer[23] = 3;

	indexBuffer[24] = 4;
	indexBuffer[25] = 6;
	indexBuffer[26] = 5;
	indexBuffer[27] = 4;
	indexBuffer[28] = 7;
	indexBuffer[29] = 6;

	indexBuffer[30] = 0;
	indexBuffer[31] = 2;
	indexBuffer[32] = 3;
	indexBuffer[33] = 0;
	indexBuffer[34] = 1;
	indexBuffer[35] = 2;

	D3D11_BUFFER_DESC vbdesc;
	vbdesc.ByteWidth = 16 * 32; // 8 * sizeof(CSGVertex);
	vbdesc.Usage = D3D11_USAGE_IMMUTABLE;
	vbdesc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
	vbdesc.CPUAccessFlags = 0;
	vbdesc.MiscFlags = 0;

	D3D11_SUBRESOURCE_DATA InitData;
	InitData.pSysMem = pVertex;
	HR_RETURN(g_pd3dDevice->CreateBuffer(&vbdesc, &InitData, &g_pMPlusVertexBuffer));

	D3D11_BUFFER_DESC ibdesc;
	ibdesc.ByteWidth = 36 * sizeof(int) ;
	ibdesc.Usage = D3D11_USAGE_IMMUTABLE;
	ibdesc.BindFlags = D3D11_BIND_INDEX_BUFFER;
	ibdesc.CPUAccessFlags = 0;
	ibdesc.MiscFlags = 0;

	InitData.pSysMem = indexBuffer;
	HR_RETURN(g_pd3dDevice->CreateBuffer(&ibdesc, &InitData, &g_pMPlusIndexBuffer));

	return S_OK ;
}

HRESULT CreateFrontAndBackFaceRasterizerState()
{
	D3D11_RASTERIZER_DESC drd = {
		D3D11_FILL_SOLID, //D3D11_FILL_MODE FillMode;
		D3D11_CULL_BACK,//D3D11_CULL_MODE CullMode;
		FALSE, //BOOL FrontCounterClockwise;
		0, //INT DepthBias;
		0.0f,//FLOAT DepthBiasClamp;
		0.0f,//FLOAT SlopeScaledDepthBias;
		TRUE,//BOOL DepthClipEnable;
		FALSE,//BOOL ScissorEnable;
		TRUE,//BOOL MultisampleEnable;
		FALSE//BOOL AntialiasedLineEnable;        
	};

	g_pd3dDevice->CreateRasterizerState(&drd, &g_pFrontFacingRasterizerState);
	drd.CullMode = D3D11_CULL_FRONT;
	return g_pd3dDevice->CreateRasterizerState(&drd, &g_pBackFacingRasterizerState);
}

HRESULT CompileScreenQuadVertexShader()
{
	ID3DBlob*pBlob = NULL;
	HRESULT hr ;

	// vertex shader
	HR_RETURN(CompileShaderFromFile(L"SortFragmentsAndRender.hlsl", "QuadVS", "vs_4_0", &pBlob));
	HR_RETURN(g_pd3dDevice->CreateVertexShader(pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &g_ScreenQuadVertexShader));

	const D3D11_INPUT_ELEMENT_DESC quadlayout [] =
	{
		{ "POSITION", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 },
	};

	HR_RETURN(g_pd3dDevice->CreateInputLayout(quadlayout, 1, pBlob->GetBufferPointer(), pBlob->GetBufferSize(), &g_pQuadLayout));
	SAFE_RELEASE(pBlob);

	// pixel shader
	HR_RETURN(CompileShaderFromFile(L"SortFragmentsAndRender.hlsl", "SortFragmentsPS", "ps_5_0", &pBlob));
	HR_RETURN(g_pd3dDevice->CreatePixelShader(pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &g_pPSSortFragmentsAndRender));
	SAFE_RELEASE(pBlob);

	// full screen quad
	SCREEN_VERTEX svQuad[4];
	svQuad[0].pos = XMFLOAT4(-1.0f, 1.0f, 0.5f, 1.0f);
	svQuad[1].pos = XMFLOAT4(1.0f, 1.0f, 0.5f, 1.0f);
	svQuad[2].pos = XMFLOAT4(-1.0f, -1.0f, 0.5f, 1.0f);
	svQuad[3].pos = XMFLOAT4(1.0f, -1.0f, 0.5f, 1.0f);

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
	HR_RETURN(g_pd3dDevice->CreateBuffer(&vbdesc, &InitData, &g_ScreenQuadVertexBuffer));

	return S_OK ;
}


//--------------------------------------------------------------------------------------
// Create Direct3D device and swap chain
//--------------------------------------------------------------------------------------
HRESULT InitDevice()
{
	HRESULT hr ;	
	HR_RETURN( CreateD3DDevice() );
	HR_RETURN( CreateRenderTargetView());
	HR_RETURN( CreateViewPort());
	HR_RETURN( CreateDepthStencilStateWithWritePermissionOff());
	HR_RETURN( CreateRasterizerState());
	HR_RETURN( CreateRenderToTextureResources());
	const int NUM_ELEMENTS = g_RenderTargetWidth * g_RenderTargetHeight * 8; 
	const int NUM_ELEMENTS2 = g_RenderTargetWidth * g_RenderTargetHeight;	
	HR_RETURN( CreateFragmentLinkBuffer(NUM_ELEMENTS));
	HR_RETURN( CreateUAV(NUM_ELEMENTS));
	HR_RETURN( CreateShaderResourceView(NUM_ELEMENTS));
	HR_RETURN( CreateLinkedListUAV(NUM_ELEMENTS2));
	HR_RETURN( CreateOffsetUAV(NUM_ELEMENTS2));
	HR_RETURN( CreateSRVForUAV(NUM_ELEMENTS2));
	HR_RETURN( CreateInputLayout());
	HR_RETURN( CreateFinalPassPixelShader());
	HR_RETURN( SetupConstantBuffer());
	HR_RETURN( FillVertexAndIndexBuffer());
	HR_RETURN( CreateFrontAndBackFaceRasterizerState());
	HR_RETURN( loadFromObj(g_pd3dDevice, g_pMMinusVertexBuffer, g_pMMinusIndexBuffer, "..\\data\\spot.ply", XMFLOAT4(0, 0, 1, 1)));
	HR_RETURN( CompileScreenQuadVertexShader());

	return S_OK;
}


//--------------------------------------------------------------------------------------
// Clean up the objects we've created
//--------------------------------------------------------------------------------------
void CleanupDevice()
{
	if (g_pImmediateContext) g_pImmediateContext->ClearState();
}


//--------------------------------------------------------------------------------------
// Called every time the application receives a message
//--------------------------------------------------------------------------------------
LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
	PAINTSTRUCT ps;
	HDC hdc;

	switch (message)
	{
	case WM_PAINT:
		hdc = BeginPaint(hWnd, &ps);
		EndPaint(hWnd, &ps);
		break;

	case WM_DESTROY:
		PostQuitMessage(0);
		break;
	case WM_MOUSEMOVE:
		g_Camera->RotateCamera(wParam, ((int) LOWORD(lParam)), ((int) HIWORD(lParam)));
		break;

	default:
		return DefWindowProc(hWnd, message, wParam, lParam);
	}

	return 0;
}

void Update(float delta)
{
	g_Camera->Update(delta);

	const float factor = 5.0f * delta;

	if (GetAsyncKeyState('F') & 0x8000)
	{
		g_MeshPosition.x += factor;
	}
	if (GetAsyncKeyState('V') & 0x8000)
	{
		g_MeshPosition.x -= factor;
	}
	if (GetAsyncKeyState('G') & 0x8000)
	{
		g_MeshPosition.z += factor;
	}
	if (GetAsyncKeyState('B') & 0x8000)
	{
		g_MeshPosition.z -= factor;
	}
	if (GetAsyncKeyState('R') & 0x8000)
	{
		g_MeshPosition.y += factor;
	}
	if (GetAsyncKeyState('T') & 0x8000)
	{
		g_MeshPosition.y -= factor;
	}
	if (GetAsyncKeyState('Z') & 0x8000)
	{
		g_isDeformationOn = true;
	}
	if (GetAsyncKeyState('X') & 0x8000)
	{
		g_isDeformationOn = false;
	}
}


//--------------------------------------------------------------------------------------
// Render a frame
//--------------------------------------------------------------------------------------
void Render()
{
	//Store render target and depth stencil view
	ID3D11RenderTargetView* pOldRTV = NULL;
	ID3D11DepthStencilView* pOldDSV = NULL;
	g_pImmediateContext->OMGetRenderTargets(1, &pOldRTV, &pOldDSV);

	// Clear the back buffer 
	float ClearColor[4] = { 0.0f, 0.0f, 0.0f, 1.0f }; // red,green,blue,alpha
	g_pImmediateContext->ClearRenderTargetView(pOldRTV, ClearColor);
	if (pOldDSV != NULL)
		g_pImmediateContext->ClearDepthStencilView(pOldDSV, D3D11_CLEAR_DEPTH, 1.0, 0);

	// Save rasterizer state
	ID3D11RasterizerState* pOldState;
	g_pImmediateContext->RSGetState(&pOldState);
	g_pImmediateContext->RSSetState(g_pRasterizerState);

	ID3D11DepthStencilState* pDepthStencilStateStored = NULL;
	UINT stencilRef;
	g_pImmediateContext->OMGetDepthStencilState(&pDepthStencilStateStored, &stencilRef);

	// Setup the constant buffer
	D3D11_MAPPED_SUBRESOURCE MappedResource;
	g_pImmediateContext->Map(g_ConstantBufferRenderTargetData, 0, D3D11_MAP_WRITE_DISCARD, 0, &MappedResource);
	PS_CB_RenderTargetData* pPS_CB = (PS_CB_RenderTargetData*) MappedResource.pData;
	pPS_CB->renderTargetWidth = g_RenderTargetWidth;
	pPS_CB->renderTargetHeight = g_RenderTargetHeight;
	g_pImmediateContext->Unmap(g_ConstantBufferRenderTargetData, 0);
	g_pImmediateContext->PSSetConstantBuffers(0, 1, &g_ConstantBufferRenderTargetData);

	// Clear the start offset buffer by magic value.
	static const UINT clearValueUINT[1] = { 0xffffffff };
	g_pImmediateContext->ClearUnorderedAccessViewUint(g_pStartOffsetUAV, clearValueUINT);

	// Set UAV
	ID3D11UnorderedAccessView* pUAVs[2];
	pUAVs[0] = g_pFragmentLinkUAV;
	pUAVs[1] = g_pStartOffsetUAV;

	// Initialize UAV counter
	UINT initIndices [] = { 0, 0 };
	g_pImmediateContext->OMSetRenderTargetsAndUnorderedAccessViews(0, NULL, pOldDSV, 0, 2, pUAVs, initIndices);

	g_pImmediateContext->OMSetDepthStencilState(g_pDepthStencilState, 0);

	g_pImmediateContext->PSSetShader(g_pPixelShaderStoreFragmentsPass, NULL, 0);

	OnRender(false);

	// Set render target and depth/stencil views to NULL,
	// we'll need to read the RTV in a shader later
	ID3D11RenderTargetView* pRTViewNULL[1] = { NULL };
	ID3D11ShaderResourceView* pSRViewNULL[1] = { NULL };
	ID3D11DepthStencilView* pDSVNULL = NULL;
	g_pImmediateContext->OMSetRenderTargets(1, pRTViewNULL, pDSVNULL);
	g_pImmediateContext->PSSetShaderResources(0, 1, pSRViewNULL);

	// Sort and render linked-list fragments into the frame buffer.
	g_pImmediateContext->OMSetRenderTargets(1, &g_pRenderTargetView_RenderToTexture, pOldDSV);

	float ClearColor2[4] = { 0.0f, 0.0f, 0.0f, 1.0f }; 
	g_pImmediateContext->ClearRenderTargetView(g_pRenderTargetView_RenderToTexture, ClearColor2);

	// Save the old viewport
	D3D11_VIEWPORT vpOld[D3D11_VIEWPORT_AND_SCISSORRECT_MAX_INDEX];
	UINT nViewPorts = 1;
	g_pImmediateContext->RSGetViewports(&nViewPorts, vpOld);

	// Setup the viewport to match the backbuffer
	D3D11_VIEWPORT vp;
	vp.Width = static_cast<float>(g_RenderTargetWidth);
	vp.Height = static_cast<float>(g_RenderTargetHeight);
	vp.MinDepth = 0.0f;
	vp.MaxDepth = 1.0f;
	vp.TopLeftX = 0;
	vp.TopLeftY = 0;
	g_pImmediateContext->RSSetViewports(1, &vp);

	UINT strides = sizeof(SCREEN_VERTEX);
	UINT offsets = 0;
	ID3D11Buffer* pBuffers[1] = { g_ScreenQuadVertexBuffer };

	g_pImmediateContext->IASetInputLayout(g_pQuadLayout);
	g_pImmediateContext->IASetVertexBuffers(0, 1, pBuffers, &strides, &offsets);
	g_pImmediateContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);

	ID3D11ShaderResourceView* aRViews[2] = { g_pFragmentLinkSRV, g_pStartOffsetSRV };
	g_pImmediateContext->PSSetShaderResources(0, 2, aRViews);

	g_pImmediateContext->VSSetShader(g_ScreenQuadVertexShader, NULL, 0);
	g_pImmediateContext->PSSetShader(g_pPSSortFragmentsAndRender, NULL, 0);
	g_pImmediateContext->Draw(4, 0);

	// Restore the Old viewport
	g_pImmediateContext->RSSetViewports(nViewPorts, vpOld);

	ID3D11ShaderResourceView* aRViewsNull[2] = { NULL, NULL };
	g_pImmediateContext->PSSetShaderResources(0, 2, aRViewsNull);
	
	//Unbind Render to texture and bind back backbuffer
	g_pImmediateContext->OMSetRenderTargets(1, &pOldRTV, pOldDSV);

	g_pImmediateContext->ClearRenderTargetView(pOldRTV, ClearColor);
	//	D3D11_DEPTH_STENCIL_VIEW_DESC dec; 
	//	pOldDSV->GetDesc(&dec);

	//renderizar a cena de novo só aceitando pixels da distancia da depth texture
	OnRender(true);

	// Restore saved render state.
	g_pImmediateContext->RSSetState(pOldState);
	SAFE_RELEASE(pOldState);

	// Restore saved depth/stencil state.
	g_pImmediateContext->OMSetDepthStencilState(pDepthStencilStateStored, stencilRef);
	SAFE_RELEASE(pDepthStencilStateStored);
	
	g_pImmediateContext->OMSetRenderTargets(1, &pOldRTV, pOldDSV);

	// Restore original render targets, and then 
	SAFE_RELEASE(pOldRTV);
	SAFE_RELEASE(pOldDSV);

	// Present the information rendered to the back buffer to the front buffer (the screen)
	g_pSwapChain->Present(0, 0);

}

void OnRender(bool finalPass)
{
	XMMATRIX scale1 = XMMatrixScaling(10.1f, 10.0f, 10.0f);

	DrawObject(g_pMPlusVertexBuffer, g_pMPlusIndexBuffer, 36, &scale1, MRM_MESH_RENDER_NORMAL, finalPass, false);

	XMMATRIX scale = XMMatrixScaling(30.0f, 30.0f, 30.0f);

	XMMATRIX world = XMMatrixTranslation(g_MeshPosition.x, g_MeshPosition.y, g_MeshPosition.z);

	world = world * scale;

	if (g_isDeformationOn)
		DrawObject(g_pMMinusVertexBuffer, g_pMMinusIndexBuffer, 208890, &world, MRM_MESH_RENDER_SUBTRACT, finalPass, true);
	else
		DrawObject(g_pMMinusVertexBuffer, g_pMMinusIndexBuffer, 208890, &world, MRM_MESH_RENDER_NORMAL, finalPass, false);
}

void DrawObject(ID3D11Buffer* vertexBuffer, ID3D11Buffer* indexBuffer, int numIndices, XMMATRIX* pmWorld, const MESH_RENDER_MODE draw_render_mode, bool IsFrameBufferRenderPass, const bool invert_normal)
{
	g_pImmediateContext->IASetInputLayout(g_pSceneVertexLayout);

	UINT uStrides = sizeof(CSGVertex);
	UINT uOffsets = 0;
	g_pImmediateContext->IASetVertexBuffers(0, 1, &vertexBuffer, &uStrides, &uOffsets);
	g_pImmediateContext->IASetIndexBuffer(indexBuffer, DXGI_FORMAT_R32_UINT, 0);
	g_pImmediateContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

	g_pImmediateContext->VSSetShader(g_pSceneVertexShader, NULL, 0);
	ID3D11DepthStencilState* pOldDSState = NULL;
	UINT oldStencilRef = (UINT)-1;

	if (IsFrameBufferRenderPass)
	{
		g_pImmediateContext->PSSetShader(g_pPixelShader, NULL, 0);
		g_pImmediateContext->PSSetShaderResources(0, 1, &g_pShaderResourceView);
		g_pImmediateContext->PSSetSamplers(0, 1, &g_pPSSampler);
		g_pImmediateContext->OMGetDepthStencilState(&pOldDSState, &oldStencilRef);
		g_pImmediateContext->OMSetDepthStencilState(g_pDSState, 0);
	}

	// Update the constant buffer
	D3D11_MAPPED_SUBRESOURCE MappedResource;
	g_pImmediateContext->Map(g_pCameraConstantBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &MappedResource);
	Camera_CB* pVS_CB = (Camera_CB*) MappedResource.pData;
	pVS_CB->World = *pmWorld;
	pVS_CB->ViewProj = g_Camera->GetViewMatrix() * g_Camera->GetProjectionMatrix();
	g_pImmediateContext->Unmap(g_pCameraConstantBuffer, 0);
	g_pImmediateContext->VSSetConstantBuffers(0, 1, &g_pCameraConstantBuffer);

	SHADER_SUBTRACT_OPERATION shader_op = (draw_render_mode == MRM_MESH_RENDER_SUBTRACT) ? SSO_MESH_MINUS_FRONT_FACING_PIXEL : SSO_MESH_PLUS_FRONT_FACING_PIXEL;

	g_pImmediateContext->Map(g_pShaderOperationConstantBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &MappedResource);
	PS_CB_ShaderOperation* pPS_CB = (PS_CB_ShaderOperation*) MappedResource.pData;
	pPS_CB->ShaderSubtractOperation = (unsigned int) shader_op;
	g_pImmediateContext->Unmap(g_pShaderOperationConstantBuffer, 0);
	g_pImmediateContext->PSSetConstantBuffers(1, 1, &g_pShaderOperationConstantBuffer);

	ID3D11RasterizerState* oldRS;
	g_pImmediateContext->RSGetState(&oldRS);

	if (!IsFrameBufferRenderPass || draw_render_mode == MRM_MESH_RENDER_NORMAL)
	{
		g_pImmediateContext->RSSetState(g_pFrontFacingRasterizerState);
		g_pImmediateContext->DrawIndexed(numIndices, 0, 0);
	}

	shader_op = (draw_render_mode == MRM_MESH_RENDER_SUBTRACT) ? SSO_MESH_MINUS_BACK_FACING_PIXEL : SSO_MESH_PLUS_BACK_FACING_PIXEL;

	g_pImmediateContext->Map(g_pShaderOperationConstantBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &MappedResource);
	pPS_CB = (PS_CB_ShaderOperation*) MappedResource.pData;
	pPS_CB->ShaderSubtractOperation = (unsigned int) shader_op;
	g_pImmediateContext->Unmap(g_pShaderOperationConstantBuffer, 0);
	g_pImmediateContext->PSSetConstantBuffers(1, 1, &g_pShaderOperationConstantBuffer);

	//Camera position CB
	g_pImmediateContext->Map(g_pRenderInformationConstantBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &MappedResource);
	PS_CB_RenderInformation* pPS_CB3 = (PS_CB_RenderInformation*) MappedResource.pData;
	pPS_CB3->eyePos.x = XMVectorGetX(g_Camera->GetPosition());
	pPS_CB3->eyePos.y = XMVectorGetY(g_Camera->GetPosition());
	pPS_CB3->eyePos.z = XMVectorGetZ(g_Camera->GetPosition());
	pPS_CB3->invert_normal = IsFrameBufferRenderPass && invert_normal ;
	g_pImmediateContext->Unmap(g_pRenderInformationConstantBuffer, 0);
	g_pImmediateContext->PSSetConstantBuffers(2, 1, &g_pRenderInformationConstantBuffer);

	if (!IsFrameBufferRenderPass || draw_render_mode == MRM_MESH_RENDER_SUBTRACT)
	{
		g_pImmediateContext->RSSetState(g_pBackFacingRasterizerState);
		g_pImmediateContext->DrawIndexed(numIndices, 0, 0);
	}

	g_pImmediateContext->RSSetState(oldRS);
	if (IsFrameBufferRenderPass)
	{
		g_pImmediateContext->OMSetDepthStencilState(pOldDSState, oldStencilRef);
	}

	SAFE_RELEASE(oldRS);
	SAFE_RELEASE(pOldDSState);
}

std::vector<std::string> splitString(std::string str, char separator)
{
	std::vector<std::string> strings;

	for (unsigned int i = 0; i < str.size(); i++)
	{
		std::string current;
		while (i < str.size() && str[i] != separator)
		{
			current += str[i];
			i++;
		}
		strings.push_back(current);
	}

	return strings;
}

HRESULT loadFromObj(ID3D11Device* pDevice, ID3D11Buffer*& vertexBuffer, ID3D11Buffer*& indexBuffer, char* filePath, XMFLOAT4 color)
{
	HRESULT hr;

	std::string line;
	std::ifstream in(filePath);

	std::vector<XMFLOAT3> objVertices;
	std::vector<XMFLOAT3> objNormals;
	std::vector<int> objIndicesVertices;	

	enum ReadState
	{
		NONE,
		READING_VERTICES,
		READING_INDEXES
	};

	ReadState read_state = NONE;

	while (std::getline(in, line))
	{
		std::vector<std::string> splitStr = splitString(line, ' ');

		if (splitStr.size() == 0) continue;

		if (!strcmp(splitStr[0].c_str(), "end_header"))
		{
			assert(read_state == NONE);
			read_state = READING_VERTICES;
			continue;
		}

		if (!strcmp(splitStr[0].c_str(), "3") && splitStr.size() == 4 && read_state != READING_INDEXES) 
		{
			assert(read_state == READING_VERTICES);
			read_state = READING_INDEXES;			
		}

		switch (read_state)
		{
		case NONE: 
			break;
		case READING_VERTICES:
			assert(splitStr.size() == 6);
			objVertices.push_back(XMFLOAT3((float) atof(splitStr[0].c_str()), (float) atof(splitStr[1].c_str()), (float) atof(splitStr[2].c_str())));
			objNormals.push_back(XMFLOAT3((float) atof(splitStr[3].c_str()), (float) atof(splitStr[4].c_str()), (float) atof(splitStr[5].c_str())));
			break;
		case READING_INDEXES:
			assert(splitStr.size() == 4);
			objIndicesVertices.push_back(atoi(splitStr[1].c_str()) );
			objIndicesVertices.push_back(atoi(splitStr[2].c_str()) );
			objIndicesVertices.push_back(atoi(splitStr[3].c_str()) );			
			break;
		}
	}


	std::vector<CSGVertex> sceneVertices;

	for (unsigned int i = 0; i < objVertices.size(); i++)
	{
		CSGVertex vertex;
		vertex.pos = XMFLOAT4(objVertices[i].x, objVertices[i].y, objVertices[i].z, 1.0f);
		vertex.normal = XMFLOAT3(objNormals[i].x, objNormals[i].y, objNormals[i].z);
		vertex.color = color;
		sceneVertices.push_back(vertex);
	}

	D3D11_BUFFER_DESC vbDesc;
	vbDesc.ByteWidth = sceneVertices.size()*sizeof(CSGVertex);
	vbDesc.Usage = D3D11_USAGE_IMMUTABLE;
	vbDesc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
	vbDesc.CPUAccessFlags = 0;
	vbDesc.MiscFlags = 0;

	D3D11_SUBRESOURCE_DATA vbSR;
	vbSR.pSysMem = &sceneVertices[0];
	HR_RETURN(pDevice->CreateBuffer(&vbDesc, &vbSR, &vertexBuffer));

	vbDesc.ByteWidth = objIndicesVertices.size()*sizeof(int) ;
	vbDesc.BindFlags = D3D11_BIND_INDEX_BUFFER;

	vbSR.pSysMem = &objIndicesVertices[0];
	HR_RETURN(pDevice->CreateBuffer(&vbDesc, &vbSR, &indexBuffer));

	return S_OK;

}




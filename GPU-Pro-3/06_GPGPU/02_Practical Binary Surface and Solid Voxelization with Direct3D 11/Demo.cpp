//==============================================================================================================================================================
// Simple demo accompanying the GPU Pro 3 book chapter "Practical Binary Surface and Solid Voxelization with Direct3D 11"
// Copyright (c) 2011 Michael Schwarz. All rights reserved.
//--------------------------------------------------------------------------------------------------------------------------------------------------------------
// Parts of the code are loosely based on examples from the DirectX SDK June 2010
// Copyright (c) Microsoft Corporation. All rights reserved.
//==============================================================================================================================================================

#include "DXUT.h"
#include "DXUTgui.h"
#include "DXUTmisc.h"
#include "DXUTCamera.h"
#include "DXUTSettingsDlg.h"
#include "SDKmisc.h"
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <boost/unordered_map.hpp>		// use boost's implementation because the one shipped with VS 2010 is catastrophically slow in Debug mode

//==============================================================================================================================================================

CModelViewerCamera g_camera;
CDXUTDialogResourceManager g_dialogResourceManager;
CD3DSettingsDlg g_settingsDlg;
CDXUTDialog g_dlg;						// currently not used as basic UI is realized by keys
CDXUTTextHelper* g_textHelper = NULL;

// states
ID3D11RasterizerState* g_rastDefault = NULL;
ID3D11RasterizerState* g_rastNoCull = NULL;

ID3D11Query* g_qryTimestamp1 = NULL;
ID3D11Query* g_qryTimestamp2 = NULL;
ID3D11Query* g_qryTimestampDisjoint = NULL;

// input mesh
ID3D11Buffer* g_vbMesh = NULL;
ID3D11Buffer* g_ibMesh = NULL;
ID3D11ShaderResourceView* g_srvVbMesh = NULL;
ID3D11ShaderResourceView* g_srvIbMesh = NULL;
ID3D11InputLayout* g_ilytMesh = NULL;
UINT g_bytesPerMeshVertex;
UINT g_numMeshVertices;
UINT g_numMeshIndices;

// full-screen quad
ID3D11Buffer* g_vbQuad = NULL;
ID3D11InputLayout* g_ilytQuad = NULL;
UINT g_bytesPerQuadVertex;

// voxelization buffer
ID3D11Buffer* g_bufVoxelization = NULL;
ID3D11UnorderedAccessView* g_uavVoxelization = NULL;
ID3D11ShaderResourceView* g_srvVoxelization = NULL;

// dummy render target for rasterization-based voxelization
ID3D11Texture2D* g_texVoxelizationDummy = NULL;
ID3D11RenderTargetView* g_rtvVoxelizationDummy = NULL;

// shaders
ID3D11VertexShader* g_vsRenderModel = NULL;
ID3D11PixelShader* g_psRenderModel = NULL;

ID3D11VertexShader* g_vsVoxelize = NULL;
ID3D11PixelShader* g_psVoxelizeSurface = NULL;
ID3D11PixelShader* g_psVoxelizeSolid = NULL;

ID3D11VertexShader* g_vsRenderVoxelizationRaycasting = NULL;
ID3D11PixelShader* g_psRenderVoxelizationRaycasting = NULL;

ID3D11ComputeShader* g_csVoxelizeSolid = NULL;
ID3D11ComputeShader* g_csVoxelizeSolid_Propagate = NULL;
ID3D11ComputeShader* g_csVoxelizeSurfaceConservative = NULL;

// configuration
bool g_displayVoxelization = false;
bool g_showVoxelBorderLines = true;

enum VoxelizationMethod {
	VOXELIZATION_SOLID_PS,
	VOXELIZATION_SOLID_COMPUTE,
	VOXELIZATION_SURFACE_PS,
	VOXELIZATION_SURFACE_CONSERVATIVE_COMPUTE,
};
bool g_voxelize = false;
UINT g_voxelizationMethod = VOXELIZATION_SURFACE_PS;

bool g_validVoxelization = false;
double g_secsVoxelization = 0.0;

UINT g_gridSizeX = 128;
UINT g_gridSizeY = 128;
UINT g_gridSizeZ = 128;
bool g_useCubeVoxels = false;

UINT g_strideX;
UINT g_strideY;
UINT g_dataSize;

D3DXVECTOR3 g_voxelSpace[2];			// min/max corners of axis-aligned box encompassing the voxel grid
D3DXMATRIX g_matWorldToVoxel;
D3DXMATRIX g_matWorldToVoxelProj;

D3DXVECTOR3 g_aabbModel[2];
D3DXMATRIX g_matView;
D3DXMATRIX g_matProj;
D3DXVECTOR3 g_wLightPos;

// constant buffer layouts
__declspec(align(16)) struct CB_PerFrame {
	D3DXVECTOR3 m_wLightPos;
};

__declspec(align(16)) struct CB_PerObject {
	D3DXMATRIX m_matModelToWorld;
	D3DXMATRIX m_matModelToProj;
	D3DXVECTOR3 m_colDiffuse;
};

__declspec(align(16)) struct CB_VoxelGrid {
	D3DXMATRIX m_matModelToProj;
	D3DXMATRIX m_matModelToVoxel;
	UINT m_stride[2+2];
	UINT m_gridSize[3];
};

__declspec(align(16)) struct CB_ModelInput {
	UINT m_numModelTriangles;
	UINT m_vertexFloatStride;
};

__declspec(align(16)) struct CB_Raycasting {
	D3DXMATRIX m_matQuadToVoxel;
	D3DXMATRIX m_matVoxelToScreen;
	D3DXVECTOR3 m_rayOrigin;
	float padding1;
	D3DXVECTOR3 m_voxLightPos;
	float padding2;
	UINT m_stride[2+2];
	UINT m_gridSize[3];
	BOOL m_showLines;
};

// constant buffers
ID3D11Buffer* g_cbPerFrame = NULL;
ID3D11Buffer* g_cbPerObject = NULL;
ID3D11Buffer* g_cbVoxelGrid = NULL;
ID3D11Buffer* g_cbModelInput = NULL;
ID3D11Buffer* g_cbRaycasting = NULL;

//==============================================================================================================================================================

bool CALLBACK IsDeviceAcceptable(const CD3D11EnumAdapterInfo* AdapterInfo, UINT Output, const CD3D11EnumDeviceInfo* DeviceInfo, DXGI_FORMAT BackBufferFormat, bool bWindowed, void* pUserContext);
bool CALLBACK ModifyDeviceSettings(DXUTDeviceSettings* pDeviceSettings, void* pUserContext);
HRESULT CALLBACK OnCreateDevice(ID3D11Device* pd3dDevice, const DXGI_SURFACE_DESC* pBackBufferSurfaceDesc, void* pUserContext);
void CALLBACK OnDestroyDevice(void* pUserContext);
HRESULT CALLBACK OnResizedSwapChain(ID3D11Device* pd3dDevice, IDXGISwapChain* pSwapChain, const DXGI_SURFACE_DESC* pBackBufferSurfaceDesc, void* pUserContext);
void CALLBACK OnReleasingSwapChain(void* pUserContext);
void CALLBACK OnFrameMove(double fTime, float fElapsedTime, void* pUserContext);
void CALLBACK OnFrameRender(ID3D11Device* pd3dDevice, ID3D11DeviceContext* pd3dImmediateContext, double fTime, float fElapsedTime, void* pUserContext);
LRESULT CALLBACK MsgProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam, bool* pbNoFurtherProcessing, void* pUserContext);
void CALLBACK OnKeyboard(UINT nChar, bool bKeyDown, bool bAltDown, void* pUserContext);
void CALLBACK OnGUIEvent(UINT nEvent, int nControlID, CDXUTControl* pControl, void* pUserContext);

void InitApp();

HRESULT CreateVoxelizationResources(ID3D11Device* pd3dDevice, ID3D11DeviceContext* pd3dImmediateContext);
void ReleaseVoxelizationResources();
void SetupVoxelization();
void VoxelizeViaRendering(ID3D11DeviceContext* pd3dImmediateContext);

void RenderModel(ID3D11DeviceContext* pd3dImmediateContext);
void RenderText();

//==============================================================================================================================================================

int WINAPI wWinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPWSTR lpCmdLine, int nCmdShow) {
#if defined(DEBUG) | defined(_DEBUG)
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
#endif

	DXUTSetCallbackMsgProc(MsgProc);
	DXUTSetCallbackKeyboard(OnKeyboard);
	DXUTSetCallbackFrameMove(OnFrameMove);
	DXUTSetCallbackDeviceChanging(ModifyDeviceSettings);

	DXUTSetCallbackD3D11DeviceAcceptable(IsDeviceAcceptable);
	DXUTSetCallbackD3D11DeviceCreated(OnCreateDevice);
	DXUTSetCallbackD3D11SwapChainResized(OnResizedSwapChain);
	DXUTSetCallbackD3D11SwapChainReleasing(OnReleasingSwapChain);
	DXUTSetCallbackD3D11DeviceDestroyed(OnDestroyDevice);
	DXUTSetCallbackD3D11FrameRender(OnFrameRender);

	InitApp();
	DXUTInit(true, true, NULL);
	DXUTSetCursorSettings(true, true);
	DXUTCreateWindow(L"Voxelization demo");
	DXUTCreateDevice(D3D_FEATURE_LEVEL_11_0, true, 1024, 768);

	DXUTMainLoop();

	return DXUTGetExitCode();
}

void InitApp() {
	g_settingsDlg.Init(&g_dialogResourceManager);
	g_dlg.Init(&g_dialogResourceManager);
	g_dlg.SetCallback(OnGUIEvent);
}

//--------------------------------------------------------------------------------------------------------------------------------------------------------------

bool CALLBACK IsDeviceAcceptable(const CD3D11EnumAdapterInfo* AdapterInfo, UINT Output, const CD3D11EnumDeviceInfo* DeviceInfo, DXGI_FORMAT BackBufferFormat, bool bWindowed, void* pUserContext) {
	return true;
}

bool CALLBACK ModifyDeviceSettings(DXUTDeviceSettings* pDeviceSettings, void* pUserContext) {
	// turn vsync off
	pDeviceSettings->d3d11.SyncInterval = 0;
	g_settingsDlg.GetDialogControl()->GetComboBox(DXUTSETTINGSDLG_PRESENT_INTERVAL)->SetEnabled(false);

	return true;
}

//--------------------------------------------------------------------------------------------------------------------------------------------------------------

HRESULT CompileShaderFromFile(WCHAR* szFileName, LPCSTR szEntryPoint, LPCSTR szShaderModel, ID3DBlob** ppBlobOut, const D3D_SHADER_MACRO* pDefines = NULL) {
	DWORD dwShaderFlags = D3DCOMPILE_ENABLE_STRICTNESS;
	#if defined(DEBUG) || defined(_DEBUG)
		dwShaderFlags |= D3DCOMPILE_DEBUG;
	#endif

	// compile the shader
	ID3DBlob* pErrorBlob = NULL;
	HRESULT hr = D3DX11CompileFromFile(szFileName, pDefines, NULL, szEntryPoint, szShaderModel, dwShaderFlags, NULL, NULL, ppBlobOut, &pErrorBlob, NULL);

	if(FAILED(hr) && pErrorBlob != NULL)
		OutputDebugStringA((char*)pErrorBlob->GetBufferPointer());

	SAFE_RELEASE(pErrorBlob);
	return hr;
}

HRESULT CreateVertexShader(ID3D11Device* pd3dDevice, WCHAR* szFileName, LPCSTR szEntryPoint, LPCSTR szShaderModel, ID3D11VertexShader** ppVertexShader, ID3DBlob** ppBlob = NULL) {
	ID3DBlob* pBlob = NULL;
	HRESULT hr;
	if(FAILED(hr = CompileShaderFromFile(szFileName, szEntryPoint, szShaderModel, &pBlob)) ||
		FAILED(hr = pd3dDevice->CreateVertexShader(pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, ppVertexShader)))
	{
		SAFE_RELEASE(pBlob);
	} else {
		DXUT_SetDebugName(*ppVertexShader, szEntryPoint);
		if(ppBlob)
			*ppBlob = pBlob;
		else
			SAFE_RELEASE(pBlob);
	}
	return hr;
}

HRESULT CreatePixelShader(ID3D11Device* pd3dDevice, WCHAR* szFileName, LPCSTR szEntryPoint, LPCSTR szShaderModel, ID3D11PixelShader** ppPixelShader) {
	ID3DBlob* pBlob = NULL;
	HRESULT hr;
	if(!FAILED(hr = CompileShaderFromFile(szFileName, szEntryPoint, szShaderModel, &pBlob)) &&
		!FAILED(hr = pd3dDevice->CreatePixelShader(pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, ppPixelShader)))
	{
		DXUT_SetDebugName(*ppPixelShader, szEntryPoint);
	}
	SAFE_RELEASE(pBlob);
	return hr;
}

HRESULT CreateComputeShader(ID3D11Device* pd3dDevice, WCHAR* szFileName, LPCSTR szEntryPoint, LPCSTR szShaderModel, ID3D11ComputeShader** ppComputeShader) {
	ID3DBlob* pBlob = NULL;
	HRESULT hr;
	if(!FAILED(hr = CompileShaderFromFile(szFileName, szEntryPoint, szShaderModel, &pBlob)) &&
		!FAILED(hr = pd3dDevice->CreateComputeShader(pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, ppComputeShader)))
	{
		DXUT_SetDebugName(*ppComputeShader, szEntryPoint);
	}
	SAFE_RELEASE(pBlob);
	return hr;
}

//--------------------------------------------------------------------------------------------------------------------------------------------------------------

HRESULT LoadModel(ID3D11Device* pd3dDevice, char* filename) {
	// extremely basic OBJ loader
	std::ifstream objFile(filename);
	if(objFile.fail())
		return E_FAIL;

	struct Vertex {
		D3DXVECTOR3 m_position;
		D3DXVECTOR3 m_normal;
	};

	std::vector<D3DXVECTOR3> positions;
	std::vector<D3DXVECTOR3> normals;
	std::vector<Vertex> vertices;
	std::vector<UINT32> indices;
	typedef boost::unordered_map<UINT64, UINT32> map_t;
	map_t existingVertices;

	std::string line, keyword;
	while(!objFile.eof() && std::getline(objFile, line)) {
		std::istringstream ss(line);
		if(!(ss >> keyword))
			continue;
		if(keyword.compare("v") == 0) {
			D3DXVECTOR3 v;
			ss >> v.x >> v.y >> v.z;
			positions.push_back(v);
		} else if(keyword.compare("vn") == 0) {
			D3DXVECTOR3 vn;
			ss >> vn.x >> vn.y >> vn.z;
			normals.push_back(vn);
		} else if(keyword.compare("f") == 0) {
			for(int i = 0; i < 3; i++) {	// v/vt/vn, v, v/vt, v//vn
				int indexP, indexT = 0, indexN = 0;
				ss >> indexP;
				if(ss.peek() == '/') {
					ss.ignore();
					if(ss.peek() != '/') {
						ss >> indexT;
					}
					if(ss.peek() == '/') {
						ss.ignore();
						ss >> indexN;
					}
				}

				if(indexP > int(positions.size()))
					return E_FAIL;
				if(indexN > int(normals.size()))
					return E_FAIL;

				UINT64 key = UINT32(indexP);
				key |= UINT64(indexN) << 32;

				map_t::iterator it = existingVertices.find(key);
				if(it != existingVertices.end()) {
					indices.push_back(it->second);
				} else {
					Vertex v;
					v.m_position = positions[indexP - 1];
					v.m_normal = (indexN > 0) ? normals[indexN - 1] : D3DXVECTOR3(0.0f, 0.0f, 0.0f);
					UINT32 index = UINT32(vertices.size());
					vertices.push_back(v);
					indices.push_back(index);
					existingVertices[key] = index;
				}
			}
		}
	}

	if(vertices.empty())
		return E_FAIL;

	// determine bounding box
	g_aabbModel[0] = g_aabbModel[1] = vertices[0].m_position;
	for(std::vector<Vertex>::iterator it = vertices.begin(); ++it != vertices.end(); ) {
		g_aabbModel[0].x = min(g_aabbModel[0].x, it->m_position.x);
		g_aabbModel[0].y = min(g_aabbModel[0].y, it->m_position.y);
		g_aabbModel[0].z = min(g_aabbModel[0].z, it->m_position.z);
		g_aabbModel[1].x = max(g_aabbModel[1].x, it->m_position.x);
		g_aabbModel[1].y = max(g_aabbModel[1].y, it->m_position.y);
		g_aabbModel[1].z = max(g_aabbModel[1].z, it->m_position.z);
	}

	// create buffers
	HRESULT hr;

	g_bytesPerMeshVertex = sizeof(Vertex);
	g_numMeshVertices = UINT(vertices.size());
	g_numMeshIndices = UINT(indices.size());

	D3D11_BUFFER_DESC bufferDesc;
	bufferDesc.CPUAccessFlags = 0;
	bufferDesc.MiscFlags = 0;

	D3D11_SUBRESOURCE_DATA initialData;
	initialData.SysMemPitch = 0;
	initialData.SysMemSlicePitch = 0;

	// create vertex buffer
	bufferDesc.ByteWidth = UINT(g_bytesPerMeshVertex * g_numMeshVertices);
	bufferDesc.Usage = D3D11_USAGE_DEFAULT;
	bufferDesc.BindFlags = D3D11_BIND_VERTEX_BUFFER | D3D11_BIND_SHADER_RESOURCE;
	bufferDesc.StructureByteStride = UINT(g_bytesPerMeshVertex);

	initialData.pSysMem = &vertices[0];

	V_RETURN(pd3dDevice->CreateBuffer(&bufferDesc, &initialData, &g_vbMesh));

	// create shader resource view for vertex buffer
	D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc;
	srvDesc.Format = DXGI_FORMAT_R32_FLOAT;
	srvDesc.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
	srvDesc.Buffer.ElementOffset = 0;
	srvDesc.Buffer.ElementWidth = bufferDesc.ByteWidth / 4;
	V_RETURN(pd3dDevice->CreateShaderResourceView(g_vbMesh, &srvDesc, &g_srvVbMesh));

	// create index buffer
	bufferDesc.ByteWidth = UINT(sizeof(UINT32) * g_numMeshIndices);
	bufferDesc.Usage = D3D11_USAGE_DEFAULT;
	bufferDesc.BindFlags = D3D11_BIND_INDEX_BUFFER | D3D11_BIND_SHADER_RESOURCE;
	bufferDesc.StructureByteStride = 0;

	initialData.pSysMem = &indices[0];

	V_RETURN(pd3dDevice->CreateBuffer(&bufferDesc, &initialData, &g_ibMesh));

	// create shader resource view for index buffer
	srvDesc.Format = DXGI_FORMAT_R32_UINT;
	srvDesc.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
	srvDesc.Buffer.ElementOffset = 0;
	srvDesc.Buffer.ElementWidth = bufferDesc.ByteWidth / 4;
	V_RETURN(pd3dDevice->CreateShaderResourceView(g_ibMesh, &srvDesc, &g_srvIbMesh));

	return S_OK;
}

//--------------------------------------------------------------------------------------------------------------------------------------------------------------

HRESULT CreateFullscreenQuad(ID3D11Device* pd3dDevice) {
	struct Vertex {
		D3DXVECTOR3 position;
		D3DXVECTOR2 texcoord;
	};

	Vertex vertices[4] = {
		D3DXVECTOR3(-1.0f, -1.0f, 0.5f), D3DXVECTOR2(0.0f, 1.0f),
		D3DXVECTOR3( 1.0f, -1.0f, 0.5f), D3DXVECTOR2(1.0f, 1.0f),
		D3DXVECTOR3(-1.0f,  1.0f, 0.5f), D3DXVECTOR2(0.0f, 0.0f),
		D3DXVECTOR3( 1.0f,  1.0f, 0.5f), D3DXVECTOR2(1.0f, 0.0f),
	};

	g_bytesPerQuadVertex = sizeof(Vertex);

	D3D11_BUFFER_DESC bufferDesc;
	bufferDesc.ByteWidth = UINT(g_bytesPerQuadVertex * 4);
	bufferDesc.Usage = D3D11_USAGE_IMMUTABLE;
	bufferDesc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
	bufferDesc.CPUAccessFlags = 0;
	bufferDesc.MiscFlags = 0;
	bufferDesc.StructureByteStride = 0;

	D3D11_SUBRESOURCE_DATA initialData;
	initialData.pSysMem = vertices;
	initialData.SysMemPitch = 0;
	initialData.SysMemSlicePitch = 0;

	return pd3dDevice->CreateBuffer(&bufferDesc, &initialData, &g_vbQuad);
}

//--------------------------------------------------------------------------------------------------------------------------------------------------------------

HRESULT CALLBACK OnCreateDevice(ID3D11Device* pd3dDevice, const DXGI_SURFACE_DESC* pBackBufferSurfaceDesc, void* pUserContext) {
	HRESULT hr;

	ID3D11DeviceContext* pd3dImmediateContext = DXUTGetD3D11DeviceContext();
	V_RETURN(g_dialogResourceManager.OnD3D11CreateDevice(pd3dDevice, pd3dImmediateContext));
	V_RETURN(g_settingsDlg.OnD3D11CreateDevice(pd3dDevice));
	g_textHelper = new CDXUTTextHelper(pd3dDevice, pd3dImmediateContext, &g_dialogResourceManager, 15);

	// shaders and input layouts
	const D3D11_INPUT_ELEMENT_DESC inputElementDescsMesh[] = {
		"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0,  0, D3D11_INPUT_PER_VERTEX_DATA, 0,
		"NORMAL",   0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 12, D3D11_INPUT_PER_VERTEX_DATA, 0,
	};

	const D3D11_INPUT_ELEMENT_DESC inputElementDescsQuad[] = {
		"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0,  0, D3D11_INPUT_PER_VERTEX_DATA, 0,
		"TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT,    0, 12, D3D11_INPUT_PER_VERTEX_DATA, 0,
	};

	ID3DBlob* pBlob = NULL;
	V_RETURN(CreateVertexShader(pd3dDevice, L"Rendering.hlsl", "VS_RenderModel", "vs_5_0", &g_vsRenderModel, &pBlob));
	V_RETURN(pd3dDevice->CreateInputLayout(inputElementDescsMesh, ARRAYSIZE(inputElementDescsMesh), pBlob->GetBufferPointer(), pBlob->GetBufferSize(), &g_ilytMesh));
	DXUT_SetDebugName(g_ilytMesh, "ilMesh");
	SAFE_RELEASE(pBlob);
	V_RETURN(CreatePixelShader(pd3dDevice, L"Rendering.hlsl", "PS_RenderModel", "ps_5_0", &g_psRenderModel));

	V_RETURN(CreateVertexShader(pd3dDevice, L"Voxelization.hlsl", "VS_Voxelize", "vs_5_0", &g_vsVoxelize));
	V_RETURN(CreatePixelShader(pd3dDevice, L"Voxelization.hlsl", "PS_VoxelizeSurface", "ps_5_0", &g_psVoxelizeSurface));
	V_RETURN(CreatePixelShader(pd3dDevice, L"Voxelization.hlsl", "PS_VoxelizeSolid", "ps_5_0", &g_psVoxelizeSolid));
	V_RETURN(CreateComputeShader(pd3dDevice, L"Voxelization.hlsl", "CS_VoxelizeSolid", "cs_5_0", &g_csVoxelizeSolid));
	V_RETURN(CreateComputeShader(pd3dDevice, L"Voxelization.hlsl", "CS_VoxelizeSolid_Propagate", "cs_5_0", &g_csVoxelizeSolid_Propagate));
	V_RETURN(CreateComputeShader(pd3dDevice, L"Voxelization.hlsl", "CS_VoxelizeSurfaceConservative", "cs_5_0", &g_csVoxelizeSurfaceConservative));

	V_RETURN(CreateVertexShader(pd3dDevice, L"Raycasting.hlsl", "VS_RenderVoxelizationRaycasting", "vs_5_0", &g_vsRenderVoxelizationRaycasting, &pBlob));
	V_RETURN(pd3dDevice->CreateInputLayout(inputElementDescsQuad, ARRAYSIZE(inputElementDescsMesh), pBlob->GetBufferPointer(), pBlob->GetBufferSize(), &g_ilytQuad));
	DXUT_SetDebugName(g_ilytQuad, "ilQuad");
	SAFE_RELEASE(pBlob);
	V_RETURN(CreatePixelShader(pd3dDevice, L"Raycasting.hlsl", "PS_RenderVoxelizationRaycasting", "ps_5_0", &g_psRenderVoxelizationRaycasting));

	// constant buffers
	D3D11_BUFFER_DESC bufDesc;
	bufDesc.Usage = D3D11_USAGE_DYNAMIC;
	bufDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    bufDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    bufDesc.MiscFlags = 0;
    bufDesc.StructureByteStride = 0;

	bufDesc.ByteWidth = sizeof(CB_PerFrame);
	V_RETURN(pd3dDevice->CreateBuffer(&bufDesc, NULL, &g_cbPerFrame));
	DXUT_SetDebugName(g_cbPerFrame, "cbPerFrame");

	bufDesc.ByteWidth = sizeof(CB_PerObject);
	V_RETURN(pd3dDevice->CreateBuffer(&bufDesc, NULL, &g_cbPerObject));
	DXUT_SetDebugName(g_cbPerObject, "cbPerObject");

	bufDesc.ByteWidth = sizeof(CB_VoxelGrid);
	V_RETURN(pd3dDevice->CreateBuffer(&bufDesc, NULL, &g_cbVoxelGrid));
	DXUT_SetDebugName(g_cbVoxelGrid, "cbVoxelGrid");

	bufDesc.ByteWidth = sizeof(CB_ModelInput);
	V_RETURN(pd3dDevice->CreateBuffer(&bufDesc, NULL, &g_cbModelInput));
	DXUT_SetDebugName(g_cbModelInput, "cbModelInput");

	bufDesc.ByteWidth = sizeof(CB_Raycasting);
	V_RETURN(pd3dDevice->CreateBuffer(&bufDesc, NULL, &g_cbRaycasting));
	DXUT_SetDebugName(g_cbRaycasting, "cbRaycasting");

	// rasterizer states
	CD3D11_RASTERIZER_DESC rastDesc = CD3D11_RASTERIZER_DESC(CD3D11_DEFAULT());
	rastDesc.FrontCounterClockwise = true;

	V_RETURN(pd3dDevice->CreateRasterizerState(&rastDesc, &g_rastDefault));
	DXUT_SetDebugName(g_rastDefault, "rastDefault");

	rastDesc.CullMode = D3D11_CULL_NONE;
	V_RETURN(pd3dDevice->CreateRasterizerState(&rastDesc, &g_rastNoCull));
	DXUT_SetDebugName(g_rastNoCull, "rastNoCull");

	// query states
	D3D11_QUERY_DESC qryDesc;

	qryDesc.Query = D3D11_QUERY_TIMESTAMP;
	qryDesc.MiscFlags = 0;
	V_RETURN(pd3dDevice->CreateQuery(&qryDesc, &g_qryTimestamp1));
	V_RETURN(pd3dDevice->CreateQuery(&qryDesc, &g_qryTimestamp2));

	qryDesc.Query = D3D11_QUERY_TIMESTAMP_DISJOINT;
	V_RETURN(pd3dDevice->CreateQuery(&qryDesc, &g_qryTimestampDisjoint));

	// example model
	V_RETURN(LoadModel(pd3dDevice, "bunny.obj"));
	SetupVoxelization();

	V_RETURN(CreateVoxelizationResources(pd3dDevice, pd3dImmediateContext));

	// full-screen quad
	V_RETURN(CreateFullscreenQuad(pd3dDevice));

    // set up the camera's view parameters
	D3DXVECTOR3 vecEye(0.0f, 0.0f, -4.0f);
	D3DXVECTOR3 vecAt (0.0f, 0.0f,  0.0f);
	g_camera.SetViewParams(&vecEye, &vecAt);

    return S_OK;
}

void CALLBACK OnDestroyDevice(void* pUserContext) {
	g_dialogResourceManager.OnD3D11DestroyDevice();
	g_settingsDlg.OnD3D11DestroyDevice();
	DXUTGetGlobalResourceCache().OnDestroyDevice();
	SAFE_DELETE(g_textHelper);

	SAFE_RELEASE(g_rastDefault);
	SAFE_RELEASE(g_rastNoCull);

	SAFE_RELEASE(g_qryTimestamp1);
	SAFE_RELEASE(g_qryTimestamp2);
	SAFE_RELEASE(g_qryTimestampDisjoint);

	SAFE_RELEASE(g_vsRenderModel);
	SAFE_RELEASE(g_psRenderModel);
	SAFE_RELEASE(g_vsVoxelize);
	SAFE_RELEASE(g_psVoxelizeSurface);
	SAFE_RELEASE(g_psVoxelizeSolid);
	SAFE_RELEASE(g_vsRenderVoxelizationRaycasting);
	SAFE_RELEASE(g_psRenderVoxelizationRaycasting);
	SAFE_RELEASE(g_csVoxelizeSolid);
	SAFE_RELEASE(g_csVoxelizeSolid_Propagate);
	SAFE_RELEASE(g_csVoxelizeSurfaceConservative);

	SAFE_RELEASE(g_ilytMesh);
	SAFE_RELEASE(g_ilytQuad);

	SAFE_RELEASE(g_cbPerFrame);
	SAFE_RELEASE(g_cbPerObject);
	SAFE_RELEASE(g_cbVoxelGrid);
	SAFE_RELEASE(g_cbModelInput);
	SAFE_RELEASE(g_cbRaycasting);

	SAFE_RELEASE(g_vbMesh);
	SAFE_RELEASE(g_ibMesh);
	SAFE_RELEASE(g_srvVbMesh);
	SAFE_RELEASE(g_srvIbMesh);

	SAFE_RELEASE(g_vbQuad);

	ReleaseVoxelizationResources();
}

//--------------------------------------------------------------------------------------------------------------------------------------------------------------

HRESULT CALLBACK OnResizedSwapChain(ID3D11Device* pd3dDevice, IDXGISwapChain* pSwapChain, const DXGI_SURFACE_DESC* pBackBufferSurfaceDesc, void* pUserContext) {
	HRESULT hr;

	V_RETURN(g_dialogResourceManager.OnD3D11ResizedSwapChain(pd3dDevice, pBackBufferSurfaceDesc));
	V_RETURN(g_settingsDlg.OnD3D11ResizedSwapChain(pd3dDevice, pBackBufferSurfaceDesc));

	float fAspectRatio = float(pBackBufferSurfaceDesc->Width) / float(pBackBufferSurfaceDesc->Height);
	g_camera.SetProjParams(D3DX_PI / 4.0f, fAspectRatio, 0.1f, 20.0f);
	g_camera.SetWindow(pBackBufferSurfaceDesc->Width, pBackBufferSurfaceDesc->Height);
	g_camera.SetButtonMasks(MOUSE_MIDDLE_BUTTON, MOUSE_WHEEL, MOUSE_LEFT_BUTTON);

	g_dlg.SetLocation(pBackBufferSurfaceDesc->Width - 200, 0);
	g_dlg.SetSize(200, 150);

	return S_OK;
}

void CALLBACK OnReleasingSwapChain(void* pUserContext) {
    g_dialogResourceManager.OnD3D11ReleasingSwapChain();
}

//--------------------------------------------------------------------------------------------------------------------------------------------------------------

void CALLBACK OnFrameMove(double fTime, float fElapsedTime, void* pUserContext) {
    g_camera.FrameMove(fElapsedTime);
	D3DXMATRIX matView(*g_camera.GetViewMatrix());
	D3DXMATRIX matProj(*g_camera.GetProjMatrix());

	// convert to matrices for right-handed coordinate system ...
	matView(2, 0) = -matView(2, 0);
	matView(2, 1) = -matView(2, 1);
	matView(0, 2) = -matView(0, 2);
	matView(1, 2) = -matView(1, 2);
	matView(3, 2) = -matView(3, 2);

	matProj(2, 2) = -matProj(2, 2);
	matProj(2, 3) = -matProj(2, 3);

	// ... and column-vector convention
	D3DXMatrixTranspose(&g_matView, &matView);
	D3DXMatrixTranspose(&g_matProj, &matProj);

	// place light above camera
	g_wLightPos = *g_camera.GetEyePt();
	g_wLightPos.z = -g_wLightPos.z;
	g_wLightPos += D3DXVECTOR3(g_matView(1, 0), g_matView(1, 1), g_matView(1, 2)) * 0.2f;
}

//--------------------------------------------------------------------------------------------------------------------------------------------------------------

HRESULT CreateVoxelizationResources(ID3D11Device* pd3dDevice, ID3D11DeviceContext* pd3dImmediateContext) {
	HRESULT hr;

	// create voxelization buffer (one bit per voxel)
	D3D11_BUFFER_DESC bufDesc;
	bufDesc.ByteWidth = g_dataSize * 4;
	bufDesc.Usage = D3D11_USAGE_DEFAULT;
	bufDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
    bufDesc.CPUAccessFlags = 0;
    bufDesc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS;
    bufDesc.StructureByteStride = 0;
	V_RETURN(pd3dDevice->CreateBuffer(&bufDesc, NULL, &g_bufVoxelization));

	// create unordered access view for voxelization buffer
	D3D11_UNORDERED_ACCESS_VIEW_DESC uavDesc;
	uavDesc.Format = DXGI_FORMAT_R32_TYPELESS;
	uavDesc.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
	uavDesc.Buffer.FirstElement = 0;
	uavDesc.Buffer.NumElements = bufDesc.ByteWidth / 4;
	uavDesc.Buffer.Flags = D3D11_BUFFER_UAV_FLAG_RAW;
	V_RETURN(pd3dDevice->CreateUnorderedAccessView(g_bufVoxelization, &uavDesc, &g_uavVoxelization));

	// create shader resource view for voxelization buffer
	D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc;
	srvDesc.Format = DXGI_FORMAT_R32_UINT;
	srvDesc.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
	srvDesc.Buffer.ElementOffset = 0;
	srvDesc.Buffer.ElementWidth = bufDesc.ByteWidth / 4;
	V_RETURN(pd3dDevice->CreateShaderResourceView(g_bufVoxelization, &srvDesc, &g_srvVoxelization));

	// create texture to serve as render target for triggering fragment generation during voxelization
	D3D11_TEXTURE2D_DESC texDesc;
	texDesc.Width = g_gridSizeX;
	texDesc.Height = g_gridSizeY;
    texDesc.MipLevels = 1;
    texDesc.ArraySize = 1;
	texDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
	texDesc.SampleDesc.Count = 1;
	texDesc.SampleDesc.Quality = 0;
	texDesc.Usage = D3D11_USAGE_DEFAULT;
	texDesc.BindFlags = D3D11_BIND_RENDER_TARGET;
    texDesc.CPUAccessFlags = 0;
    texDesc.MiscFlags = 0;
	V_RETURN(pd3dDevice->CreateTexture2D(&texDesc, NULL, &g_texVoxelizationDummy));

	// create render target view
	D3D11_RENDER_TARGET_VIEW_DESC rtvDesc;
	rtvDesc.Format = texDesc.Format;
	rtvDesc.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2D;
	rtvDesc.Texture2D.MipSlice = 0;
	V_RETURN(pd3dDevice->CreateRenderTargetView(g_texVoxelizationDummy, &rtvDesc, &g_rtvVoxelizationDummy));

	const float colClear[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
	pd3dImmediateContext->ClearRenderTargetView(g_rtvVoxelizationDummy, colClear);

	g_validVoxelization = false;

	return S_OK;
}

void ReleaseVoxelizationResources() {
	SAFE_RELEASE(g_bufVoxelization);
	SAFE_RELEASE(g_uavVoxelization);
	SAFE_RELEASE(g_srvVoxelization);

	SAFE_RELEASE(g_texVoxelizationDummy);
	SAFE_RELEASE(g_rtvVoxelizationDummy);
}

void SetupVoxelization() {
	g_strideX = (g_gridSizeZ + 31) / 32;
	g_strideY = g_strideX * g_gridSizeX;
	g_dataSize = g_strideY * g_gridSizeY;

	D3DXVECTOR3 extent = g_aabbModel[1] - g_aabbModel[0];
	extent.x *= (float(g_gridSizeX) + 2.0f) / float(g_gridSizeX);
	extent.y *= (float(g_gridSizeY) + 2.0f) / float(g_gridSizeY);
	extent.z *= (float(g_gridSizeZ) + 2.0f) / float(g_gridSizeZ);
	if(g_useCubeVoxels)
		extent.x = extent.y = extent.z = max(extent.x, max(extent.y, extent.z));

	D3DXVECTOR3 center = 0.5f * (g_aabbModel[0] + g_aabbModel[1]);
	g_voxelSpace[0] = center - 0.5f * extent;
	g_voxelSpace[1] = g_voxelSpace[0] + extent;

	D3DXMATRIX mat1, mat2;

	D3DXMatrixTranslation(&mat1, -g_voxelSpace[0].x, -g_voxelSpace[0].y, -g_voxelSpace[0].z);
	D3DXMatrixScaling(&mat2, float(g_gridSizeX) / extent.x, float(g_gridSizeY) / extent.y, float(g_gridSizeZ) / extent.z);
	D3DXMatrixMultiplyTranspose(&g_matWorldToVoxel, &mat1, &mat2);

	D3DXMatrixTranslation(&mat1, -center.x, -center.y, -g_voxelSpace[0].z);
	D3DXMatrixScaling(&mat2, 2.0f / extent.x, 2.0f / extent.y, 1.0f / extent.z);
	D3DXMatrixMultiplyTranspose(&g_matWorldToVoxelProj, &mat1, &mat2);
}

void VoxelizeViaRendering(ID3D11DeviceContext* pd3dImmediateContext) {
	const UINT clearValue[4] = { 0u, 0u, 0u, 0u };
	pd3dImmediateContext->ClearUnorderedAccessViewUint(g_uavVoxelization, clearValue);

	// rasterize into dummy render target of size gridSizeX x gridSizeY but write into voxelization buffer
	pd3dImmediateContext->OMSetRenderTargetsAndUnorderedAccessViews(1, &g_rtvVoxelizationDummy, NULL, 1, 1, &g_uavVoxelization, NULL);

	D3D11_VIEWPORT viewport;
    viewport.TopLeftX = 0.0f;
    viewport.TopLeftY = 0.0f;
    viewport.Width = float(g_gridSizeX);
    viewport.Height = float(g_gridSizeY);
    viewport.MinDepth = D3D11_MIN_DEPTH;
    viewport.MaxDepth = D3D11_MAX_DEPTH;
	pd3dImmediateContext->RSSetViewports(1, &viewport);

	pd3dImmediateContext->RSSetState(g_rastNoCull);

	pd3dImmediateContext->IASetInputLayout(g_ilytMesh);
	pd3dImmediateContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
	pd3dImmediateContext->IASetIndexBuffer(g_ibMesh, DXGI_FORMAT_R32_UINT, 0);
	UINT strides[1] = { g_bytesPerMeshVertex };
	UINT offsets[1] = { 0 };
	pd3dImmediateContext->IASetVertexBuffers(0, 1, &g_vbMesh, strides, offsets);

	D3DXMATRIX matModelToWorld;
	D3DXMatrixIdentity(&matModelToWorld);

	D3D11_MAPPED_SUBRESOURCE mappedBuf;
	pd3dImmediateContext->Map(g_cbVoxelGrid, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedBuf);
	CB_VoxelGrid* cbVoxelGrid = reinterpret_cast<CB_VoxelGrid*>(mappedBuf.pData);
	cbVoxelGrid->m_matModelToProj = g_matWorldToVoxelProj * matModelToWorld;
	cbVoxelGrid->m_matModelToVoxel = g_matWorldToVoxel * matModelToWorld;
	cbVoxelGrid->m_stride[0] = g_strideX * 4;
	cbVoxelGrid->m_stride[1] = g_strideY * 4;
	cbVoxelGrid->m_gridSize[0] = g_gridSizeX;
	cbVoxelGrid->m_gridSize[1] = g_gridSizeY;
	cbVoxelGrid->m_gridSize[2] = g_gridSizeZ;
	pd3dImmediateContext->Unmap(g_cbVoxelGrid, 0);

	ID3D11Buffer* constantBuffers[1] = {
		g_cbVoxelGrid,
	};

	pd3dImmediateContext->VSSetConstantBuffers(0, ARRAYSIZE(constantBuffers), constantBuffers);
	pd3dImmediateContext->PSSetConstantBuffers(0, ARRAYSIZE(constantBuffers), constantBuffers);

	pd3dImmediateContext->VSSetShader(g_vsVoxelize, NULL, 0);
	pd3dImmediateContext->PSSetShader(g_voxelizationMethod == VOXELIZATION_SURFACE_PS ? g_psVoxelizeSurface : g_psVoxelizeSolid, NULL, 0);

	pd3dImmediateContext->DrawIndexed(g_numMeshIndices, 0, 0);

    viewport.Width = float(DXUTGetDXGIBackBufferSurfaceDesc()->Width);
    viewport.Height = float(DXUTGetDXGIBackBufferSurfaceDesc()->Height);
	pd3dImmediateContext->RSSetViewports(1, &viewport);

	g_validVoxelization = true;
}

void VoxelizeViaCompute(ID3D11DeviceContext* pd3dImmediateContext) {
	const UINT clearValue[4] = { 0u, 0u, 0u, 0u };
	pd3dImmediateContext->ClearUnorderedAccessViewUint(g_uavVoxelization, clearValue);

	D3DXMATRIX matModelToWorld;
	D3DXMatrixIdentity(&matModelToWorld);

	D3D11_MAPPED_SUBRESOURCE mappedBuf;
	pd3dImmediateContext->Map(g_cbVoxelGrid, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedBuf);
	CB_VoxelGrid* cbVoxelGrid = reinterpret_cast<CB_VoxelGrid*>(mappedBuf.pData);
	cbVoxelGrid->m_matModelToVoxel = g_matWorldToVoxel * matModelToWorld;
	cbVoxelGrid->m_stride[0] = g_strideX * 4;
	cbVoxelGrid->m_stride[1] = g_strideY * 4;
	cbVoxelGrid->m_gridSize[0] = g_gridSizeX;
	cbVoxelGrid->m_gridSize[1] = g_gridSizeY;
	cbVoxelGrid->m_gridSize[2] = g_gridSizeZ;
	pd3dImmediateContext->Unmap(g_cbVoxelGrid, 0);

	const UINT numTriangles = g_numMeshIndices / 3;

	pd3dImmediateContext->Map(g_cbModelInput, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedBuf);
	CB_ModelInput* cbModelInput = reinterpret_cast<CB_ModelInput*>(mappedBuf.pData);
	cbModelInput->m_numModelTriangles = numTriangles;
	cbModelInput->m_vertexFloatStride = g_bytesPerMeshVertex / sizeof(float);
	pd3dImmediateContext->Unmap(g_cbModelInput, 0);

	ID3D11Buffer* constantBuffers[2] = {
		g_cbVoxelGrid,
		g_cbModelInput
	};

	pd3dImmediateContext->CSSetConstantBuffers(0, ARRAYSIZE(constantBuffers), constantBuffers);

	ID3D11ShaderResourceView* shaderResources[2] = {
		g_srvVbMesh,
		g_srvIbMesh,
	};

	pd3dImmediateContext->CSSetShaderResources(0, 2, shaderResources);

	pd3dImmediateContext->OMSetRenderTargetsAndUnorderedAccessViews(0, NULL, NULL, 0, 0, NULL, NULL);
	ID3D11UnorderedAccessView* uavs[2] = { NULL, g_uavVoxelization };
	pd3dImmediateContext->CSSetUnorderedAccessViews(0, 2, uavs, NULL);

	pd3dImmediateContext->CSSetShader(g_voxelizationMethod == VOXELIZATION_SOLID_COMPUTE ? g_csVoxelizeSolid : g_csVoxelizeSurfaceConservative, NULL, 0);

	{
		const UINT numThreads = numTriangles;
		const UINT threadsPerBlock = 256;

		pd3dImmediateContext->Dispatch(256, (numThreads + (threadsPerBlock * 256 - 1)) / (threadsPerBlock * 256), 1);
	}

	if(g_voxelizationMethod == VOXELIZATION_SOLID_COMPUTE) {
		pd3dImmediateContext->CSSetShader(g_csVoxelizeSolid_Propagate, NULL, 0);

		const UINT numThreads = g_strideY;
		const UINT threadsPerBlock = 256;

		pd3dImmediateContext->Dispatch(256, (numThreads + (threadsPerBlock * 256 - 1)) / (threadsPerBlock * 256), 1);
	}

	ID3D11UnorderedAccessView* uavsReset[2] = { NULL, NULL };
	pd3dImmediateContext->CSSetUnorderedAccessViews(0, 2, uavsReset, NULL);

	g_validVoxelization = true;
}

//--------------------------------------------------------------------------------------------------------------------------------------------------------------

void RenderModel(ID3D11DeviceContext* pd3dImmediateContext) {
	pd3dImmediateContext->IASetInputLayout(g_ilytMesh);
	pd3dImmediateContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
	pd3dImmediateContext->IASetIndexBuffer(g_ibMesh, DXGI_FORMAT_R32_UINT, 0);
	UINT strides[1] = { g_bytesPerMeshVertex };
	UINT offsets[1] = { 0 };
	pd3dImmediateContext->IASetVertexBuffers(0, 1, &g_vbMesh, strides, offsets);

	D3D11_MAPPED_SUBRESOURCE mappedBuf;
	pd3dImmediateContext->Map(g_cbPerFrame, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedBuf);
	CB_PerFrame* cbPerFrame = reinterpret_cast<CB_PerFrame*>(mappedBuf.pData);
	cbPerFrame->m_wLightPos = g_wLightPos;
	pd3dImmediateContext->Unmap(g_cbPerFrame, 0);

	D3DXMATRIX matModelToWorld;
	D3DXMatrixIdentity(&matModelToWorld);

	D3DXMATRIX matModelToProj = g_matProj * g_matView * matModelToWorld;

	pd3dImmediateContext->Map(g_cbPerObject, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedBuf);
	CB_PerObject* cbPerObject = reinterpret_cast<CB_PerObject*>(mappedBuf.pData);
	cbPerObject->m_matModelToWorld = matModelToWorld;
	cbPerObject->m_matModelToProj = matModelToProj;
	cbPerObject->m_colDiffuse = D3DXVECTOR3(0.76f, 0.64f, 0.37f);
	pd3dImmediateContext->Unmap(g_cbPerObject, 0);

	ID3D11Buffer* constantBuffers[2] = {
		g_cbPerFrame,
		g_cbPerObject
	};

	pd3dImmediateContext->VSSetConstantBuffers(0, ARRAYSIZE(constantBuffers), constantBuffers);
	pd3dImmediateContext->PSSetConstantBuffers(0, ARRAYSIZE(constantBuffers), constantBuffers);

	pd3dImmediateContext->VSSetShader(g_vsRenderModel, NULL, 0);
	pd3dImmediateContext->PSSetShader(g_psRenderModel, NULL, 0);
	pd3dImmediateContext->DrawIndexed(g_numMeshIndices, 0, 0);
}

void RenderVoxelizationViaRaycasting(ID3D11DeviceContext* pd3dImmediateContext) {
	if(!g_validVoxelization)
		return;

	pd3dImmediateContext->IASetInputLayout(g_ilytQuad);
	pd3dImmediateContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
	UINT strides[1] = { UINT(g_bytesPerQuadVertex) };
	UINT offsets[1] = { 0 };
	pd3dImmediateContext->IASetVertexBuffers(0, 1, &g_vbQuad, strides, offsets);

	D3DXMATRIX mat1, mat2, mat12, matInverse;
	D3DXMatrixTranslation(&mat1, -0.5f, -0.5f, -1.0f);
	D3DXMatrixScaling(&mat2, 2.0f / g_matProj(0, 0), -2.0f / g_matProj(1, 1), 1.0f);
	D3DXMatrixMultiplyTranspose(&mat12, &mat1, &mat2);
	D3DXMatrixInverse(&matInverse, NULL, &g_matView);
	D3DXMATRIX matQuadToWorld = matInverse * mat12;

	D3DXMATRIX matQuadToVoxel = g_matWorldToVoxel * matQuadToWorld;

	D3DXMatrixTranslation(&mat1, 1.0f, 1.0f, 0.0f);
	D3DXMatrixScaling(&mat2, float(DXUTGetDXGIBackBufferSurfaceDesc()->Width) * 0.5f, float(DXUTGetDXGIBackBufferSurfaceDesc()->Height) * 0.5f, 1.0f);
	D3DXMatrixMultiplyTranspose(&mat12, &mat1, &mat2);
	D3DXMatrixInverse(&matInverse, NULL, &g_matWorldToVoxel);
	D3DXMATRIX matVoxelToScreen = mat12 * g_matProj * g_matView * matInverse;

	D3DXMATRIX matT;

	D3DXVECTOR3 cameraPos = *g_camera.GetEyePt();
	cameraPos.z = -cameraPos.z;
	D3DXVECTOR3 rayOrigin;
	D3DXVec3TransformCoord(&rayOrigin, &cameraPos, D3DXMatrixTranspose(&matT, &g_matWorldToVoxel));
	D3DXVECTOR3 voxLightPos;
	D3DXVec3TransformCoord(&voxLightPos, &g_wLightPos, D3DXMatrixTranspose(&matT, &g_matWorldToVoxel));

	D3D11_MAPPED_SUBRESOURCE mappedBuf;
	pd3dImmediateContext->Map(g_cbRaycasting, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedBuf);
	CB_Raycasting* cbRaycasting = reinterpret_cast<CB_Raycasting*>(mappedBuf.pData);
	cbRaycasting->m_matQuadToVoxel = matQuadToVoxel;
	cbRaycasting->m_matVoxelToScreen = matVoxelToScreen;
	cbRaycasting->m_rayOrigin = rayOrigin;
	cbRaycasting->m_voxLightPos = voxLightPos;
	cbRaycasting->m_stride[0] = g_strideX;
	cbRaycasting->m_stride[1] = g_strideY;
	cbRaycasting->m_gridSize[0] = g_gridSizeX;
	cbRaycasting->m_gridSize[1] = g_gridSizeY;
	cbRaycasting->m_gridSize[2] = g_gridSizeZ;
	cbRaycasting->m_showLines = g_showVoxelBorderLines;
	pd3dImmediateContext->Unmap(g_cbRaycasting, 0);

	ID3D11Buffer* constantBuffers[1] = {
		g_cbRaycasting
	};

	pd3dImmediateContext->VSSetConstantBuffers(0, ARRAYSIZE(constantBuffers), constantBuffers);
	pd3dImmediateContext->PSSetConstantBuffers(0, ARRAYSIZE(constantBuffers), constantBuffers);

	pd3dImmediateContext->PSSetShaderResources(0, 1, &g_srvVoxelization);

	pd3dImmediateContext->VSSetShader(g_vsRenderVoxelizationRaycasting, NULL, 0);
	pd3dImmediateContext->PSSetShader(g_psRenderVoxelizationRaycasting, NULL, 0);
	pd3dImmediateContext->Draw(4, 0);
}

void RenderText() {
	g_textHelper->Begin();

	g_textHelper->SetInsertionPos(5, 5);
	g_textHelper->SetForegroundColor(D3DXCOLOR(1.0f, 1.0f, 1.0f, 1.0f));
	g_textHelper->DrawTextLine(DXUTGetFrameStats(DXUTIsVsyncEnabled()));

	g_textHelper->SetInsertionPos(5, 25);
	g_textHelper->SetForegroundColor(D3DXCOLOR(0.8f, 0.8f, 0.8f, 1.0f));
	g_textHelper->DrawFormattedTextLine(L"Grid size: %dx%dx%d", g_gridSizeX, g_gridSizeY, g_gridSizeZ);

	if(g_voxelize) {
		const char* methodName = NULL;
		switch(g_voxelizationMethod) {
			case VOXELIZATION_SOLID_PS:
				methodName = "Solid (PS)";
				break;
			case VOXELIZATION_SOLID_COMPUTE:
				methodName = "Solid (compute)";
				break;
			case VOXELIZATION_SURFACE_PS:
				methodName = "Surface (PS)";
				break;
			case VOXELIZATION_SURFACE_CONSERVATIVE_COMPUTE:
				methodName = "Conservative surface (compute)";
				break;
		}
		g_textHelper->DrawFormattedTextLine(L"Method: %S", methodName);

		if(g_secsVoxelization > 0.0) {
			g_textHelper->DrawFormattedTextLine(L"Time: %0.2f ms", g_secsVoxelization * 1000.0);
		}
	}

	g_textHelper->SetInsertionPos(5, DXUTGetDXGIBackBufferSurfaceDesc()->Height - 65);
	g_textHelper->SetForegroundColor(D3DXCOLOR(0.8f, 0.8f, 0.8f, 1.0f));
	g_textHelper->DrawTextLine(L"SPACE - Turn voxelization on/off");
	g_textHelper->DrawTextLine(L"TAB - Display model or voxelization");
	g_textHelper->DrawTextLine(L"1/2/3/4 - Select voxelization method");

	g_textHelper->End();
}

//--------------------------------------------------------------------------------------------------------------------------------------------------------------

void CALLBACK OnFrameRender(ID3D11Device* pd3dDevice, ID3D11DeviceContext* pd3dImmediateContext, double fTime, float fElapsedTime, void* pUserContext) {
	if(g_settingsDlg.IsActive()) {
		pd3dImmediateContext->RSSetState(NULL);
		g_settingsDlg.OnRender(fElapsedTime);
		return;
	}

	// perform voxelization
	if(g_voxelize) {
		pd3dImmediateContext->Begin(g_qryTimestampDisjoint);
		pd3dImmediateContext->End(g_qryTimestamp1);
		switch(g_voxelizationMethod) {
			case VOXELIZATION_SOLID_PS:
			case VOXELIZATION_SURFACE_PS:
				VoxelizeViaRendering(pd3dImmediateContext);
				break;
			case VOXELIZATION_SOLID_COMPUTE:
			case VOXELIZATION_SURFACE_CONSERVATIVE_COMPUTE:
				VoxelizeViaCompute(pd3dImmediateContext);
				break;
		}
		pd3dImmediateContext->End(g_qryTimestamp2);
		pd3dImmediateContext->End(g_qryTimestampDisjoint);
	}

	// render scene
	ID3D11RenderTargetView* rtv = DXUTGetD3D11RenderTargetView();
	ID3D11DepthStencilView* dsv = DXUTGetD3D11DepthStencilView();

	const float colClear[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
	pd3dImmediateContext->ClearRenderTargetView(rtv, colClear);
	pd3dImmediateContext->ClearDepthStencilView(dsv, D3D11_CLEAR_DEPTH, 1.0f, 0);

	pd3dImmediateContext->OMSetRenderTargets(1, &rtv, dsv);
	pd3dImmediateContext->RSSetState(g_rastDefault);

	if(g_displayVoxelization && g_validVoxelization)
		RenderVoxelizationViaRaycasting(pd3dImmediateContext);
	else
		RenderModel(pd3dImmediateContext);

	// determine voxelization performance
	if(g_voxelize) {
		UINT64 timestamp1;
		UINT64 timestamp2;
		D3D11_QUERY_DATA_TIMESTAMP_DISJOINT timestampDisjoint;
		while(pd3dImmediateContext->GetData(g_qryTimestamp1, &timestamp1, sizeof(timestamp1), 0) != S_OK);
		while(pd3dImmediateContext->GetData(g_qryTimestamp2, &timestamp2, sizeof(timestamp2), 0) != S_OK);
		while(pd3dImmediateContext->GetData(g_qryTimestampDisjoint, &timestampDisjoint, sizeof(timestampDisjoint), 0) != S_OK);

		if(timestampDisjoint.Disjoint)
			g_secsVoxelization = 0.0;
		else
			g_secsVoxelization = double(timestamp2 - timestamp1) / double(timestampDisjoint.Frequency);
	}

	// render HUD
	DXUT_BeginPerfEvent(DXUT_PERFEVENTCOLOR, L"HUD / Stats");
	g_dlg.OnRender(fElapsedTime);
	RenderText();
	DXUT_EndPerfEvent();
}

//--------------------------------------------------------------------------------------------------------------------------------------------------------------

LRESULT CALLBACK MsgProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam, bool* pbNoFurtherProcessing, void* pUserContext) {
	// pass messages to dialog resource manager calls so GUI state is updated correctly
	*pbNoFurtherProcessing = g_dialogResourceManager.MsgProc(hWnd, uMsg, wParam, lParam);
	if(*pbNoFurtherProcessing)
		return 0;

	// pass messages to settings dialog if its active
	if(g_settingsDlg.IsActive()) {
		g_settingsDlg.MsgProc(hWnd, uMsg, wParam, lParam);
		return 0;
	}

	// give the dialog a chance to handle the message first
	*pbNoFurtherProcessing = g_dlg.MsgProc(hWnd, uMsg, wParam, lParam);
	if(*pbNoFurtherProcessing)
		return 0;

	// pass all remaining windows messages to camera so it can respond to user input
	g_camera.HandleMessages(hWnd, uMsg, wParam, lParam);

    return 0;
}

void CALLBACK OnKeyboard(UINT nChar, bool bKeyDown, bool bAltDown, void* pUserContext) {
	if(bKeyDown) switch(nChar) {
		case VK_SPACE:
			g_voxelize = !g_voxelize;
			break;

		case VK_TAB:
			g_displayVoxelization = !g_displayVoxelization;
			break;

		case '1':
			g_voxelizationMethod = VOXELIZATION_SURFACE_PS;
			break;

		case '2':
			g_voxelizationMethod = VOXELIZATION_SOLID_PS;
			break;

		case '3':
			g_voxelizationMethod = VOXELIZATION_SURFACE_CONSERVATIVE_COMPUTE;
			break;

		case '4':
			g_voxelizationMethod = VOXELIZATION_SOLID_COMPUTE;
			break;

		case 'L':
			g_showVoxelBorderLines = !g_showVoxelBorderLines;
			break;
	}
}

void CALLBACK OnGUIEvent(UINT nEvent, int nControlID, CDXUTControl* pControl, void* pUserContext) {
}

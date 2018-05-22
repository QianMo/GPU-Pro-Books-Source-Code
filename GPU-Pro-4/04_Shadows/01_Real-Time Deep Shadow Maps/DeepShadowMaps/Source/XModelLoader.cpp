#include <d3d9.h>
#include <d3dx9.h>
#include "ModelLoader.h"
#include "MemoryLeakTracker.h"
#include <fstream>

struct VertexData
{
	CoreVector3 pos;
	CoreVector3 normal;
	float u, v;
};

CoreResult LoadXModel(Core *core, std::wstring &filename, Model ***models, int *numModels)
{
    // Create a dummy d3d9 device for loading the .x file
    IDirect3D9* d3d9 = Direct3DCreate9(D3D_SDK_VERSION);

    if(d3d9 == NULL)
        return CORE_MISC_ERROR;

    D3DPRESENT_PARAMETERS pp;
    pp.BackBufferWidth = 32;
    pp.BackBufferHeight = 32;
    pp.BackBufferFormat = D3DFMT_X8R8G8B8;
    pp.BackBufferCount = 1;
    pp.MultiSampleType = D3DMULTISAMPLE_NONE;
    pp.MultiSampleQuality = 0;
    pp.SwapEffect = D3DSWAPEFFECT_DISCARD;
    pp.hDeviceWindow = GetShellWindow();
    pp.Windowed = true;
    pp.Flags = 0;
    pp.FullScreen_RefreshRateInHz = 0;
    pp.PresentationInterval = D3DPRESENT_INTERVAL_DEFAULT;
    pp.EnableAutoDepthStencil = false;

	IDirect3DDevice9* d3dDev9 = NULL;
    HRESULT hr = d3d9->CreateDevice(D3DADAPTER_DEFAULT, D3DDEVTYPE_NULLREF, NULL, D3DCREATE_SOFTWARE_VERTEXPROCESSING, &pp, &d3dDev9);
    if(FAILED(hr))
	{
		d3d9->Release();
		return CORE_MISC_ERROR;
	}

	ID3DXMesh *d3dxMeshTemp = NULL;
	ID3DXBuffer *d3dxMaterialsBuffer = NULL;
	DWORD numMaterials;

	hr = D3DXLoadMeshFromX(filename.c_str(), 0, d3dDev9, NULL, &d3dxMaterialsBuffer, NULL, &numMaterials, &d3dxMeshTemp);
	if(FAILED(hr))
	{
		d3dDev9->Release();
		d3d9->Release();
		return CORE_MISC_ERROR;
	}

	D3DVERTEXELEMENT9 wantedFormat[] = 
    {
        {0, 0,  D3DDECLTYPE_FLOAT3,   D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_POSITION, 0},
        {0, 12, D3DDECLTYPE_FLOAT3,   D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_NORMAL,   0},
        {0, 24, D3DDECLTYPE_FLOAT2,   D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_TEXCOORD, 0},
        D3DDECL_END()
    };

	ID3DXMesh *d3dxMesh = NULL;
	hr = d3dxMeshTemp->CloneMesh(D3DXMESH_32BIT, wantedFormat, d3dDev9, &d3dxMesh);
	
	d3dxMeshTemp->Release();
	if(FAILED(hr))
	{
		d3dDev9->Release();
		d3d9->Release();
		return CORE_MISC_ERROR;
	}

	DWORD *adjBuffer = new DWORD[3 * d3dxMesh->GetNumFaces()];
	d3dxMesh->GenerateAdjacency(0.001f, adjBuffer);
	d3dxMesh->OptimizeInplace(D3DXMESHOPT_ATTRSORT, adjBuffer, NULL, NULL, NULL);
	delete adjBuffer;

	DWORD numAttributes;
	d3dxMesh->GetAttributeTable(NULL, &numAttributes);
	D3DXATTRIBUTERANGE *attributes = new D3DXATTRIBUTERANGE[numAttributes];
	d3dxMesh->GetAttributeTable(attributes, &numAttributes);
	
	*numModels = (int)numAttributes;
	*models = new Model *[*numModels];

	DWORD *originalIndices;
    d3dxMesh->LockIndexBuffer(0, (void **)&originalIndices);

	VertexData *originalVertices;
	d3dxMesh->LockVertexBuffer(0, (void **)&originalVertices);

	D3D11_INPUT_ELEMENT_DESC inputLayout[] =
    {
        { "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, D3D11_APPEND_ALIGNED_ELEMENT, D3D11_INPUT_PER_VERTEX_DATA, 0 },
		{ "NORMAL",   0, DXGI_FORMAT_R32G32B32_FLOAT, 0, D3D11_APPEND_ALIGNED_ELEMENT, D3D11_INPUT_PER_VERTEX_DATA, 0 },
		{ "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, D3D11_APPEND_ALIGNED_ELEMENT, D3D11_INPUT_PER_VERTEX_DATA, 0 },
    };

	D3DXMATERIAL *d3dxMaterials = (D3DXMATERIAL *)d3dxMaterialsBuffer->GetBufferPointer();

	for(DWORD attribute = 0; attribute < numAttributes; attribute++)
	{
		DWORD *indexData = new DWORD[attributes[attribute].FaceCount * 3];
		VertexData *vertexData = originalVertices + attributes[attribute].VertexStart;

		for(DWORD index = 0; index < attributes[attribute].FaceCount * 3; index++)
			indexData[index] = originalIndices[attributes[attribute].FaceStart * 3 + index] - attributes[attribute].VertexStart;

		CoreColor diffuseColor(d3dxMaterials[attribute].MatD3D.Diffuse.r, d3dxMaterials[attribute].MatD3D.Diffuse.g, d3dxMaterials[attribute].MatD3D.Diffuse.b, d3dxMaterials[attribute].MatD3D.Diffuse.a);
		CoreTexture2D *texture = NULL;
		if(d3dxMaterials[attribute].pTextureFilename != NULL)
		{
			std::string fullFilename(d3dxMaterials[attribute].pTextureFilename);
			fullFilename = "Models\\" + fullFilename;
			std::ifstream texFile(fullFilename.c_str(), std::ios::in | std::ios::binary);

			core->CreateTexture2D(texFile, 1, 0, 0, D3D11_USAGE_DEFAULT, D3D11_BIND_SHADER_RESOURCE, 1, 0, false, &texture);

			texFile.close();
		}
		
		(*models)[attribute] = new Model();
		(*models)[attribute]->Init(core, filename, indexData, attributes[attribute].FaceCount * 3, DXGI_FORMAT_R32_UINT, vertexData, sizeof(VertexData), attributes[attribute].VertexCount, inputLayout, 3, D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST, diffuseColor, texture);
		delete indexData;
	}

	d3dxMaterialsBuffer->Release();
	delete attributes;
	d3dxMesh->UnlockVertexBuffer();
	d3dxMesh->UnlockIndexBuffer();
	d3dxMesh->Release();
	d3dDev9->Release();
	d3d9->Release();
	
	
	return CORE_OK;
}
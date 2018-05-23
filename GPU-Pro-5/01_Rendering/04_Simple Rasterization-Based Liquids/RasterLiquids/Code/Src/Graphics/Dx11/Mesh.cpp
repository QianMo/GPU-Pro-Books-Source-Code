
#include <Graphics/Dx11/Mesh.hpp>
#include <Graphics/MeshImport.hpp>


#include <Common/Common.hpp>
#include <string>

#include <Graphics/VertexTraits.hpp>

///<
void Mesh::Draw(ID3D11DeviceContext* _pImmediateContext, const int32 _iSize)
{
	if (m_pConstants)
	{
		_pImmediateContext->UpdateSubresource(m_pConstants, 0, NULL, &m_World, 0, 0 );	
		int32 WorldReg=1;
		_pImmediateContext->VSSetConstantBuffers(WorldReg, 1, &m_pConstants);
		_pImmediateContext->GSSetConstantBuffers(WorldReg, 1, &m_pConstants);
	}

	if (m_pTextureRV)
	{
		_pImmediateContext->PSSetShaderResources(0,1,&m_pTextureRV);
	}
	
	ASSERT(m_Stride>0, "Vertex Stride is 0");
	uint32 offset = 0;
	_pImmediateContext->IASetVertexBuffers(0, 1, &m_pVertexBuffer, &m_Stride, &offset );	

	if (_iSize==0)
		_pImmediateContext->Draw(NumVertices(), 0);
	else
		_pImmediateContext->Draw(_iSize*2*3, 0);
}


///<
struct MeshCreator
{
	void* m_pData;
	int32 m_byteSize;

	MeshCreator() { memset(this, 0, sizeof(MeshCreator)); }

	~MeshCreator(){ ASSERT(m_pData==NULL, "Not freed!!"); }

	template<class T>
	void GenCreate(const MeshImport* _pMeshData)
	{
		int32 iNumVertices = _pMeshData->m_NumVertices;
		m_pData = M::CreateVertexArray<T>(_pMeshData->m_pVertices, iNumVertices);
		m_byteSize = sizeof(T);
	}

	///<
	void Create(const MeshImport* _pMeshData)
	{
		if (_pMeshData->m_bHasNormals && _pMeshData->m_bHasUVs)
		{
			GenCreate<M::NormalUVVertex>(_pMeshData);
		}
		else if (_pMeshData->m_bHasNormals && _pMeshData->m_bHasColors)
		{
				GenCreate<M::NormalColorVertex>(_pMeshData);
		}
		else if (_pMeshData->m_bHasNormals)
		{
			GenCreate<M::NormalVertex>(_pMeshData);
		}
		else if(_pMeshData->m_bHasUVs)
		{
			GenCreate<M::UVVertex>(_pMeshData);
		}
		else if (_pMeshData->m_bHasColors)
		{
			GenCreate<M::ColorVertex>(_pMeshData);
		}
		else
		{
			GenCreate<M::Vertex>(_pMeshData);
		}	
	}

	///<
	void Destroy()
	{
		free(m_pData);
		m_pData=NULL;
	}

	
};

///<
void Mesh::UpdateData(ID3D11DeviceContext* _pImmediateContext, const MeshImport* _pMeshData)
{
	MeshCreator mc;
	mc.Create(_pMeshData);
	_pImmediateContext->UpdateSubresource(m_pVertexBuffer, 0, NULL, mc.m_pData, 0, 0);

	mc.Destroy();
}

///<
void Mesh::Create(ID3D11Device* _pDevice, const MeshImport* _pMeshData)
{
	m_World = Matrix4f::Identity();

	m_NumVertices = _pMeshData->m_NumVertices;
	MeshCreator mc;
	mc.Create(_pMeshData);
	
	m_Stride = mc.m_byteSize;

	{	
		///< 
		D3D11_BUFFER_DESC bd;
		memset(&bd, 0, sizeof(D3D11_BUFFER_DESC));
		bd.Usage			= D3D11_USAGE_DEFAULT;
		bd.ByteWidth		= m_Stride*_pMeshData->m_NumVertices;
		bd.CPUAccessFlags	= 0;

		bd.BindFlags		= D3D11_BIND_VERTEX_BUFFER;
		if (_pMeshData->m_bGeometry)
			bd.BindFlags |= D3D10_BIND_STREAM_OUTPUT | D3D10_BIND_SHADER_RESOURCE;	

		D3D11_SUBRESOURCE_DATA InitData;
		memset(&InitData, 0, sizeof(D3D11_SUBRESOURCE_DATA));
		InitData.pSysMem = mc.m_pData;

		HRESULT hr = _pDevice->CreateBuffer(&bd, &InitData, &m_pVertexBuffer);
		ASSERT(hr==S_OK, "Failed to create Buffer");

	}
	mc.Destroy();
	
	if (_pMeshData->m_pTextureFileName)
	{
		std::string baseTexture = "..\\..\\Ressources\\Textures\\";
		std::string tFileName = std::string(_pMeshData->m_pTextureFileName);
		HRESULT hr = D3DX11CreateShaderResourceViewFromFile(_pDevice, (baseTexture+tFileName).c_str(), NULL, NULL, &m_pTextureRV, NULL );
		ASSERT(hr ==S_OK, "Failed loading texture !");
	}

	{
		D3D11_BUFFER_DESC bd;
		memset(&bd, 0, sizeof(D3D11_BUFFER_DESC));

		bd.Usage = D3D11_USAGE_DEFAULT;
		bd.ByteWidth = sizeof(Matrix4f);
		bd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
		bd.CPUAccessFlags = 0;
		HRESULT hr = _pDevice->CreateBuffer(&bd, NULL, &m_pConstants);
		ASSERT(hr==S_OK, "Constants Failed!");	
	}
}

///<
void Mesh::Release()
{	
	M::Release(&m_pVertexBuffer);
	M::Release(&m_pConstants);

	M::Release(&m_pTextureRV);
	m_NumVertices=0;
	m_Stride=0;

}

Mesh::~Mesh()
{
	Release(); 
}


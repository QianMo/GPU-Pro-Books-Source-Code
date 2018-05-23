
#include <Graphics/Dx11/Mesh.hpp>
#include <Graphics/MeshImport.hpp>

#include <Common/Common.hpp>
#include <string>

#include <Graphics/VertexTraits.hpp>

///<
void QuadColor::Create(ID3D11Device* _pDevice, Vector4f _c)
{
	M::ColorVertex vertices[] =
	{
		{ Vector3f( -1.0f, 1.0f, 0.0f ), _c },
		{ Vector3f( 1.0f, 1.0f, 0.0f), _c },
		{ Vector3f( 1.0f, -1.0f, 0.0f ), _c},

		{ Vector3f( -1.0f, 1.0f, 0.0f ),_c },
		{ Vector3f( 1.0f, -1.0f, 0.0f ), _c},
		{ Vector3f( -1.0f, -1.0f, 0.0f ), _c }
	};

	m_Stride=sizeof(M::ColorVertex);
	m_NumVertices=6;

	D3D11_BUFFER_DESC bd;
	memset(&bd, 0, sizeof(D3D11_BUFFER_DESC));
	bd.Usage = D3D11_USAGE_DEFAULT;
	bd.ByteWidth = m_Stride*m_NumVertices;
	bd.BindFlags = D3D11_BIND_VERTEX_BUFFER;
	bd.CPUAccessFlags = 0;

	D3D11_SUBRESOURCE_DATA InitData;
	memset(&InitData, 0, sizeof(D3D11_SUBRESOURCE_DATA));
	InitData.pSysMem = vertices;


	HRESULT hr = _pDevice->CreateBuffer(&bd, &InitData, &m_pVertexBuffer);
	ASSERT(hr==S_OK, "Failed to create Buffer");


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

	m_World = Matrix4f::Identity();
}

////< Different Types.
void QuadUV::Create(ID3D11Device* _pDevice, const char* _csTexture)
{
	M::UVVertex vertices[] =
	{
		{ Vector3f( -1.0f, 1.0f, 0.0f ), Vector2f(0,0) },
		{ Vector3f( 1.0f, 1.0f, 0.0f), Vector2f(1,0) },
		{ Vector3f( 1.0f, -1.0f, 0.0f ), Vector2f(1,1) },

		{ Vector3f( -1.0f, 1.0f, 0.0f ), Vector2f(0,0) },
		{ Vector3f( 1.0f, -1.0f, 0.0f ), Vector2f(1,1) },
		{ Vector3f( -1.0f, -1.0f, 0.0f ), Vector2f(0,1) }
	};

	m_Stride=sizeof(M::UVVertex);
	m_NumVertices=6;

	D3D11_BUFFER_DESC bd;
	memset(&bd, 0, sizeof(D3D11_BUFFER_DESC));
	bd.Usage = D3D11_USAGE_DEFAULT;
	bd.ByteWidth = m_Stride*m_NumVertices;
	bd.BindFlags = D3D11_BIND_VERTEX_BUFFER;
	bd.CPUAccessFlags = 0;

	D3D11_SUBRESOURCE_DATA InitData;
	memset(&InitData, 0, sizeof(D3D11_SUBRESOURCE_DATA));
	InitData.pSysMem = vertices;


	HRESULT hr = _pDevice->CreateBuffer(&bd, &InitData, &m_pVertexBuffer);
	ASSERT(hr==S_OK, "Failed to create Buffer");


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

	m_World = Matrix4f::Identity();

	if(_csTexture!=NULL)
	{
		HRESULT hr = D3DX11CreateShaderResourceViewFromFile(_pDevice, _csTexture, NULL, NULL, &m_pTextureRV, NULL );
		ASSERT(hr ==S_OK, "Failed loading texture !");
	}
}


///<
void Cube::Create(ID3D11Device* _pDevice)
{

	M::Vertex verts[] =
	{
		Vector3f(-1,1,-1),
		Vector3f(1,1,-1),
		Vector3f(1,-1,-1),
		Vector3f(-1,-1,-1),

		Vector3f(-1,1,1),
		Vector3f(1,1,1),
		Vector3f(1,-1,1),
		Vector3f(-1,-1,1)
	};

	M::Vertex vertices[]=
	{
		///< F-B
		verts[0],
		verts[1],
		verts[2],
		verts[0],
		verts[2],
		verts[3],

		verts[5],
		verts[4],
		verts[6],
		verts[6],
		verts[4],
		verts[7],

		///< Side L-R
		verts[4],
		verts[0],
		verts[7],		
		verts[7],
		verts[0],
		verts[3],

		verts[1],
		verts[5],
		verts[6],		
		verts[6],
		verts[2],
		verts[1],

		verts[0],
		verts[4],
		verts[1],
		verts[1],
		verts[4],
		verts[5],

		///< Top-Down
		verts[3],
		verts[2],
		verts[7],
		verts[7],
		verts[2],
		verts[6]
	};
	
	m_Stride=sizeof(M::Vertex);
	m_NumVertices=36;

	D3D11_BUFFER_DESC bd;
	memset(&bd, 0, sizeof(D3D11_BUFFER_DESC));
	bd.Usage = D3D11_USAGE_DEFAULT;
	bd.ByteWidth = m_Stride*m_NumVertices;
	bd.BindFlags = D3D11_BIND_VERTEX_BUFFER;
	bd.CPUAccessFlags = 0;

	D3D11_SUBRESOURCE_DATA InitData;
	memset(&InitData, 0, sizeof(D3D11_SUBRESOURCE_DATA));
	InitData.pSysMem = vertices;
	HRESULT hr = _pDevice->CreateBuffer(&bd, &InitData, &m_pVertexBuffer);
	ASSERT(hr==S_OK, "Failed to create Buffer");


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

	m_World = Matrix4f::Identity();

}

///< Cube
ID3D11Buffer* CubeUV::CreateCubeUV(ID3D11Device* _pDevice, const Vector3ui _dims)
{
	ASSERT(_dims.z()>1, "Not initialized !");

	VolumeVertex* pVertices = new VolumeVertex[6*_dims.z()];

	for (uint32 i=0; i<_dims.z();++i)
	{
		float32 z = M::SCast<float32>(i)/M::SCast<float32>(_dims.z()-1);

		pVertices[i*6 + 0]._x = Vector4f( -1.0f, 1.0f, 0.0f, 1);
		pVertices[i*6 + 0]._uv = Vector3f(0,0,z);

		pVertices[i*6 + 1]._x = Vector4f(1.0f, 1.0f, 0.0f, 1);
		pVertices[i*6 + 1]._uv = Vector3f(1,0,z);

		pVertices[i*6 + 2]._x = Vector4f(1.0f, -1.0f, 0.0f, 1);
		pVertices[i*6 + 2]._uv = Vector3f(1,1,z);

		pVertices[i*6 + 3] = pVertices[i*6 + 0];

		pVertices[i*6 + 4] = pVertices[i*6 + 2];

		pVertices[i*6 + 5]._x = Vector4f(-1.0f, -1.0f, 0.0f, 1);
		pVertices[i*6 + 5]._uv = Vector3f(0,1,z);
	}

	D3D11_BUFFER_DESC bd;
	memset(&bd, 0, sizeof(D3D11_BUFFER_DESC));
	bd.Usage = D3D11_USAGE_DEFAULT;
	bd.ByteWidth = sizeof(VolumeVertex)*6*_dims.z();
	bd.BindFlags = D3D11_BIND_VERTEX_BUFFER;
	bd.CPUAccessFlags = 0;

	D3D11_SUBRESOURCE_DATA InitData;
	memset(&InitData, 0, sizeof(D3D11_SUBRESOURCE_DATA));
	InitData.pSysMem = pVertices;

	ID3D11Buffer* pQuadBuffer=NULL;
	HRESULT hr = _pDevice->CreateBuffer(&bd, &InitData, &pQuadBuffer);
	ASSERT(hr==S_OK, "Failed to create Buffer");

	M::DeleteArray(&pVertices);

	return pQuadBuffer;
}


///< Cube UV
void CubeUV::Create(ID3D11Device* _pDevice)
{
	M::VertexUVVolume verts[] =
	{
		{Vector3f(-1,1,-1), Vector3f(0,0,0)},
		{Vector3f(1,1,-1),  Vector3f(1,0,0)},
		{Vector3f(1,-1,-1), Vector3f(1,1,0)}, 
		{Vector3f(-1,-1,-1),Vector3f(0,1,0)},

		{Vector3f(-1,1,1),  Vector3f(0,0,1)},
		{Vector3f(1,1,1),	Vector3f(1,0,1)},
		{Vector3f(1,-1,1),	Vector3f(1,1,1)},
		{Vector3f(-1,-1,1),	Vector3f(0,1,1)}
	};

	M::VertexUVVolume vertices[]=
	{
		///< F-B
		verts[0],
		verts[1],
		verts[2],
		verts[0],
		verts[2],
		verts[3],

		verts[5],
		verts[4],
		verts[6],
		verts[6],
		verts[4],
		verts[7],

		///< Side L-R
		verts[4],
		verts[0],
		verts[7],		
		verts[7],
		verts[0],
		verts[3],

		verts[1],
		verts[5],
		verts[6],		
		verts[6],
		verts[2],
		verts[1],

		verts[0],
		verts[4],
		verts[1],
		verts[1],
		verts[4],
		verts[5],

		///< Top-Down
		verts[3],
		verts[2],
		verts[7],
		verts[7],
		verts[2],
		verts[6]
	};

	m_Stride=sizeof(M::VertexUVVolume);
	m_NumVertices=36;

	D3D11_BUFFER_DESC bd;
	memset(&bd, 0, sizeof(D3D11_BUFFER_DESC));
	bd.Usage = D3D11_USAGE_DEFAULT;
	bd.ByteWidth = m_Stride*m_NumVertices;
	bd.BindFlags = D3D11_BIND_VERTEX_BUFFER;
	bd.CPUAccessFlags = 0;

	D3D11_SUBRESOURCE_DATA InitData;
	memset(&InitData, 0, sizeof(D3D11_SUBRESOURCE_DATA));
	InitData.pSysMem = vertices;
	HRESULT hr = _pDevice->CreateBuffer(&bd, &InitData, &m_pVertexBuffer);
	ASSERT(hr==S_OK, "Failed to create Buffer");


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

	m_World = Matrix4f::Identity();

}
#include <Physics\Terrain\Terrain.hpp>
#include <string>

struct TerrainVertex
{
	Vector3f _x;
	Vector2f _uv;

};

Terrain::~Terrain()
{
	M::Release(&m_pIndexBuffer);
}

///<
void Terrain::CreateDisplacementTexture(ID3D11Device* _pDevice, const char* _csDisplacement)
{
	std::string baseTexture = "..\\..\\Ressources\\Textures\\";
	std::string tFileName = std::string(_csDisplacement);
	if (tFileName.length()!=0)
	{
		HRESULT hr = D3DX11CreateShaderResourceViewFromFile(_pDevice, (baseTexture+tFileName).c_str(), NULL, NULL, &m_pDisplacementSRV, NULL );
		ASSERT(hr ==S_OK, "Failed loading texture !");
	}
}

///<
void Terrain::Create(ID3D11Device* _pDevice, Vector2i _dims)
{
	m_World = Matrix4f::Identity();

	m_iDims =_dims;

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

	{

		m_NumVertices = m_iDims.x()*m_iDims.y();

		m_Stride = sizeof(TerrainVertex);

		///< Create Vertex Buffer
		{
			D3D11_BUFFER_DESC bd;
			memset(&bd, 0, sizeof(D3D11_BUFFER_DESC));
			bd.Usage			= D3D11_USAGE_DEFAULT;
			bd.ByteWidth		= m_Stride*m_NumVertices;
			bd.BindFlags		= D3D11_BIND_VERTEX_BUFFER;
			bd.CPUAccessFlags	= 0;


			TerrainVertex* pVertices = new TerrainVertex[m_NumVertices];

			float32 fDimX = (float32)m_iDims.x();
			float32 fDimY = (float32)m_iDims.y();
			for(int32 i=0; i<m_iDims.x();++i)
			{
				for(int32 j=0; j<m_iDims.y();++j)
				{
					pVertices[i*m_iDims.y() +j]._x = Vector3f(M::SCast<float32>(j),0,M::SCast<float32>(i))-Vector3f(fDimY/2.0f, 0, fDimX/2.0f);

					float32 fi=(float32)i;
					float32 fj=(float32)j;
					pVertices[i*m_iDims.y() +j]._uv = Vector2f(fi/fDimY,fj/fDimX);
				}
			}


			D3D11_SUBRESOURCE_DATA InitData;
			memset(&InitData, 0, sizeof(D3D11_SUBRESOURCE_DATA));
			InitData.pSysMem = pVertices;

			HRESULT hr = _pDevice->CreateBuffer(&bd, &InitData, &m_pVertexBuffer);
			ASSERT(hr==S_OK, "Failed to create Buffer");

			M::DeleteArray(&pVertices);

		}

		///< Create Index Buffer
		{
			D3D11_BUFFER_DESC iBufferDesc;
			memset(&iBufferDesc, 0, sizeof(D3D11_BUFFER_DESC));

			const uint32 uiNumTris = (m_iDims.x()-1)*(m_iDims.y()-1)*2;
			m_uiNumIndices = uiNumTris * 3;
			uint32* pIndices = new uint32[m_uiNumIndices];

			for(int32 i=0; i<m_iDims.x()-1;++i)
			{
				for(int32 j=0; j<m_iDims.y()-1;++j)
				{
					uint32 uiCi = ((i*(m_iDims.x()-1))+j)*6;

					pIndices[uiCi + 0] = i*m_iDims.y() + j;
					pIndices[uiCi + 1] = (i+1)*m_iDims.y() + j;
					pIndices[uiCi + 2] = m_iDims.y()*i + j + 1;

					pIndices[uiCi + 3] = pIndices[uiCi + 2];
					pIndices[uiCi + 4] = pIndices[uiCi + 1];
					pIndices[uiCi + 5] = (i+1)*m_iDims.y() + j + 1;
				}
			}

			iBufferDesc.Usage = D3D11_USAGE_DEFAULT;
			iBufferDesc.ByteWidth = sizeof(uint32) * m_uiNumIndices;
			iBufferDesc.BindFlags			= D3D11_BIND_INDEX_BUFFER;
			iBufferDesc.CPUAccessFlags		= 0;
			iBufferDesc.MiscFlags			= 0;
			iBufferDesc.StructureByteStride = 0;

			D3D11_SUBRESOURCE_DATA  iData;
			memset(&iData, 0, sizeof(D3D11_SUBRESOURCE_DATA));

			iData.pSysMem = pIndices;
			iData.SysMemPitch = 0;
			iData.SysMemSlicePitch = 0;

			///< Create the index buffer.
			HRESULT hr = _pDevice->CreateBuffer(&iBufferDesc, &iData, &m_pIndexBuffer);
			ASSERT(hr==S_OK, "Failed Creating Index Buffer");

			M::DeleteArray(&pIndices);
		}
	}
}

///<
void Terrain::CreateWithHeightTexture(ID3D11Device* _pDevice, const char* _csHeightMap)
{
	

	CreateDisplacementTexture(_pDevice, _csHeightMap);	

	ID3D11Texture2D* pTexture;
	m_pDisplacementSRV->GetResource((ID3D11Resource**)&pTexture);
	if (pTexture)
	{
		D3D11_TEXTURE2D_DESC texDesc;
		memset(&texDesc,0,sizeof(D3D11_TEXTURE2D_DESC) );
		pTexture->GetDesc(&texDesc);

		m_iDims=Vector2i(texDesc.Width,texDesc.Height);
		pTexture->Release();
	}
	else
	{
		m_iDims=Vector2i(128,128);
	}

	Create(_pDevice,m_iDims);

	///< Visual texture:
	std::string baseTexture = "..\\..\\Ressources\\Textures\\";
	std::string tFileName = std::string("Terrain.png");
	if (tFileName.length()!=0)
	{
		HRESULT hr = D3DX11CreateShaderResourceViewFromFile(_pDevice, (baseTexture+tFileName).c_str(), NULL, NULL, &m_pTextureRV, NULL );
		ASSERT(hr ==S_OK, "Failed Loading Texture !");
	}

	
}

void Terrain::Draw(ID3D11DeviceContext* _pImmediateContext)
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

	if (m_pDisplacementSRV)
	{
		_pImmediateContext->VSSetShaderResources(1,1, &m_pDisplacementSRV);
	}

	ASSERT(m_Stride>0, "Vertex Stride is 0");
	uint32 offset = 0;
	_pImmediateContext->IASetVertexBuffers(0, 1, &m_pVertexBuffer, &m_Stride, &offset );	
	_pImmediateContext->IASetIndexBuffer(m_pIndexBuffer,DXGI_FORMAT_R32_UINT,0);

	_pImmediateContext->DrawIndexed(NumIndices(), 0, 0);
}
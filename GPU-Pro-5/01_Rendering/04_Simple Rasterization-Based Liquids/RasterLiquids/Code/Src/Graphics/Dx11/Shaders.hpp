#ifndef __SHADERS_HPP__
#define __SHADERS_HPP__

#include <Graphics\Dx11\Dx11Renderer.hpp>
#include <Math/Quaternion/Quaternion.hpp>
#include <Math/Matrix/Matrix.hpp>
#include <Common\Incopiable.hpp>

template<class T>
struct Dx11Constants
{
	T					m_t;
	int32				m_reg;
	ID3D11Buffer*		m_pBuffer;

	Dx11Constants(){ memset(this,0,sizeof(Dx11Constants)); m_reg=-1;}
	~Dx11Constants(){ M::Release(&m_pBuffer); }
	
	///<
	void Update(ID3D11DeviceContext* _pContext, int32 _iNumBuffers=1)
	{
		ASSERT(m_reg >=0, "Regestry not set !!");
			
		_pContext->UpdateSubresource(m_pBuffer, 0, NULL, &m_t, 0, 0 );	

		_pContext->VSSetConstantBuffers(m_reg, _iNumBuffers, &m_pBuffer);
		_pContext->GSSetConstantBuffers(m_reg, _iNumBuffers, &m_pBuffer);
		_pContext->PSSetConstantBuffers(m_reg, _iNumBuffers, &m_pBuffer);	
	}

	void Create(ID3D11Device* _pDevice, int32 _iReg)
	{
		m_reg=_iReg;

		D3D11_BUFFER_DESC bd;
		memset(&bd, 0, sizeof(bd));

		bd.Usage = D3D11_USAGE_DEFAULT;
		bd.ByteWidth = sizeof(T);
		bd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
		bd.CPUAccessFlags = 0;
		HRESULT hr = _pDevice->CreateBuffer(&bd, NULL, &m_pBuffer);
		ASSERT(hr==S_OK, "Constants Failed!");	
	}

};





///<
class PhongShader  : public Incopiable
{

	Shader*				m_pPhongShader;
	Shader*				m_pDisplacementShader;

	
	Vector< Quaternionf, 2> m_lightOrientations;

public:

	struct PhongConstants		{  Vector<Vector4f, 4 > _LightPos;		};
	struct MeshColor			{  Vector4f _c; };

private:

	Dx11Constants<MeshColor>			m_DefaultColor;
	Dx11Constants<PhongConstants>		m_LightConstants;

	void UpdateConstants		(ID3D11DeviceContext* _pContext);

public:

	PhongShader					(){ memset(this,0,sizeof(PhongShader));  }
	~PhongShader				();

	void Release				();

	void Create					(ID3D11Device* _pDevice);
	void Set					(ID3D11DeviceContext* _pContext);
	void SetDisplacement		(ID3D11DeviceContext* _pContext);

	void CreateMenu				();

};



///<
class ShaderManager : public Incopiable
{
	friend class	PhongShader;
	friend class	RawShader;

	bool					m_bShadowMap;

	PhongShader				m_phongShader;

	static ShaderManager	m_instance;

public:

	ShaderManager():m_bShadowMap(false){}
	///<
	static ShaderManager& Get()	{return m_instance;}

	~ShaderManager()	{Release();}

	static PhongShader& GetPhong	()		{return Get().m_phongShader; }

	void SetShadowMap(bool _bSM){m_bShadowMap=_bSM;}
	///<
	void Create		(ID3D11Device* _pDevice);
	void Release	();	
};


class CubeUV;

///<
class RayCastShader  : public Incopiable
{

	ID3D11VertexShader*     m_pVertex;
	
	ID3D11InputLayout*      m_pLayout;

	ID3D11PixelShader*      m_pDrawBackFaces;
	ID3D11PixelShader*      m_pDrawFrontFaces;
	ID3D11PixelShader*      m_pRayCastVolume;

	CubeUV*					m_pCube;
	Texture2D_SV			m_RayCastTexture;

public:

	RayCastShader	(){ memset(this,0,sizeof(RayCastShader)); }
	~RayCastShader	();

	void Create		(const char* _csFileName, ID3D11Device* _pDevice, int32 _w, int32 _h);
	void Draw		(ID3D11DeviceContext* _pContext, ID3D11ShaderResourceView* _pVolumeTexture, ID3D11ShaderResourceView* _pDepth, ID3D11RenderTargetView* _pDrawTo, ID3D11ShaderResourceView* _pEnvMap=NULL);
};

#endif
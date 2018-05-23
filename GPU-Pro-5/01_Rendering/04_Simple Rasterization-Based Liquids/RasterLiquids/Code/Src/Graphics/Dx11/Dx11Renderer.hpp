
#ifndef __DX11_RENDERER_HPP__
#define __DX11_RENDERER_HPP__

#include <d3dx11.h>
#include <d3d11.h>

#include <Common/Assert.hpp>
#include <Common/Common.hpp>

#include <Math/Matrix/Matrix.hpp>
#include <Common/Incopiable.hpp>

#include <Input\MenuManager.hpp>

class Dx11Camera;

///<
class Shader
{

	void Release();

public:

	Shader(){ memset(this,0,sizeof(Shader)); }
	~Shader(){Release();}

	ID3D11VertexShader*     m_pVertex;
	ID3D11PixelShader*      m_pPixel;
	ID3D11GeometryShader*	m_pGeometry;

	///< Recreate it each time.
	ID3D11InputLayout*      m_pLayout;

	void Set(ID3D11DeviceContext* _pContext);

};

///< PostProcess Utility
template<class T>
struct Texture_SV
{
	T*							_pT;
	ID3D11ShaderResourceView*	_pSRV;
	ID3D11RenderTargetView*		_pRTV;

	void Release()
	{
		M::Release(&_pT);
		M::Release(&_pSRV);
		M::Release(&_pRTV);	
	}

	Texture_SV(){ memset(this,0,sizeof(Texture_SV<T>)); }

	~Texture_SV(){  }
};


typedef Texture_SV<ID3D11Texture2D> Texture2D_SV;
typedef Texture_SV<ID3D11Texture3D> Texture3D_SV;


///< Renderer
class Dx11Renderer : public Incopiable
{
	
public:
	
	Dx11Camera*					m_pCamera;

	ID3D11Device*				m_pDevice;
	ID3D11DeviceContext*		m_pImmediateContext;
	IDXGISwapChain*				m_pSwapChain;
	ID3D11RenderTargetView*		m_pRenderTargetView;

	ID3D11Texture2D*			m_pDepthStencilTexture;
	ID3D11ShaderResourceView*	m_pDepthStencilSRV;
	ID3D11DepthStencilView*		m_pDepthStencilView;

	uint32	m_w;
	uint32	m_h;

public:

	Dx11Renderer():m_pDevice(NULL),
		m_pCamera(NULL),
		m_pImmediateContext(NULL),
		m_pSwapChain(NULL),
		m_pRenderTargetView(NULL),	
		m_pDepthStencilTexture(NULL),
		m_pDepthStencilView(NULL),
		m_pDepthStencilSRV(NULL),
		m_w(0),
		m_h(0) {}

	virtual ~Dx11Renderer(){ 
		ReleaseDevice();
	}

	///< Public Interface
	virtual bool Create			(HWND _hWnd)=0;
	virtual bool Update			()=0;	
	virtual void Release		()=0;
	virtual void CreateMenu		()=0;

	Vector2f				ScreenDims					() const { return Vector2f(M::SCast<float32>(m_w), M::SCast<float32>(m_h)); }
	static void				CreateShadersAndLayout		(const char* _csName, const char* _csVertexShaderName, const char* _csPixelShaderName, const char* _csGeometryShaderName, D3D11_INPUT_ELEMENT_DESC* _pLayout, int32 _NumElements, Shader* _pShader, ID3D11Device* _pDevice);
	static void				CreateInputLayout			(const char* _csFileName, const char* _csVertexShaderName, ID3D11Device* _pDevice, D3D11_INPUT_ELEMENT_DESC* _pLayout, int32 _NumElements, ID3D11InputLayout** _ppInputLayout);

	static Texture2D_SV		Create2DTexture				(ID3D11Device* _pDevice, const DXGI_FORMAT& _format, const Vector2i dims, const D3D11_SUBRESOURCE_DATA* _pData, D3D11_USAGE _usage=D3D11_USAGE_DEFAULT, uint32 _bind=D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE, uint32 _cpuAccess=0);
	static Texture3D_SV		Create3DTexture				(ID3D11Device* _pDevice, const DXGI_FORMAT& _format, const Vector3i _dims, const D3D11_SUBRESOURCE_DATA* _pData);
	
	static void				CreatePixelShader			(const char* _csFileName, const char* _csPixelShaderName, ID3D11Device* _pDevice, ID3D11PixelShader** _ppPixelShader);
	static void				CreateVertexShader			(const char* _csFileName, const char* _csVertexShaderName, ID3D11Device* _pDevice, ID3D11VertexShader** _ppVertexShader);
	static void				CreateGeometryShader		(const char* _csFileName, const char* _csGeometryShaderName, ID3D11Device* _pDevice, ID3D11GeometryShader** _ppGeometryShader);
	
	static const char*		PSLevel						(ID3D11Device* _pDevice);
	static const char*		VSLevel						(ID3D11Device* _pDevice);
	static const char*		GSLevel						(ID3D11Device* _pDevice);

	static HRESULT			CompileShaderFromFile		(const char* _strFileName, const char* _strEntryPoint, const char* _strShaderModel, ID3DBlob** _ppBlobOut);

	static ID3D11Buffer*	CreatePostProcessQuad		(ID3D11Device* _pDevice);
	static ID3D11Buffer*	CreatePostProcessQuadUVs	(ID3D11Device* _pDevice);
	
	static void				UnbindResources				(ID3D11DeviceContext* _pContext, const int32 _index, const int32 _iNum=1);

protected:
	
	bool					CreateDevice					(HWND _hWnd);
	void					SetCameraParams					();

private:

	bool					CreateTweakBar				();	
	void					ReleaseDevice				();

};

#include <Graphics/Dx11/Utility/Dx11RendererStates.hpp>

#endif


#ifndef __FXAA_HPP__
#define __FXAA_HPP__

#include <Graphics/Dx11/Dx11Renderer.hpp>

class FXAA
{

	bool	m_bFXAA;

	///< Quad Data
	ID3D11Buffer*				m_pQuadVertexBuffer;

	Shader*						m_pShader;

	///< FXAA
	ID3D11SamplerState*         m_pSamAni;   	

	ID3D11Texture2D*			m_pRenderTargetTexture;
	ID3D11ShaderResourceView*	m_pRenderTargetTextureSRV;
	ID3D11RenderTargetView*     m_pRenderTarget;

	ID3D11Buffer*				m_pAAParams;
public:
	struct CBAAParams
	{
		Vector4f wh;
		Vector4f AA_LEVEL;
	};
private:

	void					CreateMenu();

	CBAAParams					m_aaParams;

	///< 
	void						Release				();

public:

	explicit FXAA(const CBAAParams& _params){ memset(this,0,sizeof(FXAA)); memcpy(&m_aaParams, &_params, sizeof(CBAAParams)); }

	~FXAA(){ Release(); ASSERT(m_pQuadVertexBuffer==NULL, "Not Released! "); }

	///<
	void						Render				(ID3D11DeviceContext* _pImmediateContext);

	///<
	ID3D11RenderTargetView*		GetRenderTarget		(){return m_pRenderTarget;}

	///<
	void Create(ID3D11Device* _pDevice, const DXGI_SWAP_CHAIN_DESC* pSwapChainDesc);

private:

	///<
	void CreateConstantBuffers(ID3D11Device* _pDevice);

	///<
	void CreateShaders(ID3D11Device* _pDevice);
};


#endif

#ifndef __GPU_SPH_SHALLOW_WATER_HPP__
#define __GPU_SPH_SHALLOW_WATER_HPP__

#include <Graphics/Dx11/Dx11Renderer.hpp>
#include <Math/Matrix/Matrix.hpp>

class QuadUV;

///< Stable Fluids in DX11
class GPUSPHShallowWater
{

	Shader*						m_pSplat;
	Shader*						m_pRenderParticles;
	Shader*						m_pAddDensityGradient;

	ID3D11VertexShader*			m_pAdvanceParticlesVS;
	ID3D11GeometryShader*		m_pAdvanceParticlesGS;
	ID3D11InputLayout*			m_pSPHParticleLayout;

	ID3D11Buffer*				m_pParticlesVertexBuffer[2];

	ID3D11ShaderResourceView*	m_pDensityFieldSRV;
	ID3D11ShaderResourceView*	m_pOceanBottomSRV;

	QuadUV*						m_pPostQuad;

	Vector2ui					m_iParticlesPerAxis;
	uint32						m_iNumParticles;

	Vector2ui					m_iDims;	
	Matrix4f					m_World;

	Texture2D_SV				m_Up;	
	Texture2D_SV				m_UpCorrected;	
	
	///< Internal
	void					CreateParticles			(ID3D11Device* _pDevice);
	void					CreateContants			(ID3D11Device* _pDevice);	
	void					CreateTextures			(ID3D11Device* _pDevice);
	
	void					SetGridViewPort			(ID3D11DeviceContext* _pImmediateContext);

	///< Simulation Steps
	void					SplatParticles			(ID3D11DeviceContext* _pImmediateContext);
	void					AddDensityGradient		(ID3D11DeviceContext* _pImmediateContext);
	
	void					AdvanceParticles		(ID3D11DeviceContext* _pImmediateContext, Texture2D_SV& _correctedField);
	


	struct Particle
	{
		Vector4f m_x;

		Vector4f m_data;

		///< For Now
		enum Data
		{
			U=0,
			V=1,
			W=2,
			p=3
		};

	};

public:

	///<
	void						ModifyConstants		();

	///<
	void						DrawParticles		(ID3D11DeviceContext* _pImmediateContext);

	ID3D11ShaderResourceView*	GetDensitySRV()		{ return m_Up._pSRV; }
	ID3D11ShaderResourceView*	GetBottomSRV()		{ return m_pOceanBottomSRV; }
	
	void						SetDensityField		(ID3D11ShaderResourceView* _pDensity){m_pDensityFieldSRV=_pDensity;}

	const Vector2ui				GetDims				() const	{ return m_iDims; }
	
	void						Create				(ID3D11Device* _pDevice, ID3D11DeviceContext* _pImmediateContext, const char* _strGroundTexture);
	void						Update				(ID3D11DeviceContext* _pImmediate, ID3D11RenderTargetView* _pRT, ID3D11DepthStencilView* _pDSV, Vector2i _screenDims);

	void						CreateMenu			();

	GPUSPHShallowWater(){ memset(this,0,sizeof(GPUSPHShallowWater)); }

	///<
	~GPUSPHShallowWater();

};


#endif
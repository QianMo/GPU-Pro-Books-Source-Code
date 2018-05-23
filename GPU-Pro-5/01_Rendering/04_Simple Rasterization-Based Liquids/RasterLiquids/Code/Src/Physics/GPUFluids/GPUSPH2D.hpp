
#ifndef __GPU_SPH_2D_HPP__
#define __GPU_SPH_2D_HPP__

#include <Graphics/Dx11/Dx11Renderer.hpp>
#include <Math/Matrix/Matrix.hpp>

class QuadUV;

///< Stable Fluids in DX11
class GPUSPH2D
{
	bool m_bMenuUseJacobi;
	bool m_bCreateDivergenceFree;

	///< 
	ID3D11Buffer*			m_pConstants;

	///< 
	Shader*					m_pSplat;	
	Shader*					m_pRenderParticles; 
	Shader*					m_pAddDensityGradient;

	ID3D11VertexShader*		m_pAdvanceParticlesVS;
	ID3D11GeometryShader*	m_pAdvanceParticlesGS;
	ID3D11InputLayout*      m_pSPHParticleLayout;

	ID3D11Buffer*			m_pParticlesVertexBuffer[2];

	QuadUV*					m_pPostQuad;

	Vector2ui				m_iParticlesPerAxis;
	uint32					m_iNumParticles;

	Vector2ui				m_iDims;	
	Matrix4f				m_World;

	Texture2D_SV m_Up;	
	Texture2D_SV m_UpCorrected;	

	///< FLIP
	Texture2D_SV m_Div;
	Texture2D_SV m_P[2];
	Texture2D_SV m_FLIPCorrectedUp;

	ID3D11PixelShader*	m_pComputeDiv;
	ID3D11PixelShader*	m_pJacobi;
	ID3D11PixelShader*	m_pAddPressureGradient;

	///<
	struct SPHContants
	{
		Vector4f _GridSpacing;
		Vector4f _Gravity;

		Vector4f _InitialDensity;

		float32 _PressureScale;
		float32 _PIC_FLIP;
		float32 _SurfaceDensity;

		float32 _TimeStep;

	};

	SPHContants m_constants;

	void					UpdateConstants			(ID3D11DeviceContext* _pImmediateContext);
	///< 
	void					CreateParticles			(ID3D11Device* _pDevice, const Vector2i _dims, const int32 _iParticlesPerAxis);
	void					CreateContants			(ID3D11Device* _pDevice);	
	void					CreateTextures			(ID3D11Device* _pDevice, const Vector2i _dims);
	
	void					SetGridViewPort			(ID3D11DeviceContext* _pImmediateContext);

	///<
	void					SplatParticles			(ID3D11DeviceContext* _pImmediateContext);
	void					AddDensityGradient		(ID3D11DeviceContext* _pImmediateContext);	
	void					AdvanceParticles		(ID3D11DeviceContext* _pImmediateContext, Texture2D_SV& _correctedField);

	struct Particle
	{
		Vector4f m_x;

		Vector4f m_data;

		enum Data
		{
			U=0,
			V=1,
			W=2,
			p=3
		};

	};

public:

	void						DrawParticles		(ID3D11DeviceContext* _pImmediateContext, ID3D11ShaderResourceView* _pMask);

	ID3D11ShaderResourceView*	GetDensitySRV()		{ return m_P[1]._pSRV; }

	const Vector2ui				GetDims				() const	{ return m_iDims; }
	
	void						Create				(ID3D11Device* _pDevice, ID3D11DeviceContext* _pImmediateContext, const Vector2i _dims, const int32 _iParticlesPerAxis);
	void						Update				(ID3D11DeviceContext* _pImmediate, ID3D11RenderTargetView* _pRT, ID3D11DepthStencilView* _pDSV, Vector2i _screenDims);

	void						CreateMenu			();

	GPUSPH2D(){ memset(this,0,sizeof(GPUSPH2D)); }

	///<
	~GPUSPH2D();

};


#endif
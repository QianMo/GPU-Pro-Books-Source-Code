
#ifndef __SIMPLE_GPU_SPH_HPP__
#define __SIMPLE_GPU_SPH_HPP__

#include <Graphics/Dx11/Dx11Renderer.hpp>
#include <Math/Matrix/Matrix.hpp>

class Cube;
class Mesh;

///<
class GPUSPHConstants : public Incopiable
{
	static GPUSPHConstants	m_instance;

	ID3D11Buffer*			m_pConstants;

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

	

public:

	SPHContants m_constants;

	~GPUSPHConstants(){Release();}
		
	static GPUSPHConstants&		Get				(){ return m_instance;}
	void						Release			(){M::Release(&m_pConstants); }
		
	ID3D11Buffer*		GetConstantsBuffer		(){return m_pConstants;}
	void				UpdateConstants			(ID3D11DeviceContext* _pImmediateContext);
	void				CreateContants			(ID3D11Device* _pDevice, Vector3i _GridDims);	
	void				CreateMenu				();	
};

///< Stable Fluids in DX11
class GPUSPH
{

	struct VolumeVertex
	{
		Vector4f _x;
		Vector3f _uv;
	};

	bool m_bUseJacobi;
	bool m_bCreateDivergenceFree;
	bool m_bCreateJet;
	
	//< 3D Stuff
	ID3D11VertexShader*			m_pVolumeSlicesVS;	
	ID3D11GeometryShader*		m_pVolumeSlicesGS;	
	ID3D11PixelShader*			m_pAddDensityGradientPS;

	///< 
	ID3D11ShaderResourceView*	m_pUserWaterHeightFieldSRV;
	ID3D11ShaderResourceView*	m_pWaterHeightFieldSRV;


	///< FLIP
	ID3D11PixelShader*			m_pComputeDivergencePS;
	ID3D11PixelShader*			m_pJacobiPS;
	ID3D11PixelShader*			m_pAddPressureGradientPS;
		
	///< Splat
	Shader*					m_pSplat;
	Shader*					m_pRenderParticles;

	///< Recreate it each time.
	ID3D11VertexShader*		m_pAdvanceParticlesVS;
	ID3D11GeometryShader*	m_pAdvanceParticlesGS;
	ID3D11InputLayout*      m_pSPHParticleLayout;

	ID3D11InputLayout*      m_pVolumeLayout;
	ID3D11Buffer*			m_pVolumeVertexBuffer;

	ID3D11Buffer*			m_pParticlesVertexBuffer[2];


	uint32 m_iSqrtPerStep;
	uint32 m_iCurrentParticles;

	Vector3ui	m_iParticlesPerAxis;
	uint32		m_iNumParticles;

	Vector3ui	m_iDims;	
	Matrix4f	m_World;

		
	Texture3D_SV m_Div;
	Texture3D_SV m_P[2];///< for Jacobi Iterations!

	Texture3D_SV m_Up;	
	Texture3D_SV m_UpCorrected;	
	Texture3D_SV m_UpCorrectedFLIP;	

	///<
	void					UpdateConstants			(ID3D11DeviceContext* _pImmediateContext);
	void					CreateParticles			(ID3D11Device* _pDevice, const Vector3ui _dims, const uint32 _iParticlesPerAxis);
	void					CreateVolumeTextures	(ID3D11Device* _pDevice, const Vector3ui _dims, DXGI_FORMAT _VectorFormat, DXGI_FORMAT _ScalarFormat);
	void					SetGridViewPort			(ID3D11DeviceContext* _pImmediateContext);

	///<
	void					JacobiIterations		(ID3D11DeviceContext* _pImmediateContext, const uint32 _uiIterations);
	void					ComputeDivergence		(ID3D11DeviceContext* _pImmediateContext, Texture3D_SV& _vel);
	void					SplatParticles			(ID3D11DeviceContext* _pImmediateContext);
	void					AdvanceParticles		(ID3D11DeviceContext* _pImmediateContext, Texture3D_SV& _correctedField);

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

	void						DrawParticles		(ID3D11DeviceContext* _pImmediateContext);

	ID3D11ShaderResourceView*	GetDensitySRV		()			{ return m_Up._pSRV; }

	void						SetHeightField		(ID3D11ShaderResourceView* _pHeightFieldSRV){m_pUserWaterHeightFieldSRV=_pHeightFieldSRV;}
	const Vector3ui&			GetDims				() const	{ return m_iDims; }
	
	void						Create				(ID3D11Device* _pDevice, ID3D11DeviceContext* _pImmediateContext, const Vector3ui _dims, const uint32 _iParticlesPerAxis, const char* _strGround);
	void						Update				(ID3D11DeviceContext* _pImmediate, ID3D11RenderTargetView* _pRT, ID3D11DepthStencilView* _pDSV, Vector2ui _screenDims);

	void						CreateMenu			();

	GPUSPH(){ memset(this,0,sizeof(GPUSPH)); }
	~GPUSPH();
};


#endif
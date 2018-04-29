

#ifndef __DX10SOLVER_IMPL_HPP__
#define __DX10SOLVER_IMPL_HPP__

#include <3D/Solver/DX10Solver.hpp>
#include <Common/System/Assert.hpp>

#ifdef __DX10__

///< DX10
#include <d3d10.h>
#include <d3d10effect.h>
#include <d3dx9math.h>


#include <Common/Math/Vector/Vector.hpp>

///<
struct VertexDefinition
{
	VertexDefinition(){}
	VertexDefinition(D3DXVECTOR4 _pos, D3DXVECTOR3 _UV):Position(_pos),UV(_UV){}
	
	D3DXVECTOR4 Position;  
	D3DXVECTOR3 UV;
};

///<
struct ParticleVertex
{
	ParticleVertex(){}
	ParticleVertex(D3DXVECTOR4 _pos):Position(_pos){}

	D3DXVECTOR4 Position;  

};

///< 2D GPU Solver.
class GPUSolver
{
public:

	ID3D10Texture2D*			m_pField;
	ID3D10Texture2D*			m_pSmokeDensity;

	ID3D10ShaderResourceView*	m_pFieldSRV;
	ID3D10ShaderResourceView*	m_pSmokeDensitySRV;

	ID3D10RenderTargetView*		m_pFieldRTV;
	ID3D10RenderTargetView*		m_pSmokeDensityRTV;

	Vector2i					m_FieldDimension;
	Vector2i					m_SmokeDensityDimension;

	GPUSolver(){ memset(this,0, sizeof(GPUSolver)); }

	void Release();
	~GPUSolver(){}
};

///< 3D GPU Solver.
class TDSolver
{
public:
	ID3D10Texture3D*			m_pField;
	ID3D10ShaderResourceView*	m_pFieldSRV;
	ID3D10RenderTargetView*		m_pFieldRTV;
	Vector3i					m_FieldDimension;

	TDSolver(){ memset(this,0, sizeof(TDSolver)); }

	void Release();
	~TDSolver(){}
};

///< 
class ParticleSystem
{
public:
	ID3D10EffectTechnique*				m_pTechnique;
	ID3D10Buffer*						m_pDrawToBuffer;
	ID3D10Buffer*						m_pDrawFromBuffer;
	ID3D10Buffer*						m_pIndexBuffer;
	ID3D10InputLayout*					m_pVertexLayout;
	
	///< Halo Texture.
	ID3D10Texture2D*					m_pHalo;
	ID3D10ShaderResourceView*			m_pHaloSRV;
	ID3D10EffectShaderResourceVariable*	m_pHaloTextureVariable;
	int32 m_NumParticles;

	ParticleSystem(){memset(this,0, sizeof(ParticleSystem)); }

	bool Create(DX10RendererImpl* _pRenderer);
	void Update(DX10RendererImpl* _pRenderer);

	void Release();
	~ParticleSystem(){}

};


///<
struct TimeStampQuery
{	
	ID3D10Query*		m_pQueryTimeStamp;
	ID3D10Query*		m_pQueryTimeStampDisjoint;
	uint64				m_ticksBegin;
	uint64				m_ticksEnd;

	///<
	TimeStampQuery(ID3D10Device* _pDevice)
	{
		D3D10_QUERY_DESC	m_QueryDesc = {D3D10_QUERY_TIMESTAMP,0};
		HRESULT cc = _pDevice->CreateQuery(&m_QueryDesc, &m_pQueryTimeStamp);
		ASSERT(cc==S_OK, "Failed to create count.  ");

		m_QueryDesc.Query = D3D10_QUERY_TIMESTAMP_DISJOINT;
		cc = _pDevice->CreateQuery(&m_QueryDesc, &m_pQueryTimeStampDisjoint);
		ASSERT(cc==S_OK, "Failed to create count.  ");
	}

	~TimeStampQuery()
	{
		M::Release(&m_pQueryTimeStamp);
		M::Release(&m_pQueryTimeStampDisjoint);
	}

	void Begin()
	{
		m_pQueryTimeStampDisjoint->Begin();

		m_pQueryTimeStamp->End();
		while( S_OK != m_pQueryTimeStamp->GetData(&m_ticksBegin, sizeof(UINT64), 0) )
		{

		}
	}

	float32 End()
	{
		
		m_pQueryTimeStamp->End();
		while( S_OK != m_pQueryTimeStamp->GetData(&m_ticksEnd, sizeof(UINT64), 0) )
		{

		}

		m_pQueryTimeStampDisjoint->End();

		D3D10_QUERY_DATA_TIMESTAMP_DISJOINT frequency;

		while( S_OK != m_pQueryTimeStampDisjoint->GetData(&frequency,sizeof(D3D10_QUERY_DATA_TIMESTAMP_DISJOINT), 0) )
		{

		}

		if (!frequency.Disjoint)
		{
			return static_cast<float32>(m_ticksEnd-m_ticksBegin)/static_cast<float32>(frequency.Frequency);
		}

		return 0;

	}
};

///<
class SolverState
{
public:
	virtual const char* TechniqueName	()=0;
	virtual const char* EffectName		()=0;
	virtual void		Create			(DX10RendererImpl* _pRenderer)=0;
	virtual void		Draw			(DX10RendererImpl* _pRenderer)=0;
	virtual float32		GetNumSteps		() const=0;
	virtual ~SolverState(){}
};

///< 
class DX10RendererImpl
{
	bool									m_bMouseClick;

	HWND									m_handle;
	int32									m_w;
	int32									m_h;
	int32									m_NumSlices;	

	static const int32 NumScreenSteps		= 10;
	static const int32 NumTDSteps			= 3;

	float32									m_fps;
	///< Buffer size.
	static const int32						BSize=2;

	///< Device
	ID3D10Device*							m_pDevice;
	IDXGISwapChain*							m_pSwapChain;
	
	ID3D10RenderTargetView*					m_pTRV;
	ID3D10DepthStencilView*					m_pDSV;
	
	ID3D10Texture2D*						m_pDepthStencil;

	ID3D10RasterizerState*					m_pRasterState;

	///< Geometry.	
	ID3D10Buffer*							m_pVertexBuffer;
	ID3D10Buffer*							m_pIndexBuffer;
	ID3D10Buffer*							m_pConstants;
	ID3D10InputLayout*						m_pVertexLayout;

	///< Effect.
	ID3D10Effect*							m_pEffect;
	ID3D10EffectTechnique*					m_pTechnique;
	ID3D10EffectVectorVariable*				m_pBlowerPosition;
	ID3D10EffectVectorVariable*				m_pBlowerVelocity;
	
	ID3D10EffectMatrixVariable*				m_pVWorld;
	ID3D10EffectMatrixVariable*				m_pVView;
	ID3D10EffectMatrixVariable*				m_pVViewInverse;
	ID3D10EffectMatrixVariable*				m_pVProjection;

	///< GPU Fluid Solver.
	ID3D10EffectShaderResourceVariable*		m_pFieldTexutreVariable;	
	ID3D10EffectShaderResourceVariable*		m_pSmokeDensityTexutreVariable;
	SolverState*							m_pState;
	
	GPUSolver								m_SolverBuffer[BSize];
	TDSolver								m_TDSolverBuffer[BSize];
	ParticleSystem							m_particles;
	TimeStampQuery*							m_pQuery;
	D3D10_DRIVER_TYPE						m_driverType;	
	
	friend class	ScreenSolverState;
	friend class	TDSolverState;
	friend class	ParticleSystem;

	///<
	bool			CreateDevice			();
	///<
	void			SetScreenSpaceData		();

	///<
	void			CreateGPUSolver			();
	static void		CreateGPUSolverBuffer	(ID3D10Device* _pDevice, GPUSolver& _GPUSolverBuffer, const char* _strFieldName, const char* _strDensityFieldName);
	///<
	static void		DrawGPUSolver			(DX10RendererImpl* _pRenderer);
	static void		UpdateFields			(DX10RendererImpl* _pRenderer, GPUSolver& _SolverBuffer1, GPUSolver& _SolverBuffer2);

	///<
	void			CreateTDGPUSolver		();
	///<
	static void		DrawTDGPUSolver			(DX10RendererImpl* _pRenderer);
	static void		UpdateTDFields			(DX10RendererImpl* _pRenderer, TDSolver& _SolverBuffer1, TDSolver& _SolverBuffer2);

	///<
	void 			CreateCamera			();
	bool 			CreateEffect			();
	bool 			CreateScreenGeometry	();

public:

	DX10RendererImpl(HWND _handle, int32 _w, int32 _h)
	{
		memset(this,0,sizeof(DX10RendererImpl));

		m_driverType = D3D10_DRIVER_TYPE_NULL;
		m_handle=_handle;
		m_w		=_w;
		m_h		=_h;
	}

	~DX10RendererImpl(){}
	
	void	Create			(DX10Renderer::SolverType _type);
	float32 Update			(const float32 _dt);
	void	Release			();

	void	MouseClick		(){m_bMouseClick=true;}

	

};


#endif

#endif
#ifndef __SIMPLE_PARTICLE_LIQUIDS_HPP__
#define __SIMPLE_PARTICLE_LIQUIDS_HPP__

#include <Graphics/Dx11/Dx11Renderer.hpp>
#include <xnamath.h>

class FXAA;
class TimeStampQuery;
class QuadUV;

class GPUSPH;
class Cube;
class RayCastShader;
class GPUSPHShallowWater;
class Mesh;
class PhongShader;


///<
class Simple3DLiquids: public Dx11Renderer
{

	///< Sometimes had trouble with ATI hardward my Ray Casting implementation.
	bool		m_bRayCast;

	int32		m_iSliceIndex;

	float32		m_fps;

	FXAA*					m_pFxaa;
	TimeStampQuery*			m_pBench;

	///< Obstacle
	PhongShader*			m_pPhong;
	Mesh*					m_pSphere;

	Mesh*					m_pSlope;
	
	GPUSPH*					m_pGPUSPH;

	ID3D11ShaderResourceView*	m_pEnvMapTextureSRV;
	RayCastShader*				m_pRayCastShader;

	///< 
	struct Constants
	{
		Vector4f _data;
	};

	ID3D11Buffer*				m_pSliceConstants;
	Shader*						m_pRawUVSlice;	
	QuadUV*						m_pQuad;

	void DrawObstacle			(Vector3f _tx, float32 _s);
	///<
	void CreateRayCast			(ID3D11Device* _pDevice);
	///<
	void CreateContants			(ID3D11Device* _pDevice);
	///<	
	void CreatePostQuad			(Vector2i _iDims);
	///<
	void AnimateGPUSPH			(ID3D11RenderTargetView* _pCurrentRT);
	
public:

	Simple3DLiquids() : m_pEnvMapTextureSRV(NULL), m_pFxaa(NULL), m_pSlope(NULL), m_pSphere(NULL), m_pPhong(NULL),
		m_pGPUSPH(NULL),m_pBench(NULL), 
		m_pSliceConstants(NULL), m_pQuad(NULL), m_pRawUVSlice(NULL), m_iSliceIndex(32), 
		m_pRayCastShader(NULL),m_bRayCast(false){}

	virtual ~Simple3DLiquids	(){}

	///<
	virtual void CreateMenu	();
	///<
	virtual bool Create		(HWND _hWnd);
	///<
	virtual bool Update		();
	///<
	virtual void Release	();

};

#endif
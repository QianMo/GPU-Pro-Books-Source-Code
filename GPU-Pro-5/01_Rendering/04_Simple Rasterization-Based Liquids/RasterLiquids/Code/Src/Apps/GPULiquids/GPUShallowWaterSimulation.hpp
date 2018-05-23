#ifndef __GPU_SPH_HEIGHTFIELD_HPP__
#define __GPU_SPH_HEIGHTFIELD_HPP__

#include <Graphics/Dx11/Dx11Renderer.hpp>

class FXAA;
class TimeStampQuery;
class QuadUV;
class GPUSPHShallowWater;
class PhongShader;
class Terrain;

///<
class ParticleShallowWater: public Dx11Renderer
{

	float32		m_fps;

	FXAA*					m_pFxaa;
	TimeStampQuery*			m_pBench;

	struct Constants
	{
		Vector4f _data;
	};
		
	GPUSPHShallowWater*			m_pShallowWater;

	Texture2D_SV				m_up;

	Shader*						m_pRawUV;	
	QuadUV*						m_pQuad;

	PhongShader*				m_pPhong;
	Shader*						m_pWaterSurfaceShader;

	Terrain*					m_pWaterSurface;
	Terrain*					m_pTerrain;

	///<
	void	CreateContants			(ID3D11Device* _pDevice);
	///<	
	void	CreatePostQuad			();
	///<
	void	CreateSurfaces			();

public:

	ParticleShallowWater() : m_pFxaa(NULL), m_pBench(NULL), m_pQuad(NULL), m_pRawUV(NULL), m_pShallowWater(NULL), m_pWaterSurface(NULL), m_pPhong(NULL), m_pTerrain(NULL), m_pWaterSurfaceShader(NULL){}

	virtual ~ParticleShallowWater	(){}

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
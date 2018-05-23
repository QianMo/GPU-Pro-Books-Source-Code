#ifndef __GPU_SPH_BENCHMARKS_HPP__
#define __GPU_SPH_BENCHMARKS_HPP__

#include <Graphics/Dx11/Dx11Renderer.hpp>
#include <xnamath.h>

class FXAA;
class TimeStampQuery;
class QuadUV;
class GPUSPH2D;

///<
class Simple2DLiquids: public Dx11Renderer
{

	float32		m_fps;

	FXAA*					m_pFxaa;
	TimeStampQuery*			m_pBench;

	struct Constants
	{
		Vector4f _data;
	};


	GPUSPH2D*					m_p2DSPH;

	ID3D11ShaderResourceView*	m_pParticleMask;

	Texture2D_SV				m_up;

	Shader*						m_pRawUV;	
	QuadUV*						m_pQuad;

	///<
	void CreateContants			(ID3D11Device* _pDevice);
	///<	
	void CreatePostQuad			();

public:

	Simple2DLiquids() : m_pFxaa(NULL), m_pBench(NULL), m_pQuad(NULL), m_pRawUV(NULL), m_p2DSPH(NULL), m_pParticleMask(NULL){}

	virtual ~Simple2DLiquids	(){}

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
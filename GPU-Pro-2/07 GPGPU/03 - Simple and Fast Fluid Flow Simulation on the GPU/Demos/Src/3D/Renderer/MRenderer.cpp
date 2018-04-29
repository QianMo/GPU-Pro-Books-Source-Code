
#include <Renderer/MRenderer.hpp>
#include <3D/Solver/DX10Solver.hpp>


namespace M
{

	///<
	MRenderer::MRenderer(System::IntPtr _h, System::Int32 _w, System::Int32 _height) : m_handle(_h),m_width(_w),m_height(_height), m_pDX10Renderer(0)
	{
		m_bPause		= false;
		m_bNeedLoad		= false;
		m_bNeedReset	= false;
        m_bNeedRelease  = false;
        m_bReleased     = false;
	}

	///<
	void MRenderer::Load(System::String^ _pName)
	{
		m_bNeedLoad	= true;
		m_pName		= _pName;
	}

	///<
	void MRenderer::LoadImpl(System::String^ _pName)
	{
		if (_pName=="DX10GPUSolver")
		{
			m_pDX10Renderer		=	new DX10Renderer((HWND)m_handle.ToInt32(), m_width, m_height);	
			m_pDX10Renderer->Create(DX10Renderer::ScreenSolver);
		}

		if (_pName=="DX10TDGPUSolver")
		{
			m_pDX10Renderer		=	new DX10Renderer((HWND)m_handle.ToInt32(), m_width, m_height);	
			m_pDX10Renderer->Create(DX10Renderer::TDSolver);
		}
	}

	void MRenderer::MouseClick()
	{
		if (m_pDX10Renderer)
			m_pDX10Renderer->MouseClick();		
	}

	///<
	void MRenderer::Release()
	{
		if (m_pDX10Renderer)
		{
			m_pDX10Renderer->Release();
			delete m_pDX10Renderer;
			m_pDX10Renderer=0;
		}        
	}

	void MRenderer::UpdateOnOff()
	{
		m_bPause=!m_bPause;
	}

	void MRenderer::Change()	{	}

	///<
	void MRenderer::Reset(){}

	///<
	void MRenderer::UpdateUtilitary()
	{
		if (m_bNeedLoad)
		{
            Release();
            LoadImpl(m_pName);
            m_bNeedLoad=false;
        }
        
		if (m_bNeedReset)
		{

			m_bNeedReset=false;
		}

        
	}

	///<
	float MRenderer::UpdateGraphics(const float _dt)
	{
		if (m_pDX10Renderer)
		{
			if (!m_bPause)
				return m_pDX10Renderer->Update(_dt);
		}

		return 0;
	}

	///<
	void MRenderer::UpdateSimulation(const float _dt)
	{
		(void)_dt;
		if (!m_bPause)
		{

		}
	}

	MRenderer::~MRenderer(){}

}


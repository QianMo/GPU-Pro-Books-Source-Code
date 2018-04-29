

#include <3D/Solver/DX10Solver.hpp>

#ifdef __DX10__
#include <3D/Solver/DX10SolverImpl.hpp>
#else

class DX10RendererImpl
{

public:
	DX10RendererImpl(HWND handle, int32 _w, int32 _h){}
	~DX10RendererImpl(){}
	void Create(){}
	void Update(const float32 ){}
	void Release(){}
};
#endif

///< Ctr
DX10Renderer::DX10Renderer(HWND handle, int32 _w, int32 _h) : m_handle(handle), m_w(_w), m_h(_h)
{	
	m_pImpl = new DX10RendererImpl(m_handle,m_w,m_h);	
}

///<
DX10Renderer::~DX10Renderer()
{
	delete m_pImpl;
	m_pImpl=0;
}

void DX10Renderer::Create			(SolverType _type)	{if(m_pImpl)m_pImpl->Create(_type);}
float32 DX10Renderer::Update		(const float32 _dt)	{if(m_pImpl)return m_pImpl->Update(_dt);return 0;}
void DX10Renderer::Release			()					{if(m_pImpl)m_pImpl->Release();}
void DX10Renderer::MouseClick		()					{if(m_pImpl)m_pImpl->MouseClick();}
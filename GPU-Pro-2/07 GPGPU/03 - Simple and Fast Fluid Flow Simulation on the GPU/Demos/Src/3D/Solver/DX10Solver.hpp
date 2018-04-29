
#ifndef __DX10_SOLVER_HPP__
#define __DX10_SOLVER_HPP__

#include <windows.h>

#include <Common/Common.hpp>
#include <Common/System/Types.hpp>

class DX10RendererImpl;

class DX10Renderer
{
	DX10RendererImpl* m_pImpl;

	HWND			m_handle;
	int32			m_w, m_h;

public:

	enum SolverType
	{
		ScreenSolver=0,
		TDSolver
	};

	DX10Renderer	(HWND handle, int32 width, int32 height);	
	
	void	Create		(SolverType);
	float32 Update		(const float32 _dt);
	void	Release		();

	void	MouseClick	();

	~DX10Renderer	();
};

#endif
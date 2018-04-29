
#pragma once

#using <mscorlib.dll>
#include <vector>

class DX10Renderer;

namespace M
{

	

	public ref class MRenderer
	{
	
		

	public : MRenderer(System::IntPtr handle, System::Int32 width, System::Int32 height);

					
                void	Load                   (System::String^ _pName);
				
				void	UpdateUtilitary		();
				float	UpdateGraphics         (const float _dt);
				void	UpdateSimulation       (const float _dt);

				void	UpdateOnOff            ();
                void	Release                ();
				void	Change                 ();
				void	Reset                  ();

				void	MouseClick				();

	private:

				void	LoadImpl               (System::String^ _pName);

		
		bool				m_bPause,m_bNeedReset,m_bNeedLoad,m_bNeedRelease,m_bReleased;
		System::IntPtr		m_handle;
		System::Int32		m_width;
		System::Int32		m_height;

		System::String^		m_pName;
		DX10Renderer*		m_pDX10Renderer;		

	public:
		~MRenderer();

	};
}




#ifndef D3D9DRV_D3D9VIEWPORT_H_INCLUDED
#define D3D9DRV_D3D9VIEWPORT_H_INCLUDED

#include "Wrap3D/Src/Viewport.h"

namespace Mod
{
	class D3D9Viewport : public Viewport
	{
		// types
	public:
		typedef D3DVIEWPORT9 BindType;

		// construction/ destruction
	public:
		D3D9Viewport( const ViewportConfig& cfg );
		~D3D9Viewport();

		// manipulation/ access
	public:
		void		BindTo( IDirect3DDevice9* dev ) const;
		static void	SetBindToZero( IDirect3DDevice9* dev );

		// data
	private:
		D3DVIEWPORT9 mViewport;
	};

}

#endif
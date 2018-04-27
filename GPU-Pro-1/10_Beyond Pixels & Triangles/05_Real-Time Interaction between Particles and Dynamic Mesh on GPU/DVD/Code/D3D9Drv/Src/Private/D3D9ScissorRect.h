#ifndef D3D9DRV_D3D9SCISSORRECT_H_INCLUDED
#define D3D9DRV_D3D9SCISSORRECT_H_INCLUDED

#include "Wrap3D/Src/ScissorRect.h"

namespace Mod
{
	class D3D9ScissorRect : public ScissorRect
	{
		// types
	public:

		// construction/ destruction
	public:
		explicit D3D9ScissorRect( const ScissorRectConfig& cfg );
		~D3D9ScissorRect();

		// manipulation/ access
	public:
		void			BindTo( IDirect3DDevice9* dev ) const;
		static void		SetBindToZero( IDirect3DDevice9* dev );

	};

}



#endif
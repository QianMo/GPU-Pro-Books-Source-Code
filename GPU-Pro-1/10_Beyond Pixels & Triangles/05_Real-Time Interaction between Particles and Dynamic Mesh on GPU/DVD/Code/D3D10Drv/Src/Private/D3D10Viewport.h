#ifndef D3D10DRV_D3D10VIEWPORT_H_INCLUDED
#define D3D10DRV_D3D10VIEWPORT_H_INCLUDED

#include "Wrap3D/Src/Viewport.h"

namespace Mod
{
	class D3D10Viewport : public Viewport
	{
		// types
	public:
		typedef D3D10_VIEWPORT BindType;

		// construction/ destruction
	public:
		D3D10Viewport( const ViewportConfig& cfg );
		~D3D10Viewport();

		// manipulation/ access
	public:
		void		BindTo( BindType& slot ) const;
		static void	SetBindToZero( BindType& slot );


		// data
	private:
		D3D10_VIEWPORT mViewport;
	};

	bool operator ! ( const D3D10_VIEWPORT& vp );
}

#endif
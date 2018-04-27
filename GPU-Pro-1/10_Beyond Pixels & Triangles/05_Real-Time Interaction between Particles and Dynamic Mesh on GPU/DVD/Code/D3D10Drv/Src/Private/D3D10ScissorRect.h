#ifndef D3D10DRV_D3D10SCISSORRECT_H_INCLUDED
#define D3D10DRV_D3D10SCISSORRECT_H_INCLUDED

#include "Wrap3D/Src/ScissorRect.h"

namespace Mod
{
	class D3D10ScissorRect : public ScissorRect
	{
		// types
	public:
		typedef D3D10_RECT BindType;

		// construction/ destruction
	public:
		explicit D3D10ScissorRect( const ScissorRectConfig& cfg );
		~D3D10ScissorRect();

		// manipulation/ access
	public:
		void			BindTo( BindType& slot ) const;
		static void		SetBindToZero( BindType& slot );

	};

	//------------------------------------------------------------------------

	bool operator ! ( const D3D10_RECT& rect );
}



#endif
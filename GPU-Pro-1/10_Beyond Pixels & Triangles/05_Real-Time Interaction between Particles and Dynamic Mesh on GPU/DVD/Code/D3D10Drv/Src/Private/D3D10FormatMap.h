#ifndef D3D10DRV_D3D10FORMATMAP_H_INCLUDED
#define D3D10DRV_D3D10FORMATMAP_H_INCLUDED

#include "Wrap3D\Src\Forw.h"

namespace Mod
{
	class D3D10FormatMap
	{
		// types
	public:
		typedef Types2< DXGI_FORMAT, const Format* >::Map Map;

		// construction/ destruction
	public:
		explicit D3D10FormatMap( const Formats& formats );
		~D3D10FormatMap();

	public:
		const Format* GetFormat( DXGI_FORMAT fmt ) const;

	private:
		Map mMap;
		
	};
}

#endif
#ifndef D3D9HELPERS_D3D9FORMATMAP_H_INCLUDED
#define D3D9HELPERS_D3D9FORMATMAP_H_INCLUDED

#include "Wrap3D\Src\Forw.h"

namespace Mod
{
	class D3D9FormatMap
	{
		// types
	public:
		typedef Types2< D3DFORMAT, const Format* >::Map Map;

		// construction/ destruction
	public:
		explicit D3D9FormatMap( const Formats& formats );
		~D3D9FormatMap();

	public:
		const Format* GetFormat( D3DFORMAT fmt ) const;

	private:
		Map mMap;
		
	};
}

#endif
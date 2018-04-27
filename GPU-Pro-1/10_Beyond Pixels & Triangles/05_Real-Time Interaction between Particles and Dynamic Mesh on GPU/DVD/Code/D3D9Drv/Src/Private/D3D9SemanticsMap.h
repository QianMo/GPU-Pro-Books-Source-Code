#ifndef D3D9DRV_D3D9SEMANTICSMAP_H_INCLUDED
#define D3D9DRV_D3D9SEMANTICSMAP_H_INCLUDED

#include "Forw.h"


namespace Mod
{

	class D3D9SemanticsMap
	{
		// types
	public:
		typedef Types2< AnsiString, D3DDECLUSAGE > :: Map Map;

		// constructors / destructors
	public:
		explicit D3D9SemanticsMap();
		~D3D9SemanticsMap();
	
		// manipulation/ access
	public:
		D3DDECLUSAGE GetUsage( const AnsiString& semantix ) const;

		static D3D9SemanticsMap& Single();

		// data
	private:
		Map mMap;
	};
}

#endif
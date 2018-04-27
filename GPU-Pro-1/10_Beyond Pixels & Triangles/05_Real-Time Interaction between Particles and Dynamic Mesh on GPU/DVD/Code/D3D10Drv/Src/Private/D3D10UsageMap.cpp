#include "Precompiled.h"
#include "D3D10UsageMap.h"

#include "Wrap3D\Src\Usages.h"

namespace Mod
{

	namespace
	{
		typedef D3D10UsageMap::Map::value_type MapEntry;
		MapEntry Entry( D3D10_USAGE usg, const Usages::UsagePtr& ptr )
		{
			if( ptr.null() )
				return MapEntry( usg, NULL );
			else
				return MapEntry( usg, &*ptr );
		}
	}

	D3D10UsageMap::D3D10UsageMap( const Usages& usages )
	{
		const Usage *IMMUTABLE;

		// nothing goes without some hacking ;]
		{
			Bytes data(1);

#pragma warning(push)
#pragma warning( disable: 4510 4512 4610 ) // some SMFs couldnt be generated..
			struct
			{
				const Usage *& usage;
				Bytes data;
			} dummy = { IMMUTABLE };
#pragma warning(pop)

			usages.AssignImmutable( dummy, data );
		}

#define MD_INSERT_USAGE(usg) mMap.insert( Entry( D3D10_USAGE_##usg, usages.##usg ))

		const int LINE_GUARD_START = __LINE__ + 3;
		// -- DO NOT ADD UNRELATED LINES --
		MD_INSERT_USAGE( DEFAULT );
		MD_INSERT_USAGE( DYNAMIC );
		mMap.insert( MapEntry( D3D10_USAGE_IMMUTABLE, IMMUTABLE ) );
		// -- DO NOT ADD UNRELATED LINES --
		MD_STATIC_ASSERT( __LINE__ - LINE_GUARD_START == Usages::NUM_USAGES );

#undef MD_INSERT_USAGE
	}

	//------------------------------------------------------------------------

	D3D10UsageMap::~D3D10UsageMap()
	{

	}

	//------------------------------------------------------------------------

	const Usage*
	D3D10UsageMap::GetUsage( D3D10_USAGE fmt ) const
	{
		Map::const_iterator found = mMap.find( fmt );

		if( found == mMap.end() )
			return NULL;
		else
			return found->second;
	}

}

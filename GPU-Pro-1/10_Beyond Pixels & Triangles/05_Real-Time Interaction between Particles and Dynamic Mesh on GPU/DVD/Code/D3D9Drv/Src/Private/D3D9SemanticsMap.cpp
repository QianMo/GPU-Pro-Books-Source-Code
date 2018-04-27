#include "Precompiled.h"

#include "ContainerVeneers.h"

#include "D3D9SemanticsMap.h"

namespace Mod
{
	/*explicit*/
	D3D9SemanticsMap::D3D9SemanticsMap()
	{
		mMap.insert( Map::value_type( "POSITION"		, D3DDECLUSAGE_POSITION		) );
		mMap.insert( Map::value_type( "BLENDWEIGHT"		, D3DDECLUSAGE_BLENDWEIGHT	) );
		mMap.insert( Map::value_type( "BLENDINDICES"	, D3DDECLUSAGE_BLENDINDICES	) );
		mMap.insert( Map::value_type( "NORMAL"			, D3DDECLUSAGE_NORMAL		) );
		mMap.insert( Map::value_type( "TEXCOORD"		, D3DDECLUSAGE_TEXCOORD		) );
		mMap.insert( Map::value_type( "TANGENT"			, D3DDECLUSAGE_TANGENT		) );
		mMap.insert( Map::value_type( "BINORMAL"		, D3DDECLUSAGE_BINORMAL		) );
	}

	//------------------------------------------------------------------------

	D3D9SemanticsMap::~D3D9SemanticsMap()
	{

	}

	//------------------------------------------------------------------------

	D3DDECLUSAGE
	D3D9SemanticsMap::GetUsage( const AnsiString& semantix ) const
	{
		return map_get( mMap, semantix );
	}

	//------------------------------------------------------------------------

	/*static*/
	
	D3D9SemanticsMap&
	D3D9SemanticsMap::Single()
	{
		static D3D9SemanticsMap single;
		return single;
	}

}
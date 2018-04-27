#include "Precompiled.h"
#include "D3D10FormatMap.h"

#include "Wrap3D\Src\Formats.h"
#include "Wrap3D\Src\Format.h"


namespace Mod
{

	namespace
	{
		typedef D3D10FormatMap::Map::value_type MapEntry;
		template <typename FmtPtr>
		MapEntry Entry( DXGI_FORMAT fmt, const FmtPtr& ptr )
		{
			if( ptr.null() )
				return MapEntry( fmt, NULL );
			else
				return MapEntry( fmt, static_cast<const Format*>(&*ptr) );
		}
	}

	D3D10FormatMap::D3D10FormatMap( const Formats& formats )
	{
#define MD_INSERT_FORMAT(fmt) mMap.insert( Entry( DXGI_FORMAT_##fmt, formats.##fmt ));

		const int LINE_GUARD_START = __LINE__ + 3;
		// -- DO NOT ADD UNRELATED LINES --
		MD_INSERT_FORMAT( R8G8B8A8_UNORM )
		MD_INSERT_FORMAT( R8G8B8A8_SNORM )
		MD_INSERT_FORMAT( R8_UNORM )
		MD_INSERT_FORMAT( R32G32B32A32_FLOAT )
		MD_INSERT_FORMAT( R32G32B32_FLOAT )
		MD_INSERT_FORMAT( R32G32_FLOAT )
		MD_INSERT_FORMAT( R32G32B32_UINT )
		MD_INSERT_FORMAT( R32G32_UINT )
		MD_INSERT_FORMAT( R32_FLOAT )
		MD_INSERT_FORMAT( R32_UINT )
		MD_INSERT_FORMAT( R32_SINT )
		MD_INSERT_FORMAT( R16G16B16A16_FLOAT )
		MD_INSERT_FORMAT( R16G16B16A16_UNORM )
		MD_INSERT_FORMAT( R16G16B16A16_SNORM )
		MD_INSERT_FORMAT( R16G16B16A16_UINT )
		MD_INSERT_FORMAT( R16G16_FLOAT )
		MD_INSERT_FORMAT( R16G16_SNORM )
		MD_INSERT_FORMAT( R16G16_UNORM )
		MD_INSERT_FORMAT( R16_TYPELESS )
		MD_INSERT_FORMAT( R16_FLOAT )
		MD_INSERT_FORMAT( R16_UINT )
		MD_INSERT_FORMAT( R16_UNORM )
		MD_INSERT_FORMAT( D16_UNORM )
		MD_INSERT_FORMAT( R8G8_UNORM )
		MD_INSERT_FORMAT( R8_UINT )
		MD_INSERT_FORMAT( R8G8B8A8_SINT )
		MD_INSERT_FORMAT( R8G8B8A8_UINT )
		MD_INSERT_FORMAT( R8_SINT )
		MD_INSERT_FORMAT( D24_UNORM_S8_UINT )
		MD_INSERT_FORMAT( R24G8_TYPELESS )
		MD_INSERT_FORMAT( X24_TYPELESS_G8_UINT )
		MD_INSERT_FORMAT( R24_UNORM_X8_TYPELESS )
		MD_INSERT_FORMAT( BC1_UNORM )
		MD_INSERT_FORMAT( BC2_UNORM )
		MD_INSERT_FORMAT( BC3_UNORM )
		MD_INSERT_FORMAT( BC5_SNORM )
		MD_INSERT_FORMAT( BC5_UNORM )
		// -- DO NOT ADD UNRELATED LINES --
		MD_STATIC_ASSERT( __LINE__ - LINE_GUARD_START == Formats::NUM_FORMATS );

#undef MD_INSERT_FORMAT
	}

	//------------------------------------------------------------------------

	D3D10FormatMap::~D3D10FormatMap()
	{

	}

	//------------------------------------------------------------------------

	const Format*
	D3D10FormatMap::GetFormat( DXGI_FORMAT fmt ) const
	{
		Map::const_iterator found = mMap.find( fmt );

		if( found == mMap.end() )
			return NULL;
		else
			return found->second;
	}

}

#include "Precompiled.h"

#include "Wrap3D/Src/Exports.h"

#include "Common/Src/StringUtils.h"

#include "D3D9Instance.h"

namespace Mod
{

	struct D3D9GetAvailableModesSetter
	{
		D3D9GetAvailableModesSetter()
		{
			W3DSetGetAvailableModesFunction( GetAvailableModes );
			W3DSetGetModeByIndexFunction( GetModeByIndex );
			W3DSetGetModeIndexFunction( GetModeIndex );
		}

		static Strings GetAvailableModes()
		{
			const D3D9Instance::SimplifiedModeDescs& descs = D3D9Instance::Single().GetSimplifiedModeDescs();

			Strings result( descs.size() );

			for( size_t i = 0, e = descs.size(); i < e; i ++ )
			{
				const D3D9Instance::SimplifiedModeDesc& d = descs[ i ];

				result[ i ] = AsString( d.width ) + L"x" + AsString( d.height );
			}

			return result;
		}

		static void GetModeByIndex( UINT32 idx, UINT32& oWidth, UINT32& oHeight )
		{
			const D3D9Instance::SimplifiedModeDescs& descs = D3D9Instance::Single().GetSimplifiedModeDescs();
			oWidth	= descs[ idx ].width;			
			oHeight	= descs[ idx ].height;
		}

		static UINT32 GetModeIndex( UINT32 width, UINT32 height )
		{
			const D3D9Instance::SimplifiedModeDescs& descs = D3D9Instance::Single().GetSimplifiedModeDescs();

			UINT32 i = 0;
			UINT32 e = UINT32( descs.size() );
			while( width > descs[ i ].width || height > descs[ i ].height && i < e )
			{
				i++;
			}

			if( i == e ) 
				return std::max( i, (UINT32)1 ) - 1;
			else
				return i;
		}

	} gD3D9GetAvailableModesSetter;
}
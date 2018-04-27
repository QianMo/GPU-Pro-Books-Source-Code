#include "Precompiled.h"
#include "D3D10Format.h"

namespace Mod
{
	//------------------------------------------------------------------------

	namespace
	{
		void ConformRGBA( Bytes& ioBytes )
		{
			for( UINT64 i = 0, e = ioBytes.GetSize(); i < e; i += 4 )
			{
				std::swap( ioBytes[ i + 0 ], ioBytes[ i + 2 ] );
			}
		}
	}	

	//------------------------------------------------------------------------

	template <>	FormatConfig CreateFmtConfig<DXGI_FORMAT_R8G8B8A8_UNORM>( ID3D10Device* dev )
	{
		FormatConfig cfg;
		CreateFmtConfigImpl<DXGI_FORMAT_R8G8B8A8_UNORM> ( dev, cfg );

		cfg.convFuncF		= ConvertFunc_UnpackUNORM< FormatTraits<DXGI_FORMAT_R8G8B8A8_UNORM>::CompType >;
		cfg.conformFunc		= ConformRGBA;

		return cfg;
	}

	//------------------------------------------------------------------------

	template <>	FormatConfig CreateFmtConfig<DXGI_FORMAT_R16G16B16A16_SNORM>( ID3D10Device* dev )
	{
		FormatConfig cfg;
		CreateFmtConfigImpl<DXGI_FORMAT_R16G16B16A16_SNORM> ( dev, cfg );

		cfg.convFuncF		= ConvertFunc_UnpackSNORM< FormatTraits<DXGI_FORMAT_R16G16B16A16_SNORM>::CompType >;

		return cfg;
	}

	//------------------------------------------------------------------------

	template <>	FormatConfig CreateFmtConfig<DXGI_FORMAT_R16G16_SNORM>( ID3D10Device* dev )
	{
		FormatConfig cfg;
		CreateFmtConfigImpl<DXGI_FORMAT_R16G16_SNORM> ( dev, cfg );

		cfg.convFuncF		= ConvertFunc_UnpackSNORM< FormatTraits<DXGI_FORMAT_R16G16_SNORM>::CompType >;

		return cfg;
	}

	//------------------------------------------------------------------------

	D3D10Format::~D3D10Format()
	{

	}

	//------------------------------------------------------------------------

	DXGI_FORMAT
	D3D10Format::GetValue() const
	{
		return mDXGIFormat;
	}

	//------------------------------------------------------------------------

	namespace
	{
		class CapsMap
		{
			// types
		public:
			typedef Types< Format::ECaps > :: Vec SupportToCapsMap;

			// construction/ destruction
		public:
			CapsMap();

			// manipulation/ access
		public:
			static CapsMap&	Single();
			Format::ECaps	GetCaps( UINT support );

		private:
			SupportToCapsMap mMap;
		};
	}

	Format::CapsBits D3D10FormatExtractCaps( ID3D10Device* dev, DXGI_FORMAT fmt )
	{
		UINT fmtSupport;
		dev->CheckFormatSupport( fmt, &fmtSupport );

		Format::CapsBits result(0);

		for( UINT i = 0; i < 32; i ++ )
		{
			UINT suppBit = 1 << i;
			if( fmtSupport & suppBit )
				result |= CapsMap::Single().GetCaps( i );
		}

		return result;
	}


	namespace
	{
		template< UINT64 bit >
		struct NumToBitNum;

#define MD_NUM_TO_BITNUM(num) template<> struct NumToBitNum< 1ull << num > { static const UINT64 Result = num; };
		MD_NUM_TO_BITNUM( 0		)
		MD_NUM_TO_BITNUM( 1		)
		MD_NUM_TO_BITNUM( 2		)
		MD_NUM_TO_BITNUM( 3		)
		MD_NUM_TO_BITNUM( 4		)
		MD_NUM_TO_BITNUM( 5		)
		MD_NUM_TO_BITNUM( 6		)
		MD_NUM_TO_BITNUM( 7		)
		MD_NUM_TO_BITNUM( 8		)
		MD_NUM_TO_BITNUM( 9		)
		MD_NUM_TO_BITNUM( 10	)
		MD_NUM_TO_BITNUM( 11	)
		MD_NUM_TO_BITNUM( 12	)
		MD_NUM_TO_BITNUM( 13	)
		MD_NUM_TO_BITNUM( 14	)
		MD_NUM_TO_BITNUM( 15	)
		MD_NUM_TO_BITNUM( 16	)
		MD_NUM_TO_BITNUM( 17	)
		MD_NUM_TO_BITNUM( 18	)
		MD_NUM_TO_BITNUM( 19	)
		MD_NUM_TO_BITNUM( 20	)
		MD_NUM_TO_BITNUM( 21	)
		MD_NUM_TO_BITNUM( 22	)
		MD_NUM_TO_BITNUM( 23	)
		MD_NUM_TO_BITNUM( 24	)
		MD_NUM_TO_BITNUM( 25	)
		MD_NUM_TO_BITNUM( 26	)
		MD_NUM_TO_BITNUM( 27	)
		MD_NUM_TO_BITNUM( 28	)
		MD_NUM_TO_BITNUM( 29	)
		MD_NUM_TO_BITNUM( 30	)
		MD_NUM_TO_BITNUM( 31	)
		MD_NUM_TO_BITNUM( 32	)
		MD_NUM_TO_BITNUM( 33	)
		MD_NUM_TO_BITNUM( 34	)
		MD_NUM_TO_BITNUM( 35	)
		MD_NUM_TO_BITNUM( 36	)
		MD_NUM_TO_BITNUM( 37	)
		MD_NUM_TO_BITNUM( 38	)
		MD_NUM_TO_BITNUM( 39	)
		MD_NUM_TO_BITNUM( 40	)
		MD_NUM_TO_BITNUM( 41	)
		MD_NUM_TO_BITNUM( 42	)
		MD_NUM_TO_BITNUM( 43	)
		MD_NUM_TO_BITNUM( 44	)
		MD_NUM_TO_BITNUM( 45	)
		MD_NUM_TO_BITNUM( 46	)
		MD_NUM_TO_BITNUM( 47	)
		MD_NUM_TO_BITNUM( 48	)
		MD_NUM_TO_BITNUM( 49	)
		MD_NUM_TO_BITNUM( 50	)
		MD_NUM_TO_BITNUM( 51	)
		MD_NUM_TO_BITNUM( 52	)
		MD_NUM_TO_BITNUM( 53	)
		MD_NUM_TO_BITNUM( 54	)
		MD_NUM_TO_BITNUM( 55	)
		MD_NUM_TO_BITNUM( 56	)
		MD_NUM_TO_BITNUM( 57	)
		MD_NUM_TO_BITNUM( 58	)
		MD_NUM_TO_BITNUM( 59	)
		MD_NUM_TO_BITNUM( 60	)
		MD_NUM_TO_BITNUM( 61	)
		MD_NUM_TO_BITNUM( 62	)
		MD_NUM_TO_BITNUM( 63	)
#undef MD_NUM_TO_BITNUM

		CapsMap::CapsMap() :
		mMap( Format::NUM_CAPS )
		{
#define MD_MAP_FMT_CAP(suff, cap) mMap[ NumToBitNum< D3D10_FORMAT_SUPPORT_##suff##cap > :: Result ] = Format::cap;
			const int LINE_GUARD_START = __LINE__ + 3;
			// -- DO NOT ADD UNRELATED LINES --
			MD_MAP_FMT_CAP(		,BUFFER						)
			MD_MAP_FMT_CAP(	IA_	,VERTEX_BUFFER				)
			MD_MAP_FMT_CAP(	IA_	,INDEX_BUFFER				)
			MD_MAP_FMT_CAP( 	,SO_BUFFER					)
			MD_MAP_FMT_CAP( 	,TEXTURE1D					)
			MD_MAP_FMT_CAP( 	,TEXTURE2D					)
			MD_MAP_FMT_CAP( 	,TEXTURE3D					)
			MD_MAP_FMT_CAP( 	,TEXTURECUBE				)
			MD_MAP_FMT_CAP( 	,SHADER_LOAD				)
			MD_MAP_FMT_CAP( 	,SHADER_SAMPLE				)
			MD_MAP_FMT_CAP( 	,SHADER_SAMPLE_COMPARISON	)
			MD_MAP_FMT_CAP( 	,SHADER_SAMPLE_MONO_TEXT	)
			MD_MAP_FMT_CAP( 	,MIP						)
			MD_MAP_FMT_CAP( 	,MIP_AUTOGEN				)
			MD_MAP_FMT_CAP( 	,RENDER_TARGET				)
			MD_MAP_FMT_CAP( 	,BLENDABLE					)
			MD_MAP_FMT_CAP( 	,DEPTH_STENCIL				)
			MD_MAP_FMT_CAP( 	,CPU_LOCKABLE				)
			MD_MAP_FMT_CAP( 	,MULTISAMPLE_RESOLVE		)
			MD_MAP_FMT_CAP( 	,DISPLAY					)
			MD_MAP_FMT_CAP( 	,CAST_WITHIN_BIT_LAYOUT		)
			MD_MAP_FMT_CAP( 	,MULTISAMPLE_RENDERTARGET	)
			MD_MAP_FMT_CAP( 	,MULTISAMPLE_LOAD			)
			MD_MAP_FMT_CAP( 	,SHADER_GATHER				)
			// -- DO NOT ADD UNRELATED LINES --
			MD_STATIC_ASSERT( __LINE__ - LINE_GUARD_START == Format::NUM_CAPS );
#undef MD_MAP_FMT_CAP
		}

		/*static*/
		CapsMap&
		CapsMap::Single()
		{
			static CapsMap single;
			return single;
		}

		//------------------------------------------------------------------------

		Format::ECaps
		CapsMap::GetCaps( UINT support )
		{
			return mMap[ support ];
		}

	}


}
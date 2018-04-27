#include "Precompiled.h"
#include "D3D9Format.h"

namespace Mod
{
	//------------------------------------------------------------------------

	template <>	FormatConfig CreateFmtConfig<D3DFMT_A8R8G8B8, D3DFMT_UNKNOWN>( IDirect3D9 * d3d, UINT devAdapter, D3DDEVTYPE adaptType, FormatConfig::CapsBits caps )
	{
		FormatConfig cfg;
		CreateFmtImpl<D3DFMT_A8R8G8B8, D3DFMT_UNKNOWN> ( d3d, devAdapter, adaptType, cfg, caps );

		cfg.convFuncF		= ConvertFunc_UnpackUNORM< FormatTraits<D3DFMT_A8R8G8B8>::CompType >;

		return cfg;
	}

	//------------------------------------------------------------------------

	template <>	FormatConfig CreateFmtConfig<D3DFMT_A16B16G16R16, D3DFMT_UNKNOWN>( IDirect3D9 * d3d, UINT devAdapter, D3DDEVTYPE adaptType, FormatConfig::CapsBits caps )
	{
		FormatConfig cfg;
		CreateFmtImpl<D3DFMT_A16B16G16R16, D3DFMT_UNKNOWN> ( d3d, devAdapter, adaptType, cfg, caps );

		cfg.convFuncF		= ConvertFunc_UnpackSNORM< FormatTraits<D3DFMT_A16B16G16R16>::CompType >;

		return cfg;
	}

	//------------------------------------------------------------------------

	template <>	FormatConfig CreateFmtConfig<D3DFMT_G16R16, D3DFMT_UNKNOWN>( IDirect3D9 * d3d, UINT devAdapter, D3DDEVTYPE adaptType, FormatConfig::CapsBits caps )
	{
		FormatConfig cfg;
		CreateFmtImpl<D3DFMT_G16R16, D3DFMT_UNKNOWN> ( d3d, devAdapter, adaptType, cfg, caps );

		cfg.convFuncF		= ConvertFunc_UnpackSNORM< FormatTraits<D3DFMT_G16R16>::CompType >;

		return cfg;
	}

	//------------------------------------------------------------------------

	D3D9Format::~D3D9Format()
	{

	}

	//------------------------------------------------------------------------

	D3DFORMAT
	D3D9Format::GetValue() const
	{
		return mD3DFormat;
	}

	//------------------------------------------------------------------------

	D3DFORMAT
	D3D9Format::GetIBFormat() const
	{
		return mD3DIBFormat;
	}

	//------------------------------------------------------------------------

	D3DDECLTYPE
	D3D9Format::GetVertexDeclType() const
	{
		return mD3DVDeclType;
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

	Format::CapsBits D3D9FormatExtractCaps( Format::CapsBits previousCaps, IDirect3D9 * d3d, UINT32 devAdapter, D3DDEVTYPE adaptType, D3DFORMAT fmt )
	{
		Format::CapsBits result( previousCaps );
#if 0

		HRESULT hr = d3d->CheckDeviceFormat( devAdapter, adaptType, D3D9_DISPLAY_FORMAT, 0, D3DRTYPE_INDEXBUFFER, D3DFMT_INDEX16 );
		bool not_available = D3DERR_NOTAVAILABLE  == hr;
		bool invalid_call = D3DERR_INVALIDCALL == hr;
		hr, not_available, invalid_call;

#define MD_D3D9FMT_CHECK(flags,type) if( d3d->CheckDeviceFormat( devAdapter, adaptType, D3D9_DISPLAY_FORMAT, flags, type, fmt ) == D3D_OK )

		if( !IsFormatExtra( fmt ) )
		{
			MD_D3D9FMT_CHECK(D3DUSAGE_RENDERTARGET,D3DRTYPE_SURFACE)
			{
				result |= Format::RENDER_TARGET;
			}

			MD_D3D9FMT_CHECK(0, D3DRTYPE_TEXTURE)
			{
				result |= Format::TEXTURE1D;
				result |= Format::TEXTURE2D;
			}

			MD_D3D9FMT_CHECK(0, D3DRTYPE_VOLUMETEXTURE)
			{
				result |= Format::TEXTURE3D;
			}

			MD_D3D9FMT_CHECK(0, D3DRTYPE_CUBETEXTURE)
			{
				result |= Format::TEXTURECUBE;
			}

			
			MD_D3D9FMT_CHECK(0, D3DRTYPE_VERTEXBUFFER)
			{
				result |= Format::VERTEX_BUFFER;
			}

			MD_D3D9FMT_CHECK(0, D3DRTYPE_INDEXBUFFER)
			{
				result |= Format::INDEX_BUFFER;
			}
		}

#undef MD_D3D9FMT_CHECK
#else
		d3d, devAdapter, adaptType, fmt;
#endif

		return result;
	}

}
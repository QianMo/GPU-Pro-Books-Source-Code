#ifndef D3D9DRV_D3D9FORMAT_H_INCLUDED
#define D3D9DRV_D3D9FORMAT_H_INCLUDED

#include "Wrap3D\Src\Format.h"

#include "Forw.h"

#include "FormatHelpers.h"

namespace Mod
{
	template <D3DFORMAT fmt, D3DFORMAT ibfmt>
	struct FormatProxy
	{
		static const D3DFORMAT format		= fmt;
		static const D3DFORMAT ib_format	= fmt;
	};

	class D3D9Format : public Format
	{
	public:
		template < D3DFORMAT fmt, D3DFORMAT ibfmt >
		explicit D3D9Format( const FormatProxy<fmt,ibfmt>& proxy, IDirect3D9* d3d, UINT adapterID, D3DDEVTYPE devType, D3DDECLTYPE vdeclType, FormatConfig::CapsBits bits );

		virtual ~D3D9Format();

	public:
		D3DFORMAT	GetValue() const;
		D3DFORMAT	GetIBFormat() const;
		D3DDECLTYPE	GetVertexDeclType() const;

	private:
		D3DFORMAT	mD3DFormat;
		D3DFORMAT	mD3DIBFormat;
		D3DDECLTYPE	mD3DVDeclType;
	};

	Format::CapsBits D3D9FormatExtractCaps( Format::CapsBits previousCaps, IDirect3D9 * d3d, UINT devAdapter, D3DDEVTYPE adaptType, D3DFORMAT fmt );

	template < D3DFORMAT fmt, D3DFORMAT ibfmt >
	void CreateFmtImpl( IDirect3D9 * d3d, UINT devAdapter, D3DDEVTYPE adaptType, FormatConfig& cfg, FormatConfig::CapsBits bits )
	{
		cfg.bitCount		= FormatTraits<fmt>::BitCount;
		cfg.capsBits		= bits;
		cfg.capsBits		= D3D9FormatExtractCaps( cfg.capsBits, d3d, devAdapter, adaptType, fmt );
		cfg.capsBits		= D3D9FormatExtractCaps( cfg.capsBits, d3d, devAdapter, adaptType, ibfmt );
		cfg.componentCount	= FormatTraits<fmt>::ComponentCount;

		cfg.convFuncF		= ConvertFunc<FormatTraits<fmt>::CompType,float>;
		cfg.convFuncI		= ConvertFunc<FormatTraits<fmt>::CompType,int>;
	}

	template <D3DFORMAT fmt, D3DFORMAT ibfmt >
	FormatConfig CreateFmtConfig( IDirect3D9 * d3d, UINT devAdapter, D3DDEVTYPE adaptType, FormatConfig::CapsBits bits )
	{
		FormatConfig cfg;

		CreateFmtImpl<fmt,ibfmt>( d3d, devAdapter, adaptType, cfg, bits );

		return cfg;
	}

	template <>	FormatConfig CreateFmtConfig<D3DFMT_A8R8G8B8, D3DFMT_UNKNOWN>( IDirect3D9 *, UINT32, D3DDEVTYPE, FormatConfig::CapsBits );
	template <>	FormatConfig CreateFmtConfig<D3DFMT_A16B16G16R16, D3DFMT_UNKNOWN>( IDirect3D9 *, UINT32, D3DDEVTYPE, FormatConfig::CapsBits );
	template <>	FormatConfig CreateFmtConfig<D3DFMT_G16R16, D3DFMT_UNKNOWN>( IDirect3D9 *, UINT32, D3DDEVTYPE, FormatConfig::CapsBits );

	template < D3DFORMAT fmt, D3DFORMAT ibfmt >
	D3D9Format::D3D9Format( const FormatProxy<fmt,ibfmt>& proxy, IDirect3D9* d3d, UINT adapterID, D3DDEVTYPE devType, D3DDECLTYPE vdeclType, FormatConfig::CapsBits bits ):
	Format( CreateFmtConfig< fmt, ibfmt >( d3d, adapterID, devType, bits ) ),
	mD3DFormat( fmt ),
	mD3DIBFormat( ibfmt ),
	mD3DVDeclType( vdeclType )
	{
		&proxy;		
	}
}



#endif
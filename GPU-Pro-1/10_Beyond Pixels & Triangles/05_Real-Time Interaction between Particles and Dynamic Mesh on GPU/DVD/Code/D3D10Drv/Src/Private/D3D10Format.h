#ifndef D3D10DRV_D3D10FORMAT_H_INCLUDED
#define D3D10DRV_D3D10FORMAT_H_INCLUDED

#include "Wrap3D\Src\Format.h"

#include "FormatHelpers.h"

namespace Mod
{
	template <DXGI_FORMAT fmt>
	struct FormatProxy
	{
		static const DXGI_FORMAT format = fmt;
	};

	class D3D10Format : public Format
	{
	public:
		template <DXGI_FORMAT fmt>
		explicit D3D10Format( const FormatProxy<fmt>& proxy, ID3D10Device *dev );

		virtual ~D3D10Format();

	public:
		DXGI_FORMAT	GetValue() const;

	private:
		DXGI_FORMAT mDXGIFormat;
	};

	Format::CapsBits D3D10FormatExtractCaps( ID3D10Device* dev, DXGI_FORMAT fmt );

	template <DXGI_FORMAT fmt>
	void CreateFmtConfigImpl( ID3D10Device* dev, FormatConfig& cfg )
	{
		cfg.bitCount		= FormatTraits<fmt>::BitCount;
		cfg.capsBits		= D3D10FormatExtractCaps( dev, fmt );
		cfg.componentCount	= FormatTraits<fmt>::ComponentCount;

		cfg.convFuncF		= ConvertFunc<FormatTraits<fmt>::CompType,float>;
		cfg.convFuncI		= ConvertFunc<FormatTraits<fmt>::CompType,int>;
		cfg.conformFunc		= NULL;
	}

	template <DXGI_FORMAT fmt>
	FormatConfig CreateFmtConfig( ID3D10Device* dev )
	{
		FormatConfig cfg;

		CreateFmtConfigImpl<fmt>( dev, cfg );

		return cfg;
	}

	template <>	FormatConfig CreateFmtConfig<DXGI_FORMAT_R8G8B8A8_UNORM>( ID3D10Device* );
	template <>	FormatConfig CreateFmtConfig<DXGI_FORMAT_R16G16B16A16_SNORM>( ID3D10Device* );
	template <>	FormatConfig CreateFmtConfig<DXGI_FORMAT_R16G16_SNORM>( ID3D10Device* );

	template <DXGI_FORMAT fmt>
	D3D10Format::D3D10Format( const FormatProxy<fmt>& proxy, ID3D10Device * dev ):
	Format( CreateFmtConfig<fmt>( dev ) ),
	mDXGIFormat( fmt )
	{
		&proxy;		
	}
}



#endif
#include "Precompiled.h"
#include "Wrap3D\Src\BufferConfig.h"
#include "D3D10Buffer_Index.h"
#include "D3D10Format.h"

namespace Mod
{
	//------------------------------------------------------------------------
	D3D10Buffer_Index::D3D10Buffer_Index( const BufConfigType& cfg, ID3D10Device* dev ) : 
	Parent( cfg, D3D10_BIND_INDEX_BUFFER, dev )
	{
	}

	//------------------------------------------------------------------------
	D3D10Buffer_Index::~D3D10Buffer_Index()
	{
	}

	//------------------------------------------------------------------------

	void
	D3D10Buffer_Index::BindTo( ID3D10Device * dev ) const
	{
		MD_CHECK_TYPE( const IndexBufferConfig, &GetConfig() );
		const IndexBufferConfig& cfg = static_cast<const IndexBufferConfig&>(GetConfig());
		dev->IASetIndexBuffer(	GetResourceInternal(), static_cast<const D3D10Format*>( cfg.fmt )->GetValue(),	0 );
	}

	//------------------------------------------------------------------------

	/*static*/
	void
	D3D10Buffer_Index::SetBindToZero( ID3D10Device * dev )
	{
		dev->IASetIndexBuffer( NULL, DXGI_FORMAT_R16_UINT, 0 );
	}

}
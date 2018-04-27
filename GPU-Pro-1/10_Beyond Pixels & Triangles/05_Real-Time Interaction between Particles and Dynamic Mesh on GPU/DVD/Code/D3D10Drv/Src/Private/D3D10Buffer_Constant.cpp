#include "Precompiled.h"
#include "Wrap3D/Src/BufferConfig.h"

#include "D3D10Buffer_Constant.h"

namespace Mod
{
	D3D10Buffer_Constant::D3D10Buffer_Constant( const BufConfigType& cfg, ID3D10Device* dev ) : 
	Parent( cfg, D3D10_BIND_CONSTANT_BUFFER, dev )
	{
	}

	//------------------------------------------------------------------------

	D3D10Buffer_Constant::~D3D10Buffer_Constant()
	{
	}

	//------------------------------------------------------------------------

	void
	D3D10Buffer_Constant::BindToImpl( ID3D10EffectConstantBuffer* slot ) const
	{
		slot->SetConstantBuffer( GetResourceInternal() );
	}

}


#include "Precompiled.h"
#include "D3D10SOBufferImpl.h"

#include "D3D10Buffer_SOVertex.h"
#include "D3D10Buffer_SOShader.h"
#include "D3D10Buffer_SOShaderVertex.h"

namespace Mod
{

	template<typename Config>
	D3D10SOBufferImpl<Config>::D3D10SOBufferImpl ( const BufConfigType& cfg, UINT32 bindFlags, ID3D10Device* dev ) :
	Base( cfg, bindFlags, dev )
	{

	}

	//------------------------------------------------------------------------

	template<typename Config>
	D3D10SOBufferImpl<Config>::~D3D10SOBufferImpl ()
	{

	}

	//------------------------------------------------------------------------


	template<typename Config>
	void
	D3D10SOBufferImpl<Config>::BindToImpl( SOBindSlot& slot ) const
	{
		*slot.offset	= 0;
	}

	//------------------------------------------------------------------------

	template class D3D10SOBufferImpl< D3D10Buffer_SOVertex::Config >;
	template class D3D10SOBufferImpl< D3D10Buffer_SOShader::Config >;
	template class D3D10SOBufferImpl< D3D10Buffer_SOShaderVertex::Config >;

}


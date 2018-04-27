#include "Precompiled.h"

#include "D3D10EffectProviderCommon.h"

#include "D3D10EffectPoolProvider.h"

namespace Mod
{
	D3D10EffectPoolProvider::D3D10EffectPoolProvider( const EffectPoolProviderConfig& cfg ) : 
	Base( cfg )
	{
	}

	//------------------------------------------------------------------------

	D3D10EffectPoolProvider::~D3D10EffectPoolProvider()
	{

	}

	//------------------------------------------------------------------------

	bool
	D3D10EffectPoolProvider::CompileEffectPoolImpl( const Bytes& shlangCode, const EffectKey::Defines& defines, Bytes& oCode, String& oErrors )
	{
		return D3D10CompileEffectImpl( shlangCode, defines, 0, oCode, oErrors );
	}

}
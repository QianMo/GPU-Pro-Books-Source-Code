#include "Precompiled.h"

#include "D3D10EffectProviderCommon.h"

#include "D3D10EffectProvider.h"

namespace Mod
{
	D3D10EffectProvider::D3D10EffectProvider( const EffectProviderConfig& cfg ) : 
	Base( cfg )
	{
	}

	//------------------------------------------------------------------------

	D3D10EffectProvider::~D3D10EffectProvider()
	{

	}

	//------------------------------------------------------------------------

	bool
	D3D10EffectProvider::CompileEffectImpl( const Bytes& shlangCode, const EffectKey::Defines& defines, bool child, Bytes& oCode, String& oErrors )
	{
		return D3D10CompileEffectImpl( shlangCode, defines, child ? D3D10_EFFECT_COMPILE_CHILD_EFFECT : 0, oCode, oErrors );
	}

}
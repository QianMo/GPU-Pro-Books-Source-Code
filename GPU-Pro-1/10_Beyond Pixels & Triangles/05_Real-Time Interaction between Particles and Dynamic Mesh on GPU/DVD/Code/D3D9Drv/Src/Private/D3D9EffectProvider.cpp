#include "Precompiled.h"

#include "D3D9EffectProviderCommon.h"

#include "D3D9EffectProvider.h"

namespace Mod
{
	D3D9EffectProvider::D3D9EffectProvider( const EffectProviderConfig& cfg ) : 
	Base( cfg )
	{
	}

	//------------------------------------------------------------------------

	D3D9EffectProvider::~D3D9EffectProvider()
	{

	}

	//------------------------------------------------------------------------

	bool
	D3D9EffectProvider::CompileEffectImpl( const Bytes& shlangCode, const EffectKey::Defines& defines, bool child, Bytes& oCode, String& oErrors )
	{
		child;
		return D3D9CompileEffectImpl( shlangCode, defines, D3DXFX_NOT_CLONEABLE, oCode, oErrors );
	}

}
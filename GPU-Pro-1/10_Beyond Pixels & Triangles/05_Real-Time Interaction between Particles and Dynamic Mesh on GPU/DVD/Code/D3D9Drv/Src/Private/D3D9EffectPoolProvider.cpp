#include "Precompiled.h"

#include "D3D9EffectProviderCommon.h"

#include "D3D9EffectPoolProvider.h"

namespace Mod
{
	D3D9EffectPoolProvider::D3D9EffectPoolProvider( const EffectPoolProviderConfig& cfg ) : 
	Base( cfg )
	{
	}

	//------------------------------------------------------------------------

	D3D9EffectPoolProvider::~D3D9EffectPoolProvider()
	{

	}

	//------------------------------------------------------------------------

	bool
	D3D9EffectPoolProvider::CompileEffectPoolImpl( const Bytes& shlangCode, const EffectKey::Defines& defines, Bytes& oCode, String& oErrors )
	{
		shlangCode, defines, oErrors;

		AnsiString hello("*Compiled by a certified madman*");

		for( size_t i = 0, e = hello.size(); i < e; i ++ )
			oCode.Append( hello[ i ] );
	
		return true;
	}

}
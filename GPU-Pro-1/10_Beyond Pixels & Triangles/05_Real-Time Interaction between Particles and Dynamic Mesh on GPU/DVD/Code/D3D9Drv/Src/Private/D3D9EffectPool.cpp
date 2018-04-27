#include "Precompiled.h"

#include "D3D9EffectPool.h"

namespace Mod
{
	/*explicit*/
	D3D9EffectPool::D3D9EffectPool( const EffectPoolConfig& cfg ):
	Parent( cfg )
	{
		ID3DXEffectPool* res;
		MD_D3DV( D3DXCreateEffectPool( &res ) );

		mResource.set( res );
	}

	//------------------------------------------------------------------------
	D3D9EffectPool::~D3D9EffectPool()
	{

	}

	//------------------------------------------------------------------------
	D3D9EffectPool::ResourceType
	D3D9EffectPool::GetResource() const
	{
		return mResource;
	}

	//------------------------------------------------------------------------

	void
	D3D9EffectPool::UpdateEffect( const EffectPtr& eff )
	{
		if( !GetEffect() )
			SetEffect( eff );
	}

}
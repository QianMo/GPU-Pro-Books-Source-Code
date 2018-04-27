#ifndef D3D9DRV_D3D9EFFECTPOOL_H_INCLUDED
#define D3D9DRV_D3D9EFFECTPOOL_H_INCLUDED

#include "Wrap3D/Src/EffectPool.h"

namespace Mod
{
	class D3D9EffectPool : public EffectPool
	{
		// types
	public:
		typedef ComPtr<ID3DXEffectPool>		ResourceType;
		// construction/ destruction
	public:
		explicit D3D9EffectPool( const EffectPoolConfig& cfg );
		~D3D9EffectPool();

		// manipulation/ access
	public:
		ResourceType GetResource() const;
		void UpdateEffect( const EffectPtr& eff );

		// data
	private:
		ResourceType mResource;

	};
}

#endif
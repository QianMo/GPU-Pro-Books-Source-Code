#ifndef D3D10DRV_D3D10EFFECTPOOL_H_INCLUDED
#define D3D10DRV_D3D10EFFECTPOOL_H_INCLUDED

#include "Wrap3D/Src/EffectPool.h"

namespace Mod
{
	class D3D10EffectPool : public EffectPool
	{
		// types
	public:
		typedef ComPtr<ID3D10EffectPool>	ResourceType;
		// construction/ destruction
	public:
		explicit D3D10EffectPool( const EffectPoolConfig& cfg, ID3D10Device* dev );
		~D3D10EffectPool();

		// manipulation/ access
	public:
		ResourceType GetResource() const;

		// data
	private:
		ResourceType mResource;

	};
}

#endif
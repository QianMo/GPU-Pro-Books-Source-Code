#ifndef D3D9DRV_D3D9EFFECTPOOLPROVIDER_H_INCLUDED
#define D3D9DRV_D3D9EFFECTPOOLPROVIDER_H_INCLUDED

#include "Forw.h"
#include "Providers/Src/EffectPoolProvider.h"

namespace Mod
{
	class D3D9EffectPoolProvider : public EffectPoolProvider
	{
		// types
	public:
		typedef EffectPoolProvider Base;

		// construction/ destruction
	public:
		explicit D3D9EffectPoolProvider( const EffectPoolProviderConfig& cfg );
		~D3D9EffectPoolProvider();

		// polymorphism
	private:
		virtual bool CompileEffectPoolImpl( const Bytes& shlangCode, const EffectKey::Defines& defines, Bytes& oCode, String& oErrors ) OVERRIDE;
	};
}

#endif
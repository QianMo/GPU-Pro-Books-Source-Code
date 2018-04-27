#ifndef D3D9DRV_D3D9EFFECTPROVIDER_H_INCLUDED
#define D3D9DRV_D3D9EFFECTPROVIDER_H_INCLUDED

#include "Forw.h"
#include "Providers/Src/EffectProvider.h"

namespace Mod
{
	class D3D9EffectProvider : public EffectProvider
	{
		// types
	public:
		typedef EffectProvider Base;

		// construction/ destruction
	public:
		explicit D3D9EffectProvider( const EffectProviderConfig& cfg );
		~D3D9EffectProvider();

		// polymorphism
	private:
		virtual bool CompileEffectImpl( const Bytes& shlangCode, const EffectKey::Defines& defines, bool child, Bytes& oCode, String& oErrors ) OVERRIDE;

	};
}

#endif
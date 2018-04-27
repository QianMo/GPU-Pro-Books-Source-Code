#ifndef D3D10DRV_D3D10EFFECTPROVIDER_H_INCLUDED
#define D3D10DRV_D3D10EFFECTPROVIDER_H_INCLUDED

#include "Forw.h"
#include "Providers/Src/EffectProvider.h"

namespace Mod
{
	class D3D10EffectProvider : public EffectProvider
	{
		// types
	public:
		typedef EffectProvider Base;

		// construction/ destruction
	public:
		explicit D3D10EffectProvider( const EffectProviderConfig& cfg );
		~D3D10EffectProvider();

		// polymorphism
	private:
		virtual bool CompileEffectImpl( const Bytes& shlangCode, const EffectKey::Defines& defines, bool child, Bytes& oCode, String& oErrors ) OVERRIDE;

	};
}

#endif
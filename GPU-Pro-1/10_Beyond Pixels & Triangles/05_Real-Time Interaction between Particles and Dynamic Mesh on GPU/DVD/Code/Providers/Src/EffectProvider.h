#ifndef PROVIDERS_EFFECTPROVIDER_H_INCLUDED
#define PROVIDERS_EFFECTPROVIDER_H_INCLUDED

#include "Wrap3D/Src/Forw.h"

#include "EffectDefine.h"

#include "Provider.h"

#include "Forw.h"

#include "EffectKey.h"

#include "ExportDefs.h"

#define MD_NAMESPACE EffectProviderNS
#include "ConfigurableImpl.h"

namespace Mod
{

	struct EffectProviderTConfig : DefaultProvTConfig< EffectProvider, Effect, EffectProviderNS::ConfigurableImpl<EffectProviderConfig> >
	{
		typedef EffectKey KeyType;
	};


	class EffectProvider : public Provider< EffectProviderTConfig >
	{
		friend Parent;

		// construction/ destruction
	public:
		EXP_IMP	explicit EffectProvider( const EffectProviderConfig& cfg );
		EXP_IMP	virtual ~EffectProvider();

		// static polymorphism
	private:
		EffectPtr CreateItemImpl( const EffectKey& key );

		// polymorphism
	private:
		EXP_IMP virtual bool CompileEffectImpl( const Bytes& shlangCode, const EffectKey::Defines& defines, bool child, Bytes& oCode, String& oErrors ) = 0;

	};

	bool operator < ( const EffectKey& def1, const EffectKey& def2 );
}

#endif
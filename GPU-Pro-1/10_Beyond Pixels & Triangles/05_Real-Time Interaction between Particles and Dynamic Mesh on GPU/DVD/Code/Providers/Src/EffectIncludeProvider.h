#ifndef	PROVIDERS_EFFECTINCLUDEPROVIDER_H_INCLUDED
#define PROVIDERS_EFFECTINCLUDEPROVIDER_H_INCLUDED

#include "WrapSys/Src/Forw.h"

#include "Forw.h"

#include "Provider.h"

#include "ExportDefs.h"

#define MD_NAMESPACE EffectIncludeProviderNS
#include "ConfigurableImpl.h"


namespace Mod
{
	class EffectIncludeProvider : public Provider	<
											DefaultProvTConfig	< 
																	EffectIncludeProvider,
																	Bytes,
																	EffectIncludeProviderNS::ConfigurableImpl<EffectIncludeProviderConfig>
																>
												>
	{
		friend Parent;

		// types
	public:
		typedef Types2< String, FileStamp > :: Map FileStampMap;

		// construction/ destruction
	public:
		EXP_IMP explicit EffectIncludeProvider( const EffectIncludeProviderConfig& cfg );
		EXP_IMP ~EffectIncludeProvider();

		// manipulation/ access
	public:
		using Parent::RemoveItem;

		bool		AreIncludesModified() const;

		// static polymorphism
	private:
		ItemTypePtr	CreateItemImpl( const KeyType& key );

		// data
	private:
		FileStampMap	mIncludeStampMap;
		bool			mIncludesModified;

	};
}

#endif
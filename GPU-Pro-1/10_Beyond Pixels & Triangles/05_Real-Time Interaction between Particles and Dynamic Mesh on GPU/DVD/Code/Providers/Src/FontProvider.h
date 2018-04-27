#ifndef PROVIDERS_FONTPROVIDER_H_INCLUDED
#define PROVIDERS_FONTPROVIDER_H_INCLUDED

#include "Forw.h"

#include "Provider.h"

#include "ExportDefs.h"

#define MD_NAMESPACE FontProviderNS
#include "ConfigurableImpl.h"

namespace Mod
{

	class FontProvider : public Provider		<
													DefaultProvTConfig	
														<
															FontProvider,
															Font,
															FontProviderNS::ConfigurableImpl<FontProviderConfig>
														>
													>
	{
		friend Parent;

		// types
	public:

		// constructors / destructors
	public:
		explicit FontProvider( const FontProviderConfig& cfg );
		~FontProvider();
	
		// manipulation/ access
	public:

		// static polymorphism
	private:
		ItemTypePtr CreateItemImpl( const KeyType& key );

	};
}

#endif
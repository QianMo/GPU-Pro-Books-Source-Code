#ifndef PROVIDERS_CONSTEVALNODEPROVIDER_H_INCLUDED
#define PROVIDERS_CONSTEVALNODEPROVIDER_H_INCLUDED

#include "Forw.h"

#include "Provider.h"

#include "ExportDefs.h"

#define MD_NAMESPACE ConstEvalNodeProviderNS
#include "ConfigurableImpl.h"


namespace Mod
{
	class ConstEvalNodeProvider : public Provider	<
											DefaultProvTConfig	<
																		ConstEvalNodeProvider,
																		ConstEvalNode,
																		ConstEvalNodeProviderNS::ConfigurableImpl<ConstEvalNodeProviderConfig>
																>
													>
	{
		friend Parent;

		// construction/ destruction
	public:
		EXP_IMP explicit ConstEvalNodeProvider( const ConstEvalNodeProviderConfig& cfg );
		EXP_IMP ~ConstEvalNodeProvider();

		// manipulation/ access
	public:
		EXP_IMP void Parse( ConstExprParserPtr parser );

		using Parent::ForConstEach;


		// static polymorphism
	private:
		ItemTypePtr CreateItemImpl( const KeyType& key );

		// data
	private:
		bool				mParsed;

	};

	EXP_IMP ConstEvalNodeProviderPtr CreateAndParseConstEvalNodeProvider( const String& doc, ConstExprParserPtr parser );
}


#endif
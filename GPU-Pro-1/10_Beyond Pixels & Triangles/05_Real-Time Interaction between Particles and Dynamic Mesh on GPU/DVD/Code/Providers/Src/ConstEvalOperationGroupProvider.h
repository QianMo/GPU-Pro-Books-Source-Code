#ifndef PROVIDERS_CONSTEVALOPERATIONGROUPPROVIDER_H_INCLUDED
#define PROVIDERS_CONSTEVALOPERATIONGROUPPROVIDER_H_INCLUDED

#include "Forw.h"

#include "Provider.h"

#include "ExportDefs.h"

#define MD_NAMESPACE ConstEvalOperationGroupProviderNS
#include "ConfigurableImpl.h"

namespace Mod
{

	class ConstEvalOperationGroupProvider :  public Provider	<
												DefaultProvTConfig	< 
																			ConstEvalOperationGroupProvider,
																			ConstEvalOperationGroup,
																			ConstEvalOperationGroupProviderNS::ConfigurableImpl<ConstEvalOperationGroupProviderConfig>
																	>
														>
	{
		friend Parent;

		// types
	public:
		typedef Types2< String, ConstEvalOperationGroupPtr >:: Map OperationMap;

		// construction/ destruction
	public:
		EXP_IMP ConstEvalOperationGroupProvider( const ConstEvalOperationGroupProviderConfig& cfg );
		EXP_IMP ~ConstEvalOperationGroupProvider();

	};
}

#endif
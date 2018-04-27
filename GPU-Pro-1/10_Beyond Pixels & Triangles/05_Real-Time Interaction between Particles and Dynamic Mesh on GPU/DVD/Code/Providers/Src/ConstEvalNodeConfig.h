#ifndef PROVIDERS_CONSTEVALNODECONFIG_H_INCLUDED
#define PROVIDERS_CONSTEVALNODECONFIG_H_INCLUDED

#include "Common/Src/VarType.h"

#include "Math/Src/Types.h"

#include "Forw.h"

namespace Mod
{
	struct ConstEvalNodeConfig
	{
		typedef Types< ConstEvalNodePtr > :: Vec	Parents;

		// this one must be simple ptr to avoid cyclic sharer ptr links
		typedef Types< ConstEvalNode* > :: Vec		Children;

		typedef void (*Operation)( VarVariant&, Children& );

		String			name;
		String			expr;

		Operation		operation;
		Children		children;

		VarType::Type	type;

		ConstEvalNodeConfig();
	};

	inline
	ConstEvalNodeConfig::ConstEvalNodeConfig() :
	operation( NULL ),
	type( VarType::UNKNOWN )
	{

	}
}

#endif
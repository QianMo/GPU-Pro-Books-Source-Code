#ifndef PROVIDERS_CONSTEXPRPARSER_H_INCLUDED
#define PROVIDERS_CONSTEXPRPARSER_H_INCLUDED

#include "Forw.h"

#include "ExportDefs.h"

#define MD_NAMESPACE ConstExprParserNS
#include "ConfigurableImpl.h"

namespace Mod
{
	class ConstExprParser : public ConstExprParserNS::ConfigurableImpl< ConstExprParserConfig >
	{
		// types
	public:
		typedef Types< ConstEvalNodePtr > :: Vec ConstEvalNodes;

		// construction/ destruction
	public:
		EXP_IMP explicit ConstExprParser( const ConstExprParserConfig& cfg );
		EXP_IMP ~ConstExprParser();

		// manipulation/ access
	public:
		ConstEvalNodePtr Parse( const String& str, ConstEvalNodeProvider* prov );

		// helpers
	private:
		ConstEvalNodePtr			CreateNode( ConstEvalOperationGroupPtr op, ConstEvalNodePtr first ) const;
		ConstEvalNodePtr			CreateNode( ConstEvalOperationGroupPtr op, ConstEvalNodePtr first, ConstEvalNodePtr second ) const;
		ConstEvalNodePtr			CreateNode( ConstEvalOperationGroupPtr op, const ConstEvalNodes& children ) const;				
		ConstEvalNodePtr			GetVarNode( const WCHAR*& str ) const;

		template < UINT32 N >
		ConstEvalOperationGroupPtr	CheckNext( const WCHAR* (&operations)[N], const WCHAR*& str ) const;

		typedef ConstEvalNodePtr (ConstExprParser::*ParsingFunc )( const WCHAR*& );

		template < UINT32 N, const WCHAR* (&operations)[N], ParsingFunc NextPriority >
		ConstEvalNodePtr CreateUnary( const WCHAR*& str );

		template < UINT32 N, const WCHAR* (&operations)[N], ParsingFunc NextPriority >
		ConstEvalNodePtr CreateFunc( const WCHAR*& str );

		template < UINT32 N, const WCHAR* (&operations)[N], ParsingFunc NextPriority >
		ConstEvalNodePtr CreateBinary( const WCHAR*& str );

		ConstEvalNodePtr FetchValueOrRestart( const WCHAR*& str );

		ConstEvalNodeProvider* mCurrentNodeProvider;

	};
}

#endif
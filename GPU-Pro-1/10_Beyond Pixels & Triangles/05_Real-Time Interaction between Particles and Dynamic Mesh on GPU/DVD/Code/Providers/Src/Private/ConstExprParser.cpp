#include "Precompiled.h"

#include "Common/Src/StringParsing.h"

#include "ConstEvalOperationGroupProvider.h"

#include "ConstEvalOperationGroup.h"

#include "ConstEvalNodeProvider.h"

#include "ConstEvalNode.h"

#include "ConstExprParserConfig.h"
#include "ConstExprParser.h"

#define MD_NAMESPACE ConstExprParserNS
#include "ConfigurableImpl.cpp.h"

namespace Mod
{

	namespace
	{
		bool CheckPunctuation( const String& str );
		// remove all spaces
		void Shrink( String& str );

		String GetTokenFromPunctuation( const WCHAR*& str );

		bool CheckClosingBracket(const ConstEvalOperationGroupPtr& op, const WCHAR*& str );
	}

	EXP_IMP
	ConstExprParser::ConstExprParser( const ConstExprParserConfig& cfg ) :
	Parent( cfg )
	{

	}

	//------------------------------------------------------------------------

	EXP_IMP
	ConstExprParser::~ConstExprParser()
	{

	}

	//------------------------------------------------------------------------

	const WCHAR* plus_minus[]			= { L"+", L"-"											};
	const WCHAR* mul_div[]				= { L"*", L"/"											};
	const WCHAR* min_conv[]				= { L"-", L"(float3)", L"(float3_vec)"					};
	const WCHAR* funcs[]				= { L"dot(", L"mul(", L"pow(", L"float3(", L"float4(", 
											L"float4_vec(", L"float4x4(", L"float3x4(", L"inverse(",
											L"normalize(", L"get_scale(", L"scale(", L"max(",
											L"min("												};

#define MD_PARSE_START_FUNCTION_NAME														\
							CreateBinary													\
								<	2,														\
									plus_minus,												\
									&ConstExprParser::CreateBinary							\
									<	2,													\
										mul_div,											\
										&ConstExprParser::CreateUnary						\
										<	3,												\
											min_conv,										\
											&ConstExprParser::CreateFunc					\
											<	14,											\
												funcs,										\
												&ConstExprParser::FetchValueOrRestart		\
											>												\
										>													\
									>														\
								>

	ConstEvalNodePtr
	ConstExprParser::Parse( const String& str, ConstEvalNodeProvider* prov )
	{
		mCurrentNodeProvider = prov;

		struct ResetOnExit
		{
			~ResetOnExit()
			{
				parent->mCurrentNodeProvider = NULL;
			}

			ConstExprParser* parent;
		} resetOnExit = { this }; resetOnExit;


		String cpyStr= str;
		Shrink( cpyStr );

		if( !CheckPunctuation( cpyStr ) )
			return ConstEvalNodePtr();
		else
		{
			const WCHAR* cstr = cpyStr.c_str();
			return MD_PARSE_START_FUNCTION_NAME( cstr );
		}		
	}

	//------------------------------------------------------------------------

	ConstEvalNodePtr
	ConstExprParser::CreateNode( ConstEvalOperationGroupPtr op, ConstEvalNodePtr first ) const
	{
		ConstEvalNodePtr res;

		const ConstEvalOperation& typedOp = op->GetByTypes( first->GetType() );

		ConstEvalNodeConfig cfg;
		cfg.name		= op->GetName();
		cfg.children.resize( 1 );
		cfg.children[0] = first.get();
		cfg.operation	= typedOp.func;
		cfg.type		= typedOp.returnType;

		res.reset( new ConstEvalNode( cfg ) );

		first->LinkParent( res );

		return res;
	}

	//------------------------------------------------------------------------

	ConstEvalNodePtr
	ConstExprParser::CreateNode( ConstEvalOperationGroupPtr op, ConstEvalNodePtr first, ConstEvalNodePtr second ) const
	{
		ConstEvalNodePtr res;

		const ConstEvalOperation& typedOp = op->GetByTypes( first->GetType(), second->GetType() );

		ConstEvalNodeConfig cfg;
		cfg.name		= op->GetName();
		cfg.children.resize( 2 );
		cfg.children[0] = first.get();
		cfg.children[1] = second.get();
		cfg.operation	= typedOp.func;
		cfg.type		= typedOp.returnType;

		res.reset( new ConstEvalNode( cfg ) );

		first->LinkParent( res );
		second->LinkParent( res );

		return res;
	}

	//------------------------------------------------------------------------

	ConstEvalNodePtr
	ConstExprParser::CreateNode( ConstEvalOperationGroupPtr op, const ConstEvalNodes& children ) const
	{
		ConstEvalNodePtr res;

		ConstEvalOperation::VarTypes types( children.size() );

		for( size_t i = 0, e = children.size(); i < e; i ++ )
		{
			types[i] = children[i]->GetType();
		}

		const ConstEvalOperation& typedOp = op->GetByTypes( types );

		ConstEvalNodeConfig cfg;
		cfg.name		= op->GetName();
		cfg.children.resize( children.size() );
		cfg.operation	= typedOp.func;
		cfg.type		= typedOp.returnType;

		for( size_t i = 0, e = children.size(); i < e; i ++ )
		{
			cfg.children[i] = children[i].get();
		}

		res.reset( new ConstEvalNode( cfg ) );

		for( size_t i = 0, e = children.size(); i < e; i ++ )
		{
			children[i]->LinkParent( res );
		}

		return res;
	}

	//------------------------------------------------------------------------

	ConstEvalNodePtr
	ConstExprParser::GetVarNode( const WCHAR*& str ) const
	{
		String token = GetTokenFromPunctuation( str );
		if( token.empty() )
			return ConstEvalNodePtr();
		else
			return mCurrentNodeProvider->GetItem( token );
	}

	//------------------------------------------------------------------------

	template < UINT32 N >
	ConstEvalOperationGroupPtr
	ConstExprParser::CheckNext( const WCHAR* (&operations)[N], const WCHAR*& str ) const
	{
		const String& token = GetToken( operations, N, str );

		if( !token.empty() )
			return GetConfig().operationGroupProv->GetItem( token );
		else
			return ConstEvalOperationGroupPtr();
	}


	//------------------------------------------------------------------------
	// below is the parsing forest

	template < UINT32 N, const WCHAR* (&operations)[N], ConstExprParser::ParsingFunc NextPriority >
	ConstEvalNodePtr
	ConstExprParser::CreateUnary( const WCHAR*& str )
	{
		if( ConstEvalOperationGroupPtr op = CheckNext( operations, str ) )
		{
			ConstEvalNodePtr node = CreateNode( op, CreateUnary<N,operations, NextPriority>( str ) );
			
			bool closed = CheckClosingBracket( op, str );
			MD_FERROR_ON_FALSE( closed );

			return node;
		}
		else
			return (this->*NextPriority)( str );
	}

	namespace
	{
		void SkipComma( const WCHAR*& str )
		{
			MD_FERROR_ON_FALSE( *str == ',' );
			str++;
		}
	}

	template < UINT32 N, const WCHAR* (&operations)[N], ConstExprParser::ParsingFunc NextPriority >
	ConstEvalNodePtr
	ConstExprParser::CreateFunc( const WCHAR*& str )
	{		
		if( ConstEvalOperationGroupPtr op = CheckNext( operations, str ) )
		{
			ConstEvalNodes nodes;

			for(;;)
			{
				ConstEvalNodePtr node;
				if( ! ( node = MD_PARSE_START_FUNCTION_NAME( str ) ) )
					node = (this->*NextPriority)( str );

				nodes.push_back( node );

				if( !CheckClosingBracket( op, str ) )
					SkipComma( str );
				else
					break;
			}

			return CreateNode( op, nodes );
		}
		else
			return (this->*NextPriority)( str );
	}

	template < UINT32 N, const WCHAR* (&operations)[N], ConstExprParser::ParsingFunc NextPriority >
	ConstEvalNodePtr
	ConstExprParser::CreateBinary( const WCHAR*& str )
	{
		ConstEvalNodePtr node = (this->*NextPriority)( str );

		while( ConstEvalOperationGroupPtr op = CheckNext( operations, str ) )
		{
			node = CreateNode( op, node, (this->*NextPriority)( str ) );
		}

		return node;
	}

	template ConstEvalNodePtr ConstExprParser::MD_PARSE_START_FUNCTION_NAME( const WCHAR*& );

	ConstEvalNodePtr
	ConstExprParser::FetchValueOrRestart( const WCHAR*& str )
	{
		if( ConstEvalNodePtr node = GetVarNode( str ) )
			return node;
		else
		if( *str == '(' )
		{
			ConstEvalNodePtr node = MD_PARSE_START_FUNCTION_NAME( ++str );

			MD_FERROR_ON_FALSE( *str == ')' );
			str++;
			
			return node;
		}
		else
			MD_FERROR( L"Error parsing expression!" );

		return ConstEvalNodePtr();
	}

	//------------------------------------------------------------------------

	namespace
	{
		const WCHAR punctuation[] = {',',')','(','-','+','*','/'};
		const WCHAR ident[] = {
			'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
			'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
			'0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
			'_' 
		};

		template <UINT32 N>
		bool CharIs( const WCHAR (&chars)[N], WCHAR ch )
		{
			for( UINT32 i = 0; i < N; i++ )
			{
				if( chars[i] == ch )
					return true;
			}

			return false;
		}

		bool CheckPunctuation( const String& str )
		{
			INT32 brackets(0);

			INT32 punctuationBits = 0;

			for( String::const_iterator i = str.begin(), e = str.end(); i != e; ++i )
			{
				WCHAR ch = *i;
				if( ch == '(' )
				{
					brackets++;
					punctuationBits |= 1;
				}
				else
				if( ch == ')' )
				{
					brackets--;
				}
				else
				if( ch == '-' )
				{
					punctuationBits &= ~1;
					if( punctuationBits )
						return false;

					punctuationBits |= 2;					
				}
				else
				if( CharIs(ident,ch) )
					punctuationBits = 0;
				else
				if( CharIs(punctuation,ch) )
				{
					if( punctuationBits )
						return false;

					punctuationBits |= 2;
				}

				if( brackets < 0 )
					return false;				

			}

			return brackets == 0;

		}

		void Shrink( String& str )
		{
			String::iterator i = str.begin();
			String::iterator e = str.end();

			while( i != e )
			{
				if( *i == ' ' || *i == '\t' || *i == '\n' )
				{
					str.erase( i );
					e = str.end();
				}
				else
					++i;
			}
		}

		String GetTokenFromPunctuation( const WCHAR*& str )
		{
			return GetToken( str, punctuation, sizeof(punctuation)/sizeof(punctuation[0] ) );
		}

		bool CheckClosingBracket(const ConstEvalOperationGroupPtr& op, const WCHAR*& str )
		{
			MD_FERROR_ON_TRUE( op->GetName().empty() );

			if( *(op->GetName().end()-1) == '(' )
			{
				if( *str == ')' )
				{
					str++;
					return true;
				}
				else
					return false;
			}

			return true;
		}
	}

}
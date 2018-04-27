#include "Precompiled.h"

#include "Common/Src/FileHelpers.h"
#include "Common/Src/VarTypeParser.h"
#include "Common/Src/XIAttribute.h"
#include "Common/Src/XIElemArray.h"
#include "Common/Src/XMLDocConfig.h"
#include "Common/Src/XMLDoc.h"

#include "ConstEvalNode.h"

#include "ConstExprParser.h"

#include "ConstEvalNodeProviderConfig.h"
#include "ConstEvalNodeProvider.h"

#define MD_NAMESPACE ConstEvalNodeProviderNS
#include "ConfigurableImpl.cpp.h"


namespace Mod
{

	template class ConstEvalNodeProviderNS::ConfigurableImpl<ConstEvalNodeProviderConfig>;

	//------------------------------------------------------------------------

	EXP_IMP
	ConstEvalNodeProvider::ConstEvalNodeProvider( const ConstEvalNodeProviderConfig& cfg ) :
	Parent( cfg ),
	mParsed( false )
	{
		struct
		{
			ConstEvalNodePtr operator () ( const XMLElemPtr& el ) const
			{
				ConstEvalNodeConfig cfg;
				cfg.name		= XIAttString( el, L"name" );
				cfg.operation	= NULL;
				cfg.type		= VarTypeParser::Single().GetItem( XIAttString(el, L"type") );
				cfg.expr		= XIAttString(el, L"expr", L"");

				return ConstEvalNodePtr( new ConstEvalNode( cfg ) );
			}

		} convert;

		AddItemsFromXMLDoc( cfg.docBytes, convert );

	}

	//------------------------------------------------------------------------

	EXP_IMP
	ConstEvalNodeProvider::~ConstEvalNodeProvider()
	{

	}

	//------------------------------------------------------------------------

	EXP_IMP
	void
	ConstEvalNodeProvider::Parse( ConstExprParserPtr parser )
	{
		MD_FERROR_ON_TRUE( mParsed );

		struct
		{
			void operator () ( const ConstEvalNodePtr& node ) const
			{
				const String& expr = node->GetConfig().expr;

				if( !expr.empty() )
					ConstEvalNode::Set(	node, parser->Parse( expr, parent ) );
			}

			ConstExprParserPtr			parser;
			ConstEvalNodeProvider*		parent;
		} parse = { parser, this };

		ForEach( parse );

		mParsed = true;
	}

	//------------------------------------------------------------------------

	ConstEvalNodeProvider::ItemTypePtr
	ConstEvalNodeProvider::CreateItemImpl( const KeyType& key )
	{
		float val;
		int conv = swscanf( key.c_str(), L"%f", &val );
		MD_FERROR_ON_FALSE( conv );

		ConstEvalNodeConfig ncfg;
		ncfg.type = VarType::FLOAT;

		ConstEvalNodePtr node ( new ConstEvalNode( ncfg ) );
		node->SetVal( val );

		return node;
	}

	//------------------------------------------------------------------------

	EXP_IMP
	ConstEvalNodeProviderPtr
	CreateAndParseConstEvalNodeProvider( const String& doc, ConstExprParserPtr parser )
	{
		ConstEvalNodeProviderConfig cfg;
		ReadBytes( doc, cfg.docBytes );

		ConstEvalNodeProviderPtr prov( new ConstEvalNodeProvider( cfg ) );
		prov->Parse( parser );
		
		return prov;
	}



}
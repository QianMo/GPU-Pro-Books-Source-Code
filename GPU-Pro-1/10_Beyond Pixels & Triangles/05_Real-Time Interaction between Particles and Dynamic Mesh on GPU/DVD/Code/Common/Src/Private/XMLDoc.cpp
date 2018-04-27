#include "Precompiled.h"

#include "WrapSys/Src/System.h"
#include "WrapSys/Src/File.h"
#include "WrapSys/Src/FileConfig.h"

#include "FileHelpers.h"

#include "XMLDocConfig.h"
#include "XMLElem.h"
#include "XMLAttrib.h"
#include "XMLElemConfig.h"
#include "XMLAttribConfig.h"

#include "XMLDoc.h"

#define MD_NAMESPACE XMLDocNS
#include "ConfigurableImpl.cpp.h"

namespace Mod
{
	template class XMLDocNS::ConfigurableImpl< XMLDocConfig >;

	namespace
	{
		struct UserData
		{
			XMLDoc *doc;
			int		depth;
		};

		void XMLCALL startElement(void *userData, const wchar_t *name, const wchar_t **atts)
		{
			UserData* data = static_cast<UserData*>( userData );

			XMLElemConfig cfg;
			cfg.name = name;
			XMLElemPtr el = data->doc->AddElement( cfg, data->depth );

			int i(0);

			while( atts[i] )
			{
				XMLAttribConfig cfg;
				
				cfg.name	= atts[ i + 0 ];
				cfg.value	= atts[ i + 1 ];

				el->SetAttrib( cfg );
				i +=2;
			}

			data->depth ++;
		}

		void XMLCALL
		endElement(void *userData, const wchar_t * /*name*/ )
		{
			UserData* data = static_cast<UserData*>( userData );
			data->depth --;
		}
	}

	XMLDoc::XMLDoc( const XMLDocConfig& cfg ) :
	Parent( cfg )
	{
		if( cfg.source.GetSize() )
		{
			struct ParserWrapper
			{
				ParserWrapper()
				{
					ptr = XML_ParserCreate( NULL );
				}

				~ParserWrapper()
				{
					XML_ParserFree( ptr );
				}

				XML_Parser ptr;
			} parser;

			UserData ud;
			ud.doc		= this;
			ud.depth	= 0;
			
			XML_SetUserData(parser.ptr, &ud);
			XML_SetElementHandler( parser.ptr, startElement, endElement );
			MD_FERROR_ON_FALSE(	XML_Parse(	parser.ptr,	
											(const char*)cfg.source.GetRawPtr(), 
											(int)cfg.source.GetSize(), 1	) == XML_STATUS_OK );
		}
	}

	//------------------------------------------------------------------------

	XMLDoc::~XMLDoc()
	{

	}

	//------------------------------------------------------------------------

	XMLElemPtr
	XMLDoc::GetRoot() const
	{
		return mRoot;
	}

	//------------------------------------------------------------------------

	XMLElemPtr
	XMLDoc::AddElement( const XMLElemConfig& cfg, INT32 depth )
	{
		MD_ASSERT( depth || !mRoot );

		if( depth == 0 )
		{
			mRoot.reset( new XMLElem( cfg ) );
			return mRoot;
		}
		else
		{
			return mRoot->AddChild( cfg, depth - 1 );
		}
	}

	//------------------------------------------------------------------------

	XMLDocPtr CreateXMLDocFromFile( RFilePtr file )
	{
		XMLDocConfig cfg;
		file->Read( cfg.source );
		return XMLDocPtr( new XMLDoc( cfg ) );
	}

	//------------------------------------------------------------------------

	XMLDocPtr CreateXMLDocFromFile( const String& path )
	{
		RFileConfig cfg;
		cfg.fileName = path;
		return CreateXMLDocFromFile( ToRFilePtr( System::Single().CreateFile( cfg ) ) );
	}

	//------------------------------------------------------------------------

	namespace
	{
		void SaveXMLElemToFile( const XMLElemPtr& elem, const WFilePtr& file, UINT32 depth )
		{
			// initial tabs
			{
				AnsiString tabs( depth, '\t' );
				if( !tabs.empty() )
					FileAnsiPrintf( file, tabs.c_str() );
			}
			
			FileAnsiPrintf( file, "<%s ", ToAnsiString( elem->GetName() ).c_str() );

			const XMLElem::Attribs& attribs = elem->GetAttribs();

			for( size_t i = 0, n = attribs.size(); i < n; i ++ )
			{
				FileAnsiPrintf( file, "%s=\"%s\"", ToAnsiString( attribs[i]->GetName() ).c_str(), ToAnsiString( attribs[i]->GetStringValue() ).c_str() );

				if( i != n-1 )
				{
					FileAnsiPrintf( file, " " );
				}
			}

			const XMLElem::Children& children = elem->GetChildren();

			if( size_t n = children.size() )
			{
				FileAnsiPrintf( file, ">\n" );

				for( size_t i = 0; i < n ; i ++ )
					SaveXMLElemToFile( children[i], file, depth + 1 );

				FileAnsiPrintf( file, "</%s>", ToAnsiString(elem->GetName()).c_str() );
			}
			else
			{
				FileAnsiPrintf( file, "/>\n" );
			}
		}
	}

	void SaveXMLDocToFile( XMLDocPtr doc, const String& path )
	{
		WFileConfig cfg( true );
		cfg.fileName		= path;

		WFilePtr file = ToWFilePtr( System::Single().CreateFile( cfg ) );

		SaveXMLElemToFile( doc->GetRoot(), file, 0 );
	}

}
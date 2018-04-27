#include "Precompiled.h"

#include "XMLElemConfig.h"
#include "XMLElem.h"

#include "XMLAttribConfig.h"
#include "XMLAttrib.h"

#define MD_NAMESPACE XMLElemNS
#include "ConfigurableImpl.cpp.h"

namespace Mod
{

	template class XMLElemNS::ConfigurableImpl< XMLElemConfig >;

	XMLElem::XMLElem( const XMLElemConfig& cfg ) :
	Parent( cfg ),
	Named( cfg.name )
	{

	}

	//------------------------------------------------------------------------

	XMLElem::~XMLElem()
	{

	}

	//------------------------------------------------------------------------

	namespace
	{		
		template <typename T>
		typename T::value_type findNthByName( const T& cont, const String& name, INT32 idx )
		{
			struct
			{
				bool operator() ( const T::value_type& p )
				{
					return p->GetName() == name && !idx--;
				}

				String name;
				INT32 idx;
			}condition = { name, idx };

			T::const_iterator found = std::find_if(cont.begin(), cont.end(), condition );
			if( found != cont.end() )
				return *found;
			else
				return T::value_type();
		}
	}

	XMLElemPtr
	XMLElem::GetChild( const String& name, INT32 idx /*= 0*/ ) const
	{
		return findNthByName( mChildren, name, idx );
	}

	//------------------------------------------------------------------------

	XMLAttribPtr
	XMLElem::GetAttrib( const String& name ) const
	{
		return findNthByName( mAttribs, name, 0 );	
	}

	//------------------------------------------------------------------------

	const XMLElem::Children&
	XMLElem::GetChildren() const
	{
		return mChildren;
	}

	//------------------------------------------------------------------------

	const XMLElem::Attribs&
	XMLElem::GetAttribs() const
	{
		return mAttribs;
	}

	//------------------------------------------------------------------------

	XMLElemPtr
	XMLElem::AddChild( const XMLElemConfig& cfg, INT32 depth )
	{
		if( !depth )
		{
			mChildren.push_back( XMLElemPtr( new XMLElem( cfg ) ) );
			return mChildren.back();
		}
		else
		{
			MD_ASSERT( !mChildren.empty() );
			return mChildren.back()->AddChild( cfg, depth - 1 );
		}
	}

	//------------------------------------------------------------------------

	void
	XMLElem::SetAttrib( const XMLAttribConfig& cfg )
	{
		size_t i = 0, e = mAttribs.size();
		for( ; i < e; i ++ )
		{
			if( mAttribs[ i ]->GetConfig().name == cfg.name )
			{
				mAttribs[ i ].reset( new XMLAttrib( cfg ) );
				break;
			}
		}

		if( i == e )
		{
			mAttribs.push_back( XMLAttribPtr( new XMLAttrib( cfg ) ) );
		}
	}

	//------------------------------------------------------------------------

	XMLElemPtr AddChildToXMLElem( const String& name, const XMLElemPtr& elem )
	{
		XMLElemConfig cfg;

		cfg.name = name;
		return elem->AddChild( cfg, 0 );
	}

	//------------------------------------------------------------------------

	void AddAttribToXMLElem( const String& name, const String& value, const XMLElemPtr& elem )
	{
		XMLAttribConfig cfg;

		cfg.name	= name;
		cfg.value	= value;

		elem->SetAttrib( cfg );
	}
	

}

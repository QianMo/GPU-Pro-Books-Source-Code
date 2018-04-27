#ifndef COMMON_XMLELEM_H_INCLUDED
#define COMMON_XMLELEM_H_INCLUDED

#include "Forw.h"

#include "Named.h"

#include "BlankExpImp.h"

#define MD_NAMESPACE XMLElemNS
#include "ConfigurableImpl.h"

namespace Mod
{
	class XMLElem : public XMLElemNS::ConfigurableImpl< XMLElemConfig >,
					public Named
	{
		// types
	public:
		typedef Types < XMLElemPtr >	:: Vec Children;
		typedef Types < XMLAttribPtr >	:: Vec Attribs;

		// construction/ destruction
	public:
		explicit XMLElem( const XMLElemConfig& cfg );
		~XMLElem();

		// manipulation/ access
	public:
		XMLElemPtr		GetChild( const String& name, INT32 idx = 0 ) const;
		XMLAttribPtr	GetAttrib( const String& name ) const;
		const Children&	GetChildren() const;
		const Attribs&	GetAttribs() const;

		XMLElemPtr		AddChild( const XMLElemConfig& cfg, INT32 depth );
		void			SetAttrib( const XMLAttribConfig& cfg );
		
		// data
	private:
		Children	mChildren;
		Attribs		mAttribs;
	};

	//------------------------------------------------------------------------

	XMLElemPtr		AddChildToXMLElem( const String& name, const XMLElemPtr& elem );
	void			AddAttribToXMLElem( const String& name, const String& value, const XMLElemPtr& elem );
}

#endif
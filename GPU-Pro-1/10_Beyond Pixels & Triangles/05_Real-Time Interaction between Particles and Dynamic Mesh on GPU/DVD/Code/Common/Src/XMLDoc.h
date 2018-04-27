#ifndef COMMON_XMLDOC_H_INCLUDED
#define COMMON_XMLDOC_H_INCLUDED

#include "WrapSys/Src/Forw.h"

#include "Forw.h"

#include "BlankExpImp.h"

#define MD_NAMESPACE XMLDocNS
#include "ConfigurableImpl.h"

namespace Mod
{
	class XMLDoc :	public XMLDocNS::ConfigurableImpl< XMLDocConfig >
	{
		// construction/ destruction
	public:
		explicit XMLDoc( const XMLDocConfig& cfg );
		~XMLDoc();

		// manipulation/ access
	public:
		XMLElemPtr	GetRoot() const;
		XMLElemPtr	AddElement( const XMLElemConfig& cfg, INT32 depth );

		// data
	private:
		XMLElemPtr mRoot;
	};

	XMLDocPtr	CreateXMLDocFromFile( RFilePtr file );
	XMLDocPtr	CreateXMLDocFromFile( const String& path );
	void		SaveXMLDocToFile( XMLDocPtr doc, const String& path );
}

#endif
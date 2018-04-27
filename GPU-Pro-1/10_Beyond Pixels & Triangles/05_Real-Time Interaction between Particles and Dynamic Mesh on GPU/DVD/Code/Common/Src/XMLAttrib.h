#ifndef COMMON_XMLATTRIB_H_INCLUDED
#define COMMON_XMLATTRIB_H_INCLUDED

#include "Forw.h"

#include "Named.h"

#include "BlankExpImp.h"

#define MD_NAMESPACE XMLAttribNS
#include "ConfigurableImpl.h"

namespace Mod
{
	class XMLAttrib :	public XMLAttribNS::ConfigurableImpl< XMLAttribConfig >,
						public Named
	{
		// construction/ destruction
	public:
		explicit XMLAttrib( const XMLAttribConfig& cfg );
		~XMLAttrib();

		// manipulation/ access
	public:
		String	GetStringValue() const;
		INT32	GetIntValue() const;
		float	GetFloatValue() const;
		double	GetDoubleValue() const;

		template <typename T>
		T		GetValue() const;

	};
}

#endif
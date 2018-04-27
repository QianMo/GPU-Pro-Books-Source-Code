#ifndef COMMON_XIELEMATTRIBUTE_H_INCLUDED
#define COMMON_XIELEMATTRIBUTE_H_INCLUDED

#include "XIPrereqs.h"

namespace Mod
{

	template <typename T>
	class XIElemAttribute : public XITypeTraits<T>::BaseType
	{
		// types
	public:
		typedef typename XITypeTraits<T>::BaseType	Base;
		typedef typename XITypeTraits<T>::ValueType	ValueType;

		// construction/ destruction
	public:

		explicit XIElemAttribute( const T& defValue ):
		Base(defValue)
		{}

		XIElemAttribute(const XMLElemPtr& elem, const String& elemName, const String& attribName, const T& defValue):
		Base(defValue)
		{
			if(const XMLElemPtr& subElem = elem->GetChild(elemName))
			{
				parseXMLAttrib<T>(subElem, attribName, *this);
			}
		}
		//------------------------------------------------------------------------
		
		XIElemAttribute(const XMLElemPtr& elem, const String& elemName, const String& attribName)
		{
			const XMLElemPtr& subElem = elem->GetChild( elemName );

			MD_FERROR_ON_FALSE( subElem );

			MD_FERROR_ON_FALSE( parseXMLAttrib<T>(subElem, attribName, *this) );
		}

		// manipulation/ access
	public:
		using Base::operator =;
	};

	typedef XIElemAttribute<String>			XIString;
	typedef XIElemAttribute<int>			XIInt;
	typedef XIElemAttribute<unsigned int>	XIUInt;
	typedef XIElemAttribute<float>			XIFloat;
	typedef XIElemAttribute<double>			XIDouble;
}

#endif
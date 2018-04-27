#ifndef COMMON_XIATTRIBUTE_H_INCLUDED
#define COMMON_XIATTRIBUTE_H_INCLUDED

#include "XIPrereqs.h"

namespace Mod
{
	template <typename T>
	class XIAttribute : public XITypeTraits<T>::BaseType
	{
		// types
	public:

		typedef typename XITypeTraits<T>::BaseType Base;
		typedef typename XITypeTraits<T>::AssignType AssignType;

		// construction/ destruction
	public:

		explicit XIAttribute(const T& val):
		Base(val)
		{}

		XIAttribute(const XMLElemPtr& elem, const String& name)
		{
			if(!parseXMLAttrib<T>(elem, name, *this))
			{
				MD_FERROR( L"XIAttribute::XIAttribute: couldnt parse an attrib!" );
			}
		}

		XIAttribute(const XMLElemPtr& elem, const String& name, const T& defValue)
		{
			if(!parseXMLAttrib<T>(elem, name, *this))
				*this = defValue;
		}
		// manipulation/ access
	public:
		AssignType& operator = (const AssignType& base)
		{
			this->Base::operator=(base);
			return *this;
		}

		XIAttribute& operator = (const XIAttribute& base)
		{
			this->Base::operator=(base);
			return *this;
		}
	};

	typedef XIAttribute<INT32>			XIAttInt;
	typedef XIAttribute<float>			XIAttFloat;
	typedef XIAttribute<String>			XIAttString;
	typedef XIAttribute<Math::float3>	XIVector3;
	typedef XIAttribute<Math::float4>	XIVector4;
	typedef XIAttribute<XIQuatHelper>	XIQuaternion;
}


#endif
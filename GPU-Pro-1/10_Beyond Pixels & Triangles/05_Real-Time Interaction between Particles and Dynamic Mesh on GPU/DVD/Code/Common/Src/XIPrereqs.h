#ifndef COMMON_XIPREREQS_H_INCLUDED
#define COMMON_XIPREREQS_H_INCLUDED

#include "Math/Src/Types.h"
#include "Forw.h"

#include "XMLElem.h"
#include "XMLAttrib.h"

namespace Mod
{
	template <typename T>
	class ValueWrapper
	{
		// construction/ destruction
	public:

		ValueWrapper():
		mValue()
		{

		}

		ValueWrapper(const T& value):
		mValue(value)
		{

		}

	protected:
		~ValueWrapper()
		{}
		  // manipulation/ access
	public:
		T& operator = (const T& rhs)
		{
			return mValue = rhs;
		}
		operator T() const
		{
			return mValue;
		}
		operator T& ()
		{
			return mValue;
		}

		// data
	private:
		T mValue;
	};

	//------------------------------------------------------------------------

	template <typename T>
	struct XITypeTraitsBase
	{
		typedef T ValueType;
	};

	template <typename T> 
	struct XITypeTraits : XITypeTraitsBase< T >
	{
		typedef ValueWrapper<T> BaseType;
		typedef T AssignType;
	};

	template <typename T>
	struct XIDirectDeriveTraits : XITypeTraitsBase< T >
	{
		typedef T BaseType;
		typedef T AssignType;
	};

	struct XIQuatHelper : Math::float4
	{};

	template <> struct XITypeTraits<String> :			XIDirectDeriveTraits<String>{};
	template <> struct XITypeTraits<Math::float3> :		XIDirectDeriveTraits<Math::float3> {};
	template <> struct XITypeTraits<Math::float4> :		XIDirectDeriveTraits<Math::float4> {};
	template <> struct XITypeTraits<XIQuatHelper> :		XIDirectDeriveTraits<XIQuatHelper> {};


	//------------------------------------------------------------------------

	template <typename T>
	bool parseXMLAttrib(const XMLElemPtr& elem, const String& name, T& oValue);
}

#endif
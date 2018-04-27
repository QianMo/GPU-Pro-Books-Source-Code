#include "Precompiled.h"

#include "Math/Src/Types.h"
#include "Math/Src/Operations.h"

#include "XIPrereqs.h"

#include "XMLElem.h"
#include "XMLAttrib.h"

namespace Mod
{
	template <typename T>
	bool parseXMLAttrib(const XMLElemPtr& elem, const String& name, T& oValue)
	{
		if( XMLAttribPtr a = elem->GetAttrib( name ) )
		{
			oValue = a->GetValue<T>();
			return true;
		}
		else
			return false;
	}

	// instantiate those that are supported
	template bool parseXMLAttrib( const XMLElemPtr& elem, const String& name, String& oValue		);
	template bool parseXMLAttrib( const XMLElemPtr& elem, const String& name, int& oValue			);
	template bool parseXMLAttrib( const XMLElemPtr& elem, const String& name, unsigned int& oValue	);
	template bool parseXMLAttrib( const XMLElemPtr& elem, const String& name, float& oValue		);
	template bool parseXMLAttrib( const XMLElemPtr& elem, const String& name, double& oValue		);


	template <>
	bool parseXMLAttrib(const XMLElemPtr& elem, const String& name, Math::float3& oValue)
	{
		if( const XMLElemPtr& subElem = elem->GetChild(name) )
		{
			bool res = 
					parseXMLAttrib(subElem, L"x", oValue.x);
			res &=	parseXMLAttrib(subElem, L"y", oValue.y);
			res &=	parseXMLAttrib(subElem, L"z", oValue.z);
			return res;
		}
		return false;
	}

	//------------------------------------------------------------------------

	template <>
	bool parseXMLAttrib(const XMLElemPtr& elem, const String& name, Math::float4& oValue)
	{
		if(const XMLElemPtr& subElem = elem->GetChild( name ) )
		{
			bool res = 
				parseXMLAttrib(subElem, L"x", oValue.x);
			res &=	parseXMLAttrib(subElem, L"y", oValue.y);
			res &=	parseXMLAttrib(subElem, L"z", oValue.z);
			res &=	parseXMLAttrib(subElem, L"w", oValue.w);
			return res;
		}
		return false;
	}

	//------------------------------------------------------------------------
	template <>
	bool parseXMLAttrib( const XMLElemPtr& elem, const String& name, XIQuatHelper& oValue )
	{
		if( const XMLElemPtr& subElem = elem->GetChild(name) )
		{
			Math::float3 axis;
			float angle;
			bool res = 
					parseXMLAttrib(subElem, L"x", axis.x);
			res &=	parseXMLAttrib(subElem, L"y", axis.y);
			res &=	parseXMLAttrib(subElem, L"z", axis.z);

			res &= parseXMLAttrib(subElem, L"angle", angle);

			angle *= 3.14159265358979323846f / 180;

			static_cast<Math::float4&>( oValue ) = Math::quatRotAxisAngle( axis, angle );
			return res;
		}
		return false;
	}
}

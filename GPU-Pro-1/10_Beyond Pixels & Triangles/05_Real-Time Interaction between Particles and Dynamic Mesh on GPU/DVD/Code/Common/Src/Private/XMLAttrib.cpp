#include "Precompiled.h"
#include "XMLAttrib.h"
#include "XMLAttribConfig.h"

#define MD_NAMESPACE XMLAttribNS
#include "ConfigurableImpl.cpp.h"

namespace Mod
{
	template class XMLAttribNS::ConfigurableImpl< XMLAttribConfig >;

	XMLAttrib::XMLAttrib( const XMLAttribConfig& cfg ) :
	Parent( cfg ),
	Named( cfg.name )
	{

	}

	//------------------------------------------------------------------------

	XMLAttrib::~XMLAttrib()
	{

	}

	//------------------------------------------------------------------------

	String
	XMLAttrib::GetStringValue() const
	{
		return GetConfig().value;
	}

	//------------------------------------------------------------------------

	INT32
	XMLAttrib::GetIntValue() const
	{
		const String& s = GetConfig().value;
		return _wtoi( s.c_str() );	
	}

	//------------------------------------------------------------------------

	float
	XMLAttrib::GetFloatValue() const
	{
		return (float)GetDoubleValue();
	}

	//------------------------------------------------------------------------

	double
	XMLAttrib::GetDoubleValue() const
	{
		const String& s = GetConfig().value;
		return _wtof( s.c_str() );	
	}

	//------------------------------------------------------------------------

	template <>
	INT32 XMLAttrib::GetValue() const
	{
		return GetIntValue();
	}

	//------------------------------------------------------------------------

	template <>
	float XMLAttrib::GetValue() const
	{
		return GetFloatValue();
	}

	//------------------------------------------------------------------------

	template <>
	double XMLAttrib::GetValue() const
	{
		return GetDoubleValue();
	}

	//------------------------------------------------------------------------

	template <>
	String XMLAttrib::GetValue() const
	{
		return GetStringValue();
	}

	//------------------------------------------------------------------------

	template <>
	UINT32 XMLAttrib::GetValue() const
	{
		return GetIntValue();
	}





}
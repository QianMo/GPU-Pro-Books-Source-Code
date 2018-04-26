/*
**********************************************************************
 * Demo program for
 * Rule-based Geometry Synthesis in Real-time
 * ShaderX 8 article.
 *
 * @author: Milan Magdics
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted for any non-commercial programs.
 * 
 * Use it for your own risk. The author(s) do(es) not take
 * responsibility or liability for the damages or harms caused by
 * this software.
**********************************************************************
*/

#include "DXUT.h"
#include "GrammarBasicTypes.h"

bool stringToAttributeType( const String& str, AttributeType& attributeType )
{
	if ( str == "float" )
	{
		attributeType = ATTRIBUTE_TYPE_FLOAT;
		return true;
	}
	if ( str == "float2" )
	{
		attributeType = ATTRIBUTE_TYPE_FLOAT2;
		return true;
	}
	if ( str == "float3" )
	{
		attributeType = ATTRIBUTE_TYPE_FLOAT3;
		return true;
	}
	if ( str == "float4" )
	{
		attributeType = ATTRIBUTE_TYPE_FLOAT4;
		return true;
	}
	if ( str == "int" )
	{
		attributeType = ATTRIBUTE_TYPE_INT;
		return true;
	}
	if ( str == "uint" )
	{
		attributeType = ATTRIBUTE_TYPE_UINT;
		return true;
	}

	return false;
}

String attributeTypeToString( const AttributeType& attributeType )
{
	if (  ATTRIBUTE_TYPE_FLOAT == attributeType  )
	{
		return "float";		
	}
	if (  ATTRIBUTE_TYPE_FLOAT2 == attributeType  )
	{
		return "float2";		
	}
	if (  ATTRIBUTE_TYPE_FLOAT3 == attributeType  )
	{
		return "float3";		
	}
	if (  ATTRIBUTE_TYPE_FLOAT4 == attributeType  )
	{
		return "float4";		
	}
	if (  ATTRIBUTE_TYPE_INT == attributeType  )
	{
		return "int";		
	}
	if (  ATTRIBUTE_TYPE_UINT == attributeType  )
	{
		return "uint";		
	}

	return "";
}

bool stringToRuleSelectionMethod( const String& str, RuleSelectionMethod& method )
{
	if ( str == "first" )
	{
		method = RULESELECTION_ALWAYSFIRST;
		return true;
	}
	if ( str == "random" || str == "stochastic" )
	{
		method = RULESELECTION_RANDOM;
		return true;
	}
	if ( str == "lod" )
	{
		method = RULESELECTION_LEVEL_OF_DETAIL;
		return true;
	}
	if ( str == "condition" )
	{
		method = RULESELECTION_CONDITION;
		return true;
	}
	return false;
}

String ruleSelectionMethodToString( const RuleSelectionMethod& method )
{
	if ( RULESELECTION_ALWAYSFIRST == method )
	{
		return "first";
	}
	if ( RULESELECTION_RANDOM == method )
	{
		return "random";
	}
	if ( RULESELECTION_LEVEL_OF_DETAIL == method )
	{
		return "lod";
	}
	if ( RULESELECTION_CONDITION == method )
	{
		return "condition";
	}
	return "";
}

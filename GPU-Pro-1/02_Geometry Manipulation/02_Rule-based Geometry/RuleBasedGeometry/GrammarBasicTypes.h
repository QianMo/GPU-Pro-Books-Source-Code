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

#pragma once

#pragma warning(disable:4995)		// disable warnings indicating deprecated codes in cstdio
#include <vector>
#include <string>
#include <map>
#pragma warning(default:4995)

// use your favourite string type
typedef std::string String;
typedef std::wstring WString;

typedef String SymbolName;
typedef std::vector<SymbolName> SymbolNameVector;   // a sequence of Symbol(name)s is represented as a vector

typedef String OperatorName;

typedef String AttributeName;                       // module attribute name
enum AttributeType                                  // module attribute type name
{
	ATTRIBUTE_TYPE_FLOAT,
	ATTRIBUTE_TYPE_FLOAT2,
	ATTRIBUTE_TYPE_FLOAT3,
	ATTRIBUTE_TYPE_FLOAT4,
	ATTRIBUTE_TYPE_UINT,
	ATTRIBUTE_TYPE_INT
};
// conversion functions for grammar parsing and code generation
bool stringToAttributeType( const String& str, AttributeType& attributeType );
String attributeTypeToString( const AttributeType& attributeType );
// set of attributes is implemented as a map to support searching for a specific name
typedef std::map<AttributeName,AttributeType> AttributeSet;
typedef unsigned int IDType;

typedef String MeshName;
typedef String RenderingTechniqueName;


// Rule Selection Methods
// - in contrast to (predefined) Operators, these are simple enums. 
//   this is beacuse it is harder to define a general interface for rule selection methods, than operators.
// - thus, for simplicity, we decided to implement rule selection methods as enum constants instead of 
//   objects of a class
// - however, since when we add a new method that requires parameters, we need to modify the Successor class,
//   thus, it would be better to define a general, abstract RuleSelectionMethod interface and inherit the specific
//   selection methods
enum RuleSelectionMethod
{
	RULESELECTION_ALWAYSFIRST,             // choose always the first
	RULESELECTION_RANDOM,                  // random (stochastic) selection based on probabilities
	RULESELECTION_LEVEL_OF_DETAIL,         // LoD-based selection
	RULESELECTION_CONDITION                // rules have a condition
};
// conversion functions for grammar parsing and code generation
bool stringToRuleSelectionMethod( const String& str, RuleSelectionMethod& method );
String ruleSelectionMethodToString( const RuleSelectionMethod& method );

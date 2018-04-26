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
#include <map>
#pragma warning(default:4995)

#include "GrammarBasicTypes.h"

////////////////////////////////////////////////////////////////////
// OperatorParameter struct - represents a parameter of an operator
////////////////////////////////////////////////////////////////////

struct OperatorParameter
{
	AttributeName name;
	AttributeType type;
	bool isRequired;

	OperatorParameter():isRequired(true) {}
};

typedef std::vector<OperatorParameter> OperatorParameterVector;
typedef std::map<AttributeName,unsigned int> OperatorParameterMap;


//////////////////////////////////////////////////////////////////////
// OperatorSignature struct - represents the signature of an operator
//////////////////////////////////////////////////////////////////////

struct OperatorSignature
{
	// we store parameters both in a vector (to determine their order) and a map (for faster search)
	// TODO: it would be much more clever to use indices only in the map :)
	OperatorParameterVector parameters;
	OperatorParameterMap parameterMap;

	// true if the return value of the implemented operator depends on the parent (predecessor) module
	bool useParent;

	OperatorSignature():useParent(false) {}

	void addParameter( const OperatorParameter& parameter ) 
	{ 
		parameters.push_back(parameter); 
		parameterMap[parameter.name] = parameters.size()-1;
	}
	unsigned int parameterNumber() const { return parameters.size(); }
	OperatorParameter& getParameter( unsigned int index ) { return parameters[index]; }
	OperatorParameter& getParameter( const AttributeName& attributeName ) { return getParameter(parameterMap[attributeName]); }
	bool hasParameter( const AttributeName& attributeName ) const { return parameterMap.count(attributeName) > 0; }
	unsigned int getParameterIndex( const AttributeName& attributeName ) { return parameterMap[attributeName]; }
	void clear() { parameters.clear(); parameterMap.clear(); }
};

/////////////////////////////////////////////////////////////////////
// ModuleOperation class 
// - represents operators (calls) with their actual parameter values
/////////////////////////////////////////////////////////////////////

const unsigned int maxParameters = 30;

class ModuleOperation
{
public:
	ModuleOperation(void);
	~ModuleOperation(void);

	String& getParameter( unsigned int index ) { return parameterValues[index]; }
	String& getAnimationMin( unsigned int index ) { return animationMin[index]; }
	String& getAnimationMax( unsigned int index ) { return animationMax[index]; }
	String& getPeriodLength( unsigned int index ) { return period_length[index]; }
	String& getRandomMin( unsigned int index ) { return randomMin[index]; }
	String& getRandomMax( unsigned int index ) { return randomMax[index]; }

public:
	OperatorName operatorName;
	bool animated;
	bool randomized;
protected:
	String parameterValues[maxParameters];
	String animationMin[maxParameters];
	String animationMax[maxParameters];
	String period_length[maxParameters];
	String randomMin[maxParameters];
	String randomMax[maxParameters];
};

// a sequence of operators is represented as a vector
typedef std::vector<ModuleOperation> ModuleOperationVector;

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
#include <map>
#pragma warning(default:4995)

#include "CodeGenerator.h"

//////////////////////////////////////////////////////////
// ShaderVariableType struct - a variable in shader codes
//////////////////////////////////////////////////////////

struct ShaderVariableType
{
	CodeGenerator::TypeName typeName;      // type name in the shader code
	String defaultValue;                   // given as a String, since it is used in code generation
	unsigned int component_number;         // component number for vector types (should be between 1-4)
};

//////////////////////////////////////////////////////////////////////////////
// ShaderCodeGenerator class - abstract base class for shader code generation
//////////////////////////////////////////////////////////////////////////////

class ShaderCodeGenerator : 
	public CodeGenerator
{
public:
	// attribute types used in the grammar description have to be mapped to shader types
	typedef std::map<AttributeType, ShaderVariableType> TypeMap;	
public:
	ShaderCodeGenerator(GrammarDescriptor *aGrammarDesc);
	virtual ~ShaderCodeGenerator(void);

	virtual void generateCode();		            // code generation
protected:
	// generate different parts of the shader code
	virtual void createModuleTypes() = 0;
	virtual void createSymbolIDs() = 0;
	virtual void createRuleInfos() = 0;
	virtual void createRuleManagement() = 0;
	virtual void createRules() = 0;

	virtual void buildTypeMap() = 0;               // builds up the type map
	void addType(const AttributeType &attributeType, const CodeGenerator::TypeName &typeName,
		const String &defaultValue, unsigned int component_number);
protected:
	TypeMap typeMap;                               // maps grammar types to shader types
};

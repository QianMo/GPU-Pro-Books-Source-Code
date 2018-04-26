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
#include "ShaderCodeGenerator.h"

ShaderCodeGenerator::ShaderCodeGenerator(GrammarDescriptor *aGrammarDesc):CodeGenerator(aGrammarDesc)
{
}

ShaderCodeGenerator::~ShaderCodeGenerator(void)
{
}

void ShaderCodeGenerator::generateCode()
{
	openFiles();

	createModuleTypes();
	createSymbolIDs();
	createRuleInfos();
	createRuleManagement();
	createRules();

	closeFiles();
}

void ShaderCodeGenerator::addType(const AttributeType &attributeType, const CodeGenerator::TypeName &typeName,
    const String &defaultValue, unsigned int component_number)
{
  typeMap[attributeType].typeName = typeName;
  typeMap[attributeType].defaultValue = defaultValue;
  typeMap[attributeType].component_number = component_number;
}

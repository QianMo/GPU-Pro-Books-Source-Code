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
#include "shadercodegenerator.h"

///////////////////////////////////////////////////////////////////////////////////////
// HLSLCodeGenerator struct - generates HLSL shader codes from an L-system description
///////////////////////////////////////////////////////////////////////////////////////

class HLSLCodeGenerator :
	public ShaderCodeGenerator
{
public:
	HLSLCodeGenerator(GrammarDescriptor *aGrammarDesc);
	virtual ~HLSLCodeGenerator(void);

	const static unsigned int MAX_ROW_LENGTH;    // row length limit in the generated code (used for fragmentation)
protected:
	void openFiles();

	// generate different parts of the shader code
	void createModuleTypes();
	void createSymbolIDs();
	void createRuleInfos();
	void createRuleManagement();
	void createRules();
	void createOperation(OutFile &output, ModuleOperation& operation);

	// adds animation (a time dependent term) to the operation parameter
	void addAnimationString(String &parameterValue,unsigned int parameterIndex,OperatorSignature& signature,
		ModuleOperation& operation);
	// adds random term to the operation parameter
	void addRandomString(String &parameterValue,unsigned int parameterIndex,OperatorSignature& signature,
		ModuleOperation& operation);

	void buildTypeMap();                         // builds up the type map
};

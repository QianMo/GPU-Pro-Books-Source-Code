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
#include "CodeGenerator.h"

///////////////////////////////////////////////////////////////////////////
// ShaderCodeGenerator class - abstract base class for CPU code generation
///////////////////////////////////////////////////////////////////////////

class CPUCodeGenerator :
	public CodeGenerator
{
public:
	CPUCodeGenerator(GrammarDescriptor *aGrammarDesc);
	virtual ~CPUCodeGenerator(void);

	virtual void generateCode();		            // code generation

	// creates a fingerprint of the generated CPU Module type
	// - it can be used to decide whether recompilation of the whole
	//   project is needed
	virtual String moduleTypeFingerprint() const = 0;
protected:
	// generate different parts of the shader code
	virtual void createModuleTypes() = 0;
	virtual void createSymbolIDs() = 0;

	virtual void buildTypeMap() = 0;               // builds up the type map
};

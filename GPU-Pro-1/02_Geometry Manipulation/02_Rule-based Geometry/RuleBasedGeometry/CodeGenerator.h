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
#include <fstream>
#include <vector>
#pragma warning(default:4995)

#include "GrammarBasicTypes.h"
#include "ModuleOperation.h"

//////////////////////////////////////////////////////////////////////////////////////////////
// CodeGenerator class - abstract base class for code generation from an L-system description
//////////////////////////////////////////////////////////////////////////////////////////////

class GrammarDescriptor;

class CodeGenerator
{
public:
	// use your favourite types here
	typedef std::ofstream OutFile;
	typedef std::vector<OutFile*> OutFileVector;
	typedef String TypeName;
	typedef std::map<OperatorName,OperatorSignature> OperatorMap;
public:
	CodeGenerator(GrammarDescriptor *aGrammarDesc);
	virtual ~CodeGenerator(void);

	virtual void generateCode() = 0;        // code generation

	const static IDType SYMBOLID_INVALID;
	const static IDType SYMBOLID_START;
	const static IDType RULEID_START;
	const static IDType RULEID_NORULE;

	// in shader generation, predefined attributes have special role
	// - this guarantees e.g. that implemented operators use the same position attribute etc.
	// - note that only symbol id is a compulsory attribute, e.g. position is optional (also it is not 
	//   added automatically). we made this decision to, for example, allow to use this code for general, 
	//   non-graphics-related L-system applications
	static AttributeSet predefinedAttributes;
	static bool isPredefinedAttribute(const AttributeName &attributeName);
	static bool getType(const AttributeName &attributeName, AttributeType &type);

	// to allow code generation for operators, we have to define the set of predefined operators
	static OperatorMap predefinedOperators;
	static bool isPredefinedOperator(const OperatorName &operatorName);
	static bool getSignature(const OperatorName &operatorName, OperatorSignature &signature);

	// creates and initializes predefined attributes and operators
	// NOTE: this should be called before using any shader generator or grammar loader object!
	static void initPredefined();
protected:
	static void initPredefinedAttributes();
	static void initPredefinedOperators();

	virtual void openFiles() = 0;           // open output files for writing
	void closeFiles();                      // close (save) output files

	OutFileVector outputs;                  // output file handlers
	GrammarDescriptor *grammarDesc;         // pointer to the L-system descriptor
};

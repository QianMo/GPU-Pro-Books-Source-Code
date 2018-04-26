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
#include "cpucodegenerator.h"

//////////////////////////////////////////////////////////////////
// DXCPPVariableType struct - a variable in C++ with DirectX 10.0 
//////////////////////////////////////////////////////////////////

struct DXCPPVariableType
{
	CodeGenerator::TypeName typeName;      // type name in the cpp code
	String defaultValue;                   // given as a String, since it is used in code generation
	DXGI_FORMAT dxgiFormat;                // used in DX input layout generation
	unsigned int component_number;         // component number for vector types (should be between 1-4)
	unsigned int size;                     // size in bytes - used for DX input layout generation etc.
};

////////////////////////////////////////////////////////////////////////////////////////////////////////
// DXCPPCodeGenerator class - generates CPU code for C++ with DirectX 10.0 from an L-system description
////////////////////////////////////////////////////////////////////////////////////////////////////////

class DXCPPCodeGenerator :
	public CPUCodeGenerator
{
public:
	// attribute types used in the grammar description have to be mapped to shader types
	typedef std::map<AttributeType, DXCPPVariableType> TypeMap;
	// LayoutDescriptor stores an input layout description used by the DX10 input assembler
	// - an additional String is added to store the semantic name parameter, since string it is more comfortable
	//   to handle than the LPCSTR type (it would need strcpy, dynamic allocation and free... too much work :))
	// - this implies that the SemanticName parameter of the D3D10_INPUT_ELEMENT_DESC is uninitialized 
	//   (do not use it!)
	typedef std::vector<std::pair<D3D10_INPUT_ELEMENT_DESC,String> > LayoutDescriptor;
public:
	DXCPPCodeGenerator(GrammarDescriptor *aGrammarDesc);
	virtual ~DXCPPCodeGenerator(void);

	// creates the D3D10_INPUT_ELEMENT_DESC descriptions used in generation/sorting and instancing steps
	void createInputLayouts(LayoutDescriptor& generationLayout, LayoutDescriptor& instancingLayout);

	virtual String moduleTypeFingerprint() const;
protected:
	// adds a new DXCPPVariableType type to the current set
	void addType(const AttributeType &attributeType, const CodeGenerator::TypeName &typeName,
		const String &defaultValue, const DXGI_FORMAT &dxgiFormat, unsigned int component_number,
		unsigned int size);

	void openFiles();

	// generate different parts of the shader code
	void createModuleTypes();
	                                       // currently this is not used. call it if you want to use the symbol names
	                                       //   as pre-compiler constants
	void createSymbolIDs();                // NOTE: GrammarDescriptor::symbolIDMap does the same mapping (in run-time)

	void buildTypeMap();                   // builds up the type map
protected:
	TypeMap typeMap;
};

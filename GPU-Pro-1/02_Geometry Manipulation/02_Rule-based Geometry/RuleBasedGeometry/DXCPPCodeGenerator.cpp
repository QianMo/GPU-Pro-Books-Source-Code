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
#include "DXCPPCodeGenerator.h"

#include <algorithm>

#include "ErrorMessages.h"
#include "GrammarSymbol.h"
#include "GrammarDescriptor.h"

// constant parameters
const unsigned int INDEX_SYMBOLID_FILE = 0;
const unsigned int INDEX_MODULETYPE_H_FILE = INDEX_SYMBOLID_FILE + 1;
const unsigned int INDEX_MODULETYPE_CPP_FILE = INDEX_MODULETYPE_H_FILE + 1;
const unsigned int OUTPUT_FILE_NUM = INDEX_MODULETYPE_CPP_FILE + 1;

const String OUTFILENAME_SYMBOLID = "symbolids.h";
const String OUTFILENAME_MODULETYPE_H = "Module.h";
const String OUTFILENAME_MODULETYPE_CPP = "Module.cpp";


DXCPPCodeGenerator::DXCPPCodeGenerator(GrammarDescriptor *aGrammarDesc):CPUCodeGenerator(aGrammarDesc)
{
	buildTypeMap();
}

DXCPPCodeGenerator::~DXCPPCodeGenerator(void)
{
}

void DXCPPCodeGenerator::createInputLayouts(LayoutDescriptor& generationLayout, LayoutDescriptor& instancingLayout)
{
	generationLayout.clear();
	instancingLayout.clear();

	// adding the the description of the ID attribute
	AttributeType IDType;
	getType("symbolID",IDType);
	String typeString = typeMap[IDType].typeName;
	D3D10_INPUT_ELEMENT_DESC idDesc =
		{ NULL, 0, typeMap[IDType].dxgiFormat, 0, 0, D3D10_INPUT_PER_VERTEX_DATA, 0 };
	generationLayout.push_back( std::pair<D3D10_INPUT_ELEMENT_DESC,String> (idDesc,"ID") );
	int offset = typeMap[IDType].size;

	GrammarDescriptor::ModuleType &moduleType = grammarDesc->getModuleType();
	String semanticName;
	for ( GrammarDescriptor::ModuleType::iterator it = moduleType.begin(); it != moduleType.end(); ++it )
	{
		semanticName = it->first;
		transform( it->first.begin(), it->first.end(), semanticName.begin(), toupper);
		D3D10_INPUT_ELEMENT_DESC vDesc =
			{ NULL, 0, typeMap[it->second].dxgiFormat, 0, offset, D3D10_INPUT_PER_VERTEX_DATA, 0 };
		generationLayout.push_back( std::pair<D3D10_INPUT_ELEMENT_DESC,String> (vDesc,semanticName) );

		// POSITION semantic is renamed to INSTANCE_POSITION in the sorted type
		if ( it->first == "position" )
		{
			semanticName = "INSTANCE_POSITION";
		}

		vDesc.InputSlotClass = D3D10_INPUT_PER_INSTANCE_DATA;
		vDesc.InputSlot = 1;
		vDesc.InstanceDataStepRate = 1;
		  // NOTE: if you remove more variables from the SortedModule type, reduce the offset
		vDesc.AlignedByteOffset -= typeMap[IDType].size;
		instancingLayout.push_back( std::pair<D3D10_INPUT_ELEMENT_DESC,String> (vDesc,semanticName) );
		offset += typeMap[it->second].size;
	}

}

String DXCPPCodeGenerator::moduleTypeFingerprint() const
{
	String fingerPrint;

	GrammarDescriptor::ModuleType& moduleType = grammarDesc->getModuleType();
	for ( GrammarDescriptor::ModuleType::const_iterator type_it = moduleType.begin();
		type_it != moduleType.end(); ++type_it )
	{
		fingerPrint += type_it->first + attributeTypeToString( type_it->second );
	}

	return fingerPrint;
}

void DXCPPCodeGenerator::openFiles()
{
	// initializing output file names
	std::vector<String> outputNames(OUTPUT_FILE_NUM);
	
	outputNames[INDEX_SYMBOLID_FILE] = OUTFILENAME_SYMBOLID;
	outputNames[INDEX_MODULETYPE_H_FILE] = OUTFILENAME_MODULETYPE_H;
	outputNames[INDEX_MODULETYPE_CPP_FILE] = OUTFILENAME_MODULETYPE_CPP;

	for ( int i = 0; i < OUTPUT_FILE_NUM; ++i ) 
	{
		OutFile *output = new OutFile;
		outputs.push_back( output );
		output->open( outputNames[i].c_str() );
		if ( output->fail() )
		{
			ERROR_MSG( "Cannot create output file", 
				TO_C_STRING("Shader generation error: Cannot create output file: " + outputNames[i] + "!"));
			exit(1);
		}
	}
}

void DXCPPCodeGenerator::createModuleTypes()
{
	OutFile &hOutput = (*outputs[INDEX_MODULETYPE_H_FILE]);
	OutFile &cppOutput = (*outputs[INDEX_MODULETYPE_CPP_FILE]);
	String cppModuleConstructor = "Module( ";
	String cppSortedConstructor = "SortedModule( ";
	String cppModuleConstructorInit;
	String cppSortedConstructorInit;
	String cppModuleConstructorDefaultInit;
	String cppSortedConstructorDefaultInit;

	String header;
	header = (String)"////////////////////////////////////////////////////\n"
	       + "// Auto-generated code from the L-system description\n"
	       + "////////////////////////////////////////////////////\n\n";
	hOutput << header;
	cppOutput << header;

	AttributeType attType;
	getType("symbolID",attType);
	String typeString = typeMap[attType].typeName;
	hOutput	<< "#pragma once" << std::endl << std::endl
			<< "#include \"GrammarBasicTypes.h\"" << std::endl << std::endl
			<< "typedef " << typeString << " IDType;\t\t\t// Type of Symbol ID-s" << std::endl << std::endl
			<< "//*********************************************" << std::endl
			<< "// Struct of a module for the generation phase" << std::endl
			<< "//*********************************************" << std::endl << std::endl
			<< "struct Module" << std::endl
			<< "{" << std::endl;

	// compulsory parameters (symbol ID)
	hOutput << "\t" << typeString << " symbolID;" << std::endl;
	cppModuleConstructor += typeString + " symbolID_";
	cppModuleConstructorInit += "symbolID(symbolID_)";
	cppModuleConstructorDefaultInit += "symbolID(" + typeMap[attType].defaultValue + ")";

	// additional parameters
	String typeName;
	bool first = true;			// yes, im noob.
	GrammarDescriptor::ModuleType moduleType = grammarDesc->getModuleType();
	for ( GrammarDescriptor::ModuleType::iterator it = moduleType.begin(); it != moduleType.end(); ++it )
	{
		String attributeName = it->first;
		attType = it->second;
		typeString = typeMap[attType].typeName;

		if ( first )
		{
			first = false;
		}
		else
		{
			cppSortedConstructor += ", ";
			cppSortedConstructorInit += ",";
			cppSortedConstructorDefaultInit += ",";
		}

		hOutput << "\t" << typeString << " " << attributeName << ";" << std::endl;
		cppModuleConstructor += String(", ") + typeString + " " + attributeName + "_";
		cppSortedConstructor += typeString + " " + attributeName + "_";
		cppModuleConstructorInit += String(",") + attributeName + "(" + attributeName + "_)";
		cppSortedConstructorInit += attributeName + "(" + attributeName + "_)";
		cppModuleConstructorDefaultInit += String(",") + attributeName + "(" + typeMap[attType].defaultValue + ")";
		cppSortedConstructorDefaultInit += attributeName + "(" + typeMap[attType].defaultValue + ")";
	}

	cppModuleConstructor += " )";
	cppSortedConstructor += " )";

	hOutput << std::endl << "\t" << cppModuleConstructor << ";" << std::endl;
	// default constructor
	hOutput << "\tModule();" << std::endl;
	// general attribute setter function
	hOutput << "\tvoid setAttribute( const String& attributeName, const String& value );" << std::endl;
	hOutput << "\tstatic String moduleTypeFingerprint() { return \"" << moduleTypeFingerprint() << "\"; }" << std::endl;
	hOutput << "};" << std::endl;

	// SortedModule definition creation
	hOutput << std::endl 
			<< "//***********************************" << std::endl
			<< "// Sorted module, used for instancing" << std::endl
			<< "//***********************************" << std::endl << std::endl
			<< "struct SortedModule" << std::endl
			<< "{" << std::endl;

	for ( GrammarDescriptor::ModuleType::iterator it = moduleType.begin(); it != moduleType.end(); ++it )
	{
		String attributeName = it->first;
		attType = it->second;
		typeString = typeMap[attType].typeName;

		hOutput << "\t" << typeString << " " << attributeName << ";" << std::endl;
	}
	hOutput << std::endl << "\t" << cppSortedConstructor << ";" << std::endl;
	// default constructor
	hOutput << "\tSortedModule();" << std::endl;
	hOutput << "};" << std::endl;

	// cpp code generation
	cppOutput	<< "#include \"DXUT.h\"" << std::endl
				<< "#include \"" << OUTFILENAME_MODULETYPE_H << "\"" << std::endl << std::endl
				<< "#include <sstream>" << std::endl << std::endl
				<< "//*****************" << std::endl
				<< "// Module functions" << std::endl
				<< "//*****************" << std::endl << std::endl
				<< "Module::" << cppModuleConstructor << ":" << cppModuleConstructorInit << std::endl
				<< "{" << std::endl
				<< "}" << std::endl << std::endl;
	// default constructor
	cppOutput	<< "Module::Module():" << cppModuleConstructorDefaultInit << std::endl
				<< "{" << std::endl
				<< "}" << std::endl << std::endl;

	// general attribute setter function
	cppOutput	<< "void Module::setAttribute( const String& attributeName, const String& value )" << std::endl
				<< "{" << std::endl
				<< "\tstd::istringstream sstr(value);" << std::endl
				<< "\tif( attributeName == \"symbolID\" )" << std::endl
				<< "\t{" << std::endl
				<< "\t\tsstr >> symbolID;" << std::endl
				<< "\t}" << std::endl;

	for ( GrammarDescriptor::ModuleType::iterator it = moduleType.begin(); it != moduleType.end(); ++it )
	{
		AttributeName attributeName = it->first;
		AttributeType attributeType = it->second;
		cppOutput	<< "\tif( attributeName == \"" << attributeName << "\" )" << std::endl
					<< "\t{" << std::endl;
		unsigned int component_number = typeMap[attributeType].component_number;
		if ( 1 == component_number )
			cppOutput	<< "\t\tsstr >> " << attributeName << ";" << std::endl
						<< "\t}" << std::endl;
		else
		{
			const String component[4] = {"x","y","z","w"};
			if ( attributeName == "position" ) --component_number;
			for ( unsigned int i = 0; i < component_number; ++i )
			{
				cppOutput << "\t\tsstr >> " << attributeName << "." << component[i] << ";" << std::endl;
							
			}
			if ( attributeName == "position" )
				cppOutput << "\t\t" << attributeName << "." << component[3] << " = 1.0F;" << std::endl;
			cppOutput << "\t}" << std::endl;
		}
					
	}

	cppOutput	<< "}" << std::endl
				<< std::endl << std::endl;

	// default constructor
	cppOutput	<< "//***********************" << std::endl
				<< "// SortedModule functions" << std::endl
				<< "//***********************" << std::endl << std::endl
				<< "SortedModule::" << cppSortedConstructor << ":" << cppSortedConstructorInit << std::endl
				<< "{" << std::endl
				<< "}" << std::endl << std::endl
				<< "SortedModule::SortedModule():" << cppSortedConstructorDefaultInit << std::endl
				<< "{" << std::endl
				<< "}" << std::endl << std::endl;
}

void DXCPPCodeGenerator::createSymbolIDs()
{
	OutFile &output = (*outputs[INDEX_SYMBOLID_FILE]);
	output	<< "#include \"" << OUTFILENAME_MODULETYPE_H << "\"" << std::endl << std::endl;

	SymbolVector &symbols = grammarDesc->getSymbols();

	for ( unsigned int i = 0; i < symbols.size(); ++i )
	{
		output << "#define " << symbols[i].name << "\t\tIDType(" << (SYMBOLID_START+i) << ")" << std::endl;
	}
}

void DXCPPCodeGenerator::buildTypeMap()
{
	addType(ATTRIBUTE_TYPE_FLOAT,"float","0",DXGI_FORMAT_R32_FLOAT,1,4);
	addType(ATTRIBUTE_TYPE_FLOAT2,"D3DXVECTOR2","D3DXVECTOR2(0.0F,0.0F)",DXGI_FORMAT_R32G32_FLOAT,2,8);
	addType(ATTRIBUTE_TYPE_FLOAT3,"D3DXVECTOR3","D3DXVECTOR3(0.0F,0.0F,0.0F)",DXGI_FORMAT_R32G32B32_FLOAT,3,12);
	addType(ATTRIBUTE_TYPE_FLOAT4,"D3DXVECTOR4","D3DXVECTOR4(0.0F,0.0F,0.0F,0.0F)",DXGI_FORMAT_R32G32B32A32_FLOAT,4,16);
	addType(ATTRIBUTE_TYPE_INT,"int","0",DXGI_FORMAT_R32_SINT,1,4);
	addType(ATTRIBUTE_TYPE_UINT,"unsigned int","0",DXGI_FORMAT_R32_UINT,1,4);
}

void DXCPPCodeGenerator::addType(const AttributeType &attributeType, const CodeGenerator::TypeName &typeName,
		const String &defaultValue, const DXGI_FORMAT &dxgiFormat, unsigned int component_number,
		unsigned int size)
{
	typeMap[attributeType].typeName = typeName;
	typeMap[attributeType].defaultValue = defaultValue;
	typeMap[attributeType].dxgiFormat = dxgiFormat;
	typeMap[attributeType].component_number = component_number;
	typeMap[attributeType].size = size;
}

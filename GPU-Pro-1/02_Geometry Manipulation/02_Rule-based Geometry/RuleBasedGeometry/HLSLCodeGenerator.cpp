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
#include "HLSLCodeGenerator.h"

#include <algorithm>
#include <sstream>

#include "GrammarBasicTypes.h"
#include "ErrorMessages.h"
#include "GrammarSymbol.h"
#include "GrammarDescriptor.h"


// constant parameters
const unsigned int INDEX_SYMBOLID_FILE = 0;
const unsigned int INDEX_RULEINFO_FILE = INDEX_SYMBOLID_FILE + 1;
const unsigned int INDEX_RULE_FILE = INDEX_RULEINFO_FILE + 1;
const unsigned int INDEX_RULEMANAGEMENT_FILE = INDEX_RULE_FILE + 1;
const unsigned int INDEX_MODULETYPE_FILE = INDEX_RULEMANAGEMENT_FILE + 1;
const unsigned int OUTPUT_FILE_NUM = INDEX_MODULETYPE_FILE + 1;

const String OUTFILENAME_SYMBOLID = "symbolids.fx";
const String OUTFILENAME_RULEINFO = "ruleinfos.fx";
const String OUTFILENAME_RULE = "rules.fx";
const String OUTFILENAME_RULEMANAGEMENT = "rulemanagement.fx";
const String OUTFILENAME_MODULETYPE = "modules.fx";

HLSLCodeGenerator::HLSLCodeGenerator(GrammarDescriptor *aGrammarDesc):ShaderCodeGenerator(aGrammarDesc)
{
  buildTypeMap();
}

HLSLCodeGenerator::~HLSLCodeGenerator(void)
{
}

void HLSLCodeGenerator::openFiles()
{
	// initializing output file names
	std::vector<String> outputNames(OUTPUT_FILE_NUM);
	
	outputNames[INDEX_SYMBOLID_FILE] = OUTFILENAME_SYMBOLID;
	outputNames[INDEX_RULEINFO_FILE] = OUTFILENAME_RULEINFO;
	outputNames[INDEX_RULE_FILE] = OUTFILENAME_RULE;
	outputNames[INDEX_RULEMANAGEMENT_FILE] = OUTFILENAME_RULEMANAGEMENT;
	outputNames[INDEX_MODULETYPE_FILE] = OUTFILENAME_MODULETYPE;


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

void HLSLCodeGenerator::createModuleTypes()
{
	OutFile &fxOutput = (*outputs[INDEX_MODULETYPE_FILE]);

	// contains the arguments needed in ConstructGSWithSO
	String generationStreamOutputArgument = "ID.x";
	String sortingStreamOutputArgument;

	fxOutput	<< "#ifndef __GR_MODULES_FX" << std::endl
				<< "#define __GR_MODULES_FX" << std::endl << std::endl;

	// the following few "if"-s are for generating necessary typedefs if builtin types are used
	// NOTE: it would be more clever to iteraton on every predefined attribute that is set instead of
	//   write an if for each of them :)
	if ( grammarDesc->isAttributeDefined("position") )
	{
		fxOutput << "#define " << "__GR_POS_DEFINED" << std::endl << std::endl;
	}
	if ( grammarDesc->isAttributeDefined("size") )
	{
		fxOutput << "#define " << "__GR_SIZE_DEFINED"  << std::endl << std::endl;
	}
	if ( grammarDesc->isAttributeDefined("size3") )
	{
		fxOutput << "#define " << "__GR_SIZE3_DEFINED"  << std::endl << std::endl;
	}
	if ( grammarDesc->isAttributeDefined("orientation") )
	{
		fxOutput << "#define " << "__GR_ORIENTATION_DEFINED"  << std::endl << std::endl;
	}
	if ( grammarDesc->isAttributeDefined("terminated") )
	{
		fxOutput << "#define " << "__GR_TERMINATION_DEFINED"  << std::endl << std::endl;
	}

	// this is always generated (every module has a symbolID)
	AttributeType attType;
	getType("symbolID",attType);
	String typeString = typeMap[attType].typeName;
	String idTypeString = typeString;
	fxOutput	<< "typedef " << typeString << " IDType;" << std::endl;
	
	// same as the "if"-s above
	getType("position",attType);
	typeString = typeMap[attType].typeName;
	if ( grammarDesc->isAttributeDefined("position") )
	{
		fxOutput << "typedef " << typeString << " PositionType;" << std::endl << std::endl;
	}

	// struct definition
	fxOutput	<< std::endl
				<< "// Represents a module of the grammar and its parameters" << std::endl
				<< "struct Module" << std::endl
				<< "{" << std::endl;
	// compulsory parameters (e.g. symbol ID, we do not have more currently for allow GPGPU use)
	fxOutput << "\t" << idTypeString << " symbolID : ID;" << std::endl;

	// NOTE: modify this, if you want to save more memory after sorting. 
	// e.g. sorted modules do not need symbolID
	String sortedModuleStruct = 
		String("// Represents a module of the grammar after sorting and its parameters\n") + 
		"struct sortedModule\n" +
		"{\n";
	String convertFunction = 
		String("// convert method from Module to SortedModule\n") +
		"sortedModule convertToSorted( Module input )\n" +
		"{\n" +
		"\tsortedModule output;\n";

	// additional parameters
	bool first = true;
	GrammarDescriptor::ModuleType moduleType = grammarDesc->getModuleType();
	for ( GrammarDescriptor::ModuleType::iterator it = moduleType.begin(); it != moduleType.end(); ++it )
	{
		String attributeName = it->first;
		attType = it->second;
		typeString = typeMap[attType].typeName;

		// HLSL semantic simply the variable name transformed to upper case
		String tmp = attributeName;
		String semantic = tmp;
		transform( tmp.begin(), tmp.end(), semantic.begin(), toupper);

		fxOutput << "\t" << typeString << " " << attributeName << " : "<< semantic << ";" << std::endl;
		sortedModuleStruct += "\t" + typeString + " " + attributeName + " : " + semantic + ";\n";
		convertFunction += "\toutput." + attributeName + " = " + "input." + attributeName + ";\n";

		generationStreamOutputArgument += "; " + tmp + ".";
		if ( first ) 
		{
			first = false;
		}
		else
		{
			sortingStreamOutputArgument += "; ";
		}
		sortingStreamOutputArgument += tmp + ".";
		String componentString;
		switch( typeMap[attType].component_number )
		{
			case 1:
				componentString = "x";
				break;
			case 2: 
				componentString = "xy";
				break;
			case 3: 
				componentString = "xyz";
				break;
			case 4:
				componentString = "xyzw";
				break;
		}
		generationStreamOutputArgument += componentString;
		sortingStreamOutputArgument += componentString;
	}
	fxOutput << "};" << std::endl << std::endl;
	sortedModuleStruct += "};\n\n";
	fxOutput << sortedModuleStruct;
	convertFunction += 
		String("\treturn output;\n") +
		"}\n\n";
	fxOutput << convertFunction;

	fxOutput << "#define __GR_GENERATION_SO_ARG \t\"" << generationStreamOutputArgument << "\"" << std::endl;
	fxOutput << "#define __GR_SORTING_SO_ARG \t\"" << sortingStreamOutputArgument << "\"" << std::endl << std::endl;

	fxOutput << "#endif" << std::endl;
}

void HLSLCodeGenerator::createSymbolIDs()
{
	OutFile &output = (*outputs[INDEX_SYMBOLID_FILE]);
	output	<< "#include \"" << OUTFILENAME_MODULETYPE << "\"" << std::endl
			<< "#define __GR_SYMBOLID_START\tIDType(" << SYMBOLID_START << ")" << std::endl << std::endl
			<< "#define SYMBOL_INVALID\tIDType(" << SYMBOLID_INVALID << ")" << std::endl << std::endl;

	SymbolVector &symbols = grammarDesc->getSymbols();

	for ( unsigned int i = 0; i < symbols.size(); ++i )
	{
		output << "#define " << symbols[i].name << "\t\tIDType(" << (SYMBOLID_START+i) << ")" << std::endl;
	}
}

void HLSLCodeGenerator::createRuleInfos()
{
	OutFile &output = (*outputs[INDEX_RULEINFO_FILE]);
	String ruleIDs;				// rule IDs, which are calculated during iterating rule map should be at the beginning
								//  of the file -> we write everything to a buffer first (noob solution, but easy to implement)
	ruleIDs += "//********************************\n";
	ruleIDs += "// Rule IDs\n";
	ruleIDs += "//********************************\n\n";

	String ruleLengths;			// rule lengths. see comment for String ruleIDs
	ruleLengths += "//********************************\n";
	ruleLengths += "// Rule Lengths\n";
	ruleLengths += "//********************************\n\n";

	GrammarDescriptor::RuleMap &rules = grammarDesc->getRules();
	unsigned int maxSuccessorNumber = 0;
	unsigned int maxSuccessorLength = 0;
	for ( GrammarDescriptor::RuleMap::iterator map_it = rules.begin(); map_it != rules.end(); ++map_it )
	{
		String predecessor;
		predecessor = map_it->first.symbol;

		SuccessorVector &currentSuccessors = map_it->second;
		maxSuccessorNumber = max( (unsigned int)currentSuccessors.size(), maxSuccessorNumber );
		for ( unsigned int i = 0; i < currentSuccessors.size(); ++i )
		{
			char i_str[255];
			char ruleSize_str[255];
			_itoa_s( i+RULEID_START, i_str, 10 );
			_itoa_s( currentSuccessors[i].symbolNumber(), ruleSize_str, 10 );
			ruleIDs += "#define " + predecessor + "_RULE_" + i_str + "_ID\t\t(" + predecessor + " * MAX_SUCCESSOR_NUMBER + "
					+ i_str + ")\n";
			ruleLengths += "#define " + predecessor + "_RULE_" + i_str + "_LENGTH\t" + ruleSize_str + "\n";

			maxSuccessorLength = max( currentSuccessors[i].symbolNumber(), maxSuccessorLength );
		}

		ruleIDs += "\n";
		ruleLengths += "\n";
	}
	ruleIDs += "\n";
	ruleLengths += "\n";


	String ruleInfos;
	char buffer[255];
	_itoa_s( maxSuccessorLength, buffer, 10 );
	ruleInfos	+= (String)"#define MAX_SUCCESSOR_LENGTH\t" + buffer 
				+ "\t\t\t\t// Maximum number of symbols in a successor\n";
	_itoa_s( maxSuccessorNumber, buffer, 10 );
	ruleInfos	+= (String)"#define MAX_SUCCESSOR_NUMBER\t" + buffer 
				+ "\t\t\t\t// Maximum number of rules with the same predecessor\n";
	ruleInfos += "\n\n";


	output << ruleInfos << ruleIDs << ruleLengths;
}

void HLSLCodeGenerator::createRuleManagement()
{
	OutFile &output = (*outputs[INDEX_RULEMANAGEMENT_FILE]);

	const String ruleSelectionHeader = "( in Module module, out int successor_length, out int ruleID )";
	const String ruleSelectionCall = "( module, successor_length, ruleID )";
	const String moduleGenerationCall = "( parent_module, number )";

	// Header
	output	<< "#include \"" << OUTFILENAME_RULE << "\"" << std::endl
			<< "#include \"" << OUTFILENAME_RULEINFO << "\"" << std::endl << std::endl
			<< "//************************************************" << std::endl
			<< "// Rule selection algorithms for each predecessor" << std::endl
			<< "//	- Furthermore, these programs return the" << std::endl
			<< "//		successor length, since it can depend on" << std::endl
			<< "//		the module (e.g. a repetition rule)" << std::endl
			<< "//************************************************" << std::endl << std::endl;

	// Rule selection functions for each predecessor
	GrammarDescriptor::RuleMap &rules = grammarDesc->getRules();
	for ( GrammarDescriptor::RuleMap::iterator it = rules.begin(); it != rules.end(); ++it )
	{
		String predecessorStr;
		predecessorStr = it->first.symbol;
		output	<< "void selectRuleSymbol" << predecessorStr << ruleSelectionHeader << std::endl
				<< "{" << std::endl
				<< "\t// TODO: implement rule selection algorithm for predecessor " << predecessorStr << " here"
				<< std::endl << std::endl;

		// finding the grammar symbol for the predecessor (using a map would be better here)
		GrammarSymbol symbol;
		for ( SymbolVector::iterator symbol_it = grammarDesc->getSymbols().begin();
			symbol_it != grammarDesc->getSymbols().end(); ++symbol_it )
		{
			if ( symbol_it->name == it->first.symbol )
			{
				symbol = *symbol_it;
				break;
			}
		}
		
		// TODO: implement code generation for different selection strategies here
		SuccessorVector &successors = it->second;
		unsigned int ruleIndex = 1;
		float randomLowerBorder = 0.0F;
		// initializations in the rule selection function for a specific predecessor
		String conditionStr;
			switch( symbol.ruleSelection )
			{
			case RULESELECTION_ALWAYSFIRST: 
				// nothing to do here
				break;
			case RULESELECTION_RANDOM:
			{
				output << "\tfloat random = module_random(module);\n";
				break;
			}
			case RULESELECTION_LEVEL_OF_DETAIL:
				// nothing to do here
				break;
			case RULESELECTION_CONDITION:
				output << "\tfloat lod = module_pixelSize(module);\n";
				break;
			default:
				break;
			}
		// rule conditions
		for ( SuccessorVector::iterator succ_it = successors.begin(); succ_it != successors.end(); ++succ_it )
		{
			String conditionStr;
			switch( symbol.ruleSelection )
			{
			case RULESELECTION_ALWAYSFIRST:
			{
				conditionStr = "true";
				break;
			}
			case RULESELECTION_RANDOM:
			{
				std::ostringstream stringStr;
				stringStr << randomLowerBorder << " <= random && random < " << randomLowerBorder+succ_it->probability;
				conditionStr = stringStr.str();
				randomLowerBorder += succ_it->probability;
				break;
			}
			case RULESELECTION_LEVEL_OF_DETAIL:
			{
				std::ostringstream stringStr;
				stringStr << succ_it->lodMinPixels << " <= lod && lod < " << succ_it->lodMaxPixels;
				conditionStr = stringStr.str();
				break;
			}
			case RULESELECTION_CONDITION:
			{
				conditionStr = succ_it->condition;
				if ( "" == conditionStr ) conditionStr = "true";
				break;
			}
			default:
				break;
			}
			output  << "\t" << "if ( " << conditionStr << " )" << std::endl
					<< "\t{" << std::endl
					<< "\t\truleID = " << predecessorStr << "_RULE_" << ruleIndex << "_ID;" << std::endl
					<< "\t\tsuccessor_length = " << predecessorStr << "_RULE_" << 1 << "_LENGTH;" << std::endl
					<< "\t\treturn;" << std::endl
					<< "\t}" << std::endl;
			++ruleIndex;
		}

		// /TODO

		output  << "\truleID = 0;" << std::endl
		        << "\tsuccessor_length = 0;" << std::endl;

		output	<< "}" << std::endl << std::endl;
	}

 	// Rule Selection main function
	output	<< "//************************************************" << std::endl
			<< "// Rule selection main function" << std::endl
			<< "//	- calls the rule selection function of the" << std::endl
			<< "//		proper predecessor" << std::endl
			<< "//************************************************" << std::endl << std::endl
			<< "void selectRule( in Module module, out int successor_length, out int ruleID )" << std::endl
			<< "{" << std::endl
			<< "\tswitch ( module.symbolID )" << std::endl
			<< "\t{" << std::endl;
	for ( GrammarDescriptor::RuleMap::iterator it = rules.begin(); it != rules.end(); ++it )
	{
		String predecessorStr;
		predecessorStr = it->first.symbol;
		output	<< "\t\tcase " << predecessorStr << ":" << std::endl
				<< "\t\t\tselectRuleSymbol" << predecessorStr << ruleSelectionCall << ";" << std::endl
				<< "\t\t\tbreak;" << std::endl;
	}
	output	<< "\t\tdefault:" << std::endl
			<< "\t\t\tsuccessor_length = 0; ruleID = " << RULEID_NORULE << ";" << std::endl
			<< "\t\t\tbreak;" << std::endl
			<< "\t};" << std::endl
			<< "}" << std::endl << std::endl;

	// Module generator main funtion
	output	<< "//********************************************************" << std::endl
			<< "// Module generation function" << std::endl
			<< "//	- returns the next module of the corresponding rule" << std::endl
			<< "//********************************************************" << std::endl << std::endl
			<< "Module getNextModule" << std::endl
			<< "(" << std::endl
			<< "\tModule parent_module,\t\t// predecessor module" << std::endl
			<< "\tint ruleID,\t\t\t// ID of the selected rule" << std::endl
			<< "\tint number\t\t\t// position of the new (output) module in the successor" << std::endl
			<< ")" << std::endl
			<< "{" << std::endl
			<< "\tModule output;" << std::endl
			<< "\tswitch ( ruleID )" << std::endl
			<< "\t{" << std::endl;

	for ( GrammarDescriptor::RuleMap::iterator pred_it = rules.begin(); pred_it != rules.end(); ++pred_it )
	{
		String predecessorStr;
		predecessorStr = pred_it->first.symbol;
		for ( unsigned int i = 1; i <= pred_it->second.size(); ++i )
		{
			output	<< "\t\tcase " << predecessorStr << "_RULE_" << i << "_ID:" << std::endl
					<< "\t\t\treturn getNextModule_" << predecessorStr << "_RULE_"  << i << moduleGenerationCall
					<< ";" << std::endl;
		}
	}


	output	<< "\t\tdefault:" << std::endl
			<< "\t\t\tbreak;" << std::endl
			<< "\t};" << std::endl
			<< "\treturn output;" << std::endl
			<< "}" << std::endl << std::endl;
}

void HLSLCodeGenerator::createRules()
{
	OutFile &output = (*outputs[INDEX_RULE_FILE]);

	const String moduleHeader = "( Module parent, int number )";

	// header
	output	<< "#include \"" << OUTFILENAME_SYMBOLID << "\"" << std::endl 
			<< "#include \"moduleoperations.fx\"" << std::endl << std::endl
			<< "//*************************************************" << std::endl
			<< "// Rule shaders" << std::endl
			<< "//	- output is the nth module of the rule successor" << std::endl
			<< "//	- output can depend on the predecessor (parent" << std::endl
			<< "//		module)" << std::endl
			<< "//*************************************************" << std::endl << std::endl;

	// Rule functions
	GrammarDescriptor::RuleMap &rules = grammarDesc->getRules();
	bool first;
	for ( GrammarDescriptor::RuleMap::iterator pred_it = rules.begin(); pred_it != rules.end(); ++pred_it )
	{
		String predecessorStr;
		predecessorStr = pred_it->first.symbol;

		for ( unsigned int i = 1; i <= pred_it->second.size(); ++i )
		{
			String buffer;
			buffer += "// Rule: " + predecessorStr + " -> ";
			first = true;
			for ( SuccessorSymbolVector::iterator succ_it = pred_it->second[i-1].symbols.begin(); 
					succ_it != pred_it->second[i-1].symbols.end(); ++succ_it )
			{
				if ( first )
				{
					first = false;
				}
				else
				{
					buffer += " | ";
				}
				if ( MAX_ROW_LENGTH < buffer.length())
				{
					output << buffer << std::endl;
					buffer = "//";
					for ( unsigned int i = 0; i < String(" Rule:  -> ").length() + predecessorStr.length(); ++i )
					{
						buffer += " ";
					}
				}
				buffer += succ_it->symbol;
			}
			output << buffer;

			output	<< std::endl
					<< "Module getNextModule_" << predecessorStr << "_RULE_" << i << moduleHeader << std::endl
					<< "{" << std::endl
					<< "\tModule output = parent;" << std::endl
					<< "\tfloat dummyVariable;" << std::endl << std::endl
					<< "\t// TODO: Implement module initialization (i.e. rule behavior) here" << std::endl
					<< "\t//	- set symbol ID (done already, modify if needed)" << std::endl
					<< "\t//	- set module parameters (e.g. position)" << std::endl << std::endl;
			
			output	<< "\tswitch( number )" << std::endl
					<< "\t{" << std::endl;
			int succNumber = 0;
			for (	SuccessorSymbolVector::iterator succ_it = pred_it->second[i-1].symbols.begin(); 
					succ_it != pred_it->second[i-1].symbols.end(); ++succ_it )
			{
				++succNumber;
				output	<< "\t\tcase " << succNumber << ":" << std::endl
				        << "\t\t\toutput.symbolID = " << succ_it->symbol << ";" << std::endl << std::endl;
				if ( succ_it->operations.size() != 0 )
				{
					output << "\t\t\t// Module operations generated automatically from the grammar file:" << std::endl;
				}
				for ( ModuleOperationVector::iterator op_it = succ_it->operations.begin(); op_it != succ_it->operations.end();
						++op_it )
				{
					output << "\t\t\t";
					createOperation( output, *op_it );
					output << "\n";
				}
				if ( succ_it->operations.size() != 0 )
				{
					output << std::endl;
				}

				output	<< "\t\t\t// TODO: implement initialization of " << succNumber << ". module in the successor ("
				        << succ_it->symbol << ") here" << std::endl << std::endl
						<< "\t\t\tbreak;" << std::endl;
			}
			output	<< "\t\tdefault:" << std::endl
					<< "\t\t\toutput.symbolID = SYMBOL_INVALID;" << std::endl
					<< "\t\t\tbreak;" << std::endl
					<< "\t};" << std::endl;			
			output	<< std::endl
					<< "\treturn output;" << std::endl
					<< "}" << std::endl << std::endl;
		}
	}
}

void HLSLCodeGenerator::createOperation(OutFile &output, ModuleOperation& operation)
{
	String str;
	String paramValues[maxParameters]; // maxParameters defined in ModuleOperation.h
	OperatorSignature signature;
	CodeGenerator::getSignature(operation.operatorName,signature);
	for ( unsigned int parameterIndex = 0; parameterIndex < maxParameters; ++parameterIndex )
	{
		paramValues[parameterIndex] = operation.getParameter(parameterIndex);

		if ( operation.animated )
			addAnimationString(paramValues[parameterIndex],parameterIndex,signature,operation);
		if ( operation.randomized )
			addRandomString(paramValues[parameterIndex],parameterIndex,signature,operation);
	}

	// every pre-implemented has the string "module_" in its name
	str += "module_" + operation.operatorName; 
	str += "( output";

	bool isFunctionCall = true; // function calls should be terminated with ");"	

	if ( signature.useParent )
	{
		str += ", parent";
	}

	if ( "terminate" == operation.operatorName )
	{
		// nothing to do here.
	}
	else if ( "resize" == operation.operatorName ||
		"resize_x" == operation.operatorName ||
		"resize_y" == operation.operatorName ||
		"resize_z" == operation.operatorName || 
		"rotate_x" == operation.operatorName ||
		"rotate_y" == operation.operatorName ||
		"rotate_z" == operation.operatorName )
	{
		str += ", ";
		str += paramValues[0];
	}
	else if ( "scaled_move" == operation.operatorName )
	{
		str += ", ";
		str += paramValues[0] + ", ";
		str += paramValues[1] + ", ";
		str += paramValues[2];
		if ( ! paramValues[3].empty() )
			str += ", " + paramValues[3];
	}
	else if ( "scaled_move_x" == operation.operatorName || 
		"scaled_move_y" == operation.operatorName ||
		"scaled_move_z" == operation.operatorName )
	{
		str += ", ";
		str += paramValues[0] + ", ";
		if ( ! paramValues[1].empty() )
			str += ", " + paramValues[1];
	}
	else if ( "rotate" == operation.operatorName )
	{
		str += ", ";
		str += paramValues[0] + ", ";
		str += paramValues[1] + ", ";
		str += paramValues[2];
	}
	else if ( "add_noise" == operation.operatorName ||
		"add_noise_x" == operation.operatorName ||
		"add_noise_y" == operation.operatorName ||
		"add_noise_z" == operation.operatorName ||
		"add_noise_w" == operation.operatorName )
	{
		String attributeName = paramValues[0];
		if ( grammarDesc->getModuleType().count(attributeName) == 0 )
		{
			ERROR_MSG( "Invalid attribute", 
				TO_C_STRING("Shader generation error: no such attribute name: " + attributeName + "!"));
			exit(1);
		}
		// TODO: add more error checks, e.g. the type has the component y for add_noise_y 
		//   (i.e. the type has at least 2 components)
		str = "";
		String rndMin = paramValues[1];
		String rndMax = paramValues[2];
		String attributeStr = "output." + attributeName;
		if ( "add_noise_x" == operation.operatorName )
			attributeStr += ".x";
		else if ( "add_noise_y" == operation.operatorName )
			attributeStr += ".y";
		else if ( "add_noise_z" == operation.operatorName )
			attributeStr += ".z";
		else if ( "add_noise_w" == operation.operatorName )
			attributeStr += ".w";
		str += attributeStr + " = module_random(output) * (" + rndMax
			+ "-" + rndMin + ") + " + rndMin + ";";
		isFunctionCall = false;
	}
	else if ( "set" == operation.operatorName ||
		"set_x" == operation.operatorName ||
		"set_y" == operation.operatorName ||
		"set_z" == operation.operatorName ||
		"set_w" == operation.operatorName )
	{
		String attributeName = paramValues[0];
		if ( grammarDesc->getModuleType().count(attributeName) == 0 )
		{
			ERROR_MSG( "Invalid attribute", 
				TO_C_STRING("Shader generation error: no such attribute name: " + attributeName + "!"));
			exit(1);
		}
		// TODO: add more error checks, e.g. the type has the component y for set_y
		//   or check whether the value string is really an arithmetic expression of the attributes
		str = "";
		String attributeStr = "output." + attributeName;
		if ( "set_x" == operation.operatorName )
			attributeStr += ".x";
		else if ( "set_y" == operation.operatorName )
			attributeStr += ".y";
		else if ( "set_z" == operation.operatorName )
			attributeStr += ".z";
		else if ( "set_w" == operation.operatorName )
			attributeStr += ".w";
		// the attribute is set to the expression (string) given in the XML descriptor
		str += attributeStr + " = " + paramValues[1] + ";";
		isFunctionCall = false;
	}

	if ( isFunctionCall )
		str += " );";

	output << str;
}

void HLSLCodeGenerator::addAnimationString(String &parameterValue,unsigned int parameterIndex,
										   OperatorSignature& signature,ModuleOperation& operation)
{
	String periodStr = operation.getPeriodLength( parameterIndex );
	if ( "" == periodStr ) return;
	String minStr = operation.getAnimationMin( parameterIndex );
	String maxStr = operation.getAnimationMax( parameterIndex );
	parameterValue += " + (modf(currentTime/" + periodStr + ",dummyVariable) < 0.5 ? "
		+ "lerp(" + minStr + "," + maxStr + ",2*(modf(currentTime/" + periodStr + ",dummyVariable))) : "
		+ "lerp(" + minStr + "," + maxStr + ",-2*(modf(currentTime/" + periodStr + ",dummyVariable))+2 ))";
}

void HLSLCodeGenerator::addRandomString(String &parameterValue,unsigned int parameterIndex,
										   OperatorSignature& signature,ModuleOperation& operation)
{
	String minStr = operation.getRandomMin( parameterIndex );
	String maxStr = operation.getRandomMax( parameterIndex );
	parameterValue += " + (" + minStr + " + (" + maxStr + "-(" + minStr + "))*module_random(output)" + ")";
}

void HLSLCodeGenerator::buildTypeMap()
{
  addType(ATTRIBUTE_TYPE_FLOAT,"float","0",1);
  addType(ATTRIBUTE_TYPE_FLOAT2,"float2","float2(0,0)",2);
  addType(ATTRIBUTE_TYPE_FLOAT3,"float3","float3(0,0,0)",3);
  addType(ATTRIBUTE_TYPE_FLOAT4,"float4","float4(0,0,0,0)",4);
  addType(ATTRIBUTE_TYPE_INT,"int","0",1);
  addType(ATTRIBUTE_TYPE_UINT,"uint","0",1);
}

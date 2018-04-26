/*
**********************************************************************
 * Demo program for
 * Rule-based Geometry Synthesis in Real-time
 * ShaderX 8 article.
 *
 * @authors: Milan Magdics, Gergely Klar
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
#include "XMLGrammarLoader.h"

#include <set>
#include <algorithm>
#include <sstream>

#include "tinyxml.h"
#include "ErrorMessages.h"
#include "GrammarBasicTypes.h"
#include "GrammarSymbol.h"
#include "GrammarDescriptor.h"
#include "CodeGenerator.h"

#include "AxiomDescriptor.h"

XMLGrammarLoader::XMLGrammarLoader(GrammarDescriptor *aGrammarDesc, AxiomDescriptor *anAxiomDesc):GrammarLoader(aGrammarDesc, anAxiomDesc)
{
}

XMLGrammarLoader::~XMLGrammarLoader(void)
{
}

void XMLGrammarLoader::loadGrammar(const FileName& fileName)
{
	TiXmlDocument doc( fileName.c_str() );
	bool loadOkay = doc.LoadFile();

	if ( !loadOkay )
	{

		ERROR_MSG( "Cannot open grammar file", 
			TO_C_STRING("Grammar loading error: Cannot load file: " + fileName + "!"));
		exit(1);
	}

	TiXmlHandle docHandle( &doc );
	TiXmlHandle root = docHandle.FirstChild( "grammar" );
	TiXmlHandle symbolsNode = root.FirstChild( "symbols" );
	TiXmlHandle rulesNode = root.FirstChild( "rules" );
	TiXmlHandle propertiesNode = root.FirstChild( "properties" );
	TiXmlHandle moduleAttributesNode = root.FirstChild( "module_attributes" );
	TiXmlHandle axiomNode = root.FirstChild( "axiom" );

	loadSymbols(&symbolsNode);
	loadRules(&rulesNode);
	loadProperties(&propertiesNode);
	grammarDesc->prepare();                          // loading module data requires sorted symbol IDs
	loadModuleAttributes(&moduleAttributesNode);
	if (axiomNode.Element() != NULL)
		loadAxiom(&axiomNode);
}

void XMLGrammarLoader::loadSymbols( NodePtr symbolsNode )
{
	GrammarSymbol newSymbol;
	std::set<SymbolName> previousNames;

	SymbolVector &symbols = grammarDesc->getSymbols();
	for( TiXmlElement* element = symbolsNode->FirstChildElement("symbol").Element(); element;
		element = element->NextSiblingElement() )
	{
		newSymbol.name = element->Attribute("name");
		newSymbol.ruleSelection = RULESELECTION_ALWAYSFIRST;

		if ( 0 < previousNames.count( newSymbol.name ) )
		{
			ERROR_MSG( "Bad input format", 
				TO_C_STRING("Grammar loading error: duplicated symbol name: " + newSymbol.name + "!"));
			exit(1);
		}

		if ( element->Attribute("mesh") != NULL && element->Attribute("technique") != NULL )
		{
			newSymbol.instanced = true;
			newSymbol.instancingType.meshName = element->Attribute("mesh");;
			newSymbol.instancingType.technique = element->Attribute("technique");
		}
		else
		{
			newSymbol.instanced = false;
		}

		// if rule selection method is set
		if ( element->Attribute("rule_selection") != NULL )
		{
			String ruleSelectionStr = element->Attribute("rule_selection");
			RuleSelectionMethod ruleSelection;
			if ( stringToRuleSelectionMethod(ruleSelectionStr,ruleSelection) )
			{
				newSymbol.ruleSelection = ruleSelection;
			}
			else
			{
				ERROR_MSG( "Bad input format", 
				TO_C_STRING("Grammar loading error: invalid rule selection method: " + ruleSelectionStr + "!"));
			}
		}

		previousNames.insert( newSymbol.name );
		symbols.push_back(newSymbol);
	} 
}

void XMLGrammarLoader::loadRules( NodePtr rulesNode )
{
	GrammarDescriptor::RuleMap &rules = grammarDesc->getRules();
	for( TiXmlElement* ruleElement = rulesNode->FirstChildElement("rule").Element(); ruleElement;
		ruleElement = ruleElement->NextSiblingElement() )
	{
		// parsing predecessor
		String predecessor = ruleElement->FirstChildElement("predecessor")->Attribute("symbol");

		// parsing successors
		Successor successor;
		for( TiXmlElement* successorElement = ruleElement->FirstChildElement("successor"); successorElement;
			successorElement = successorElement->NextSiblingElement() )
		{
			SuccessorSymbol successorSymbol;
			successorSymbol.symbol = successorElement->Attribute("symbol");

			// loading operations assigned to the symbol
			TiXmlHandle successorNode = successorElement;
			loadModuleOperations(&successorNode,successorSymbol);

			successor.addSymbol(successorSymbol);
		}

		// loading probability, condition, LoD info (if set)
		if ( ruleElement->Attribute("probability") != NULL )
		{
			std::istringstream strStream(ruleElement->Attribute("probability"));
			strStream >> successor.probability;
		}
		if ( ruleElement->Attribute("condition") != NULL )
		{
			successor.condition = ruleElement->Attribute("condition");
		}
		if ( ruleElement->Attribute("lod_min_pixels") != NULL )
		{
			std::istringstream strStream(ruleElement->Attribute("lod_min_pixels"));
			strStream >> successor.lodMinPixels;
		}
		if ( ruleElement->Attribute("lod_max_pixels") != NULL )
		{
			std::istringstream strStream(ruleElement->Attribute("lod_max_pixels"));
			strStream >> successor.lodMaxPixels;
		}

		rules[predecessor].push_back(successor);
	}
}

void XMLGrammarLoader::loadProperties( NodePtr propertiesNode )
{
	// loading symbol name prefix
	TiXmlElement *propertyNode = propertiesNode->FirstChildElement("symbol_prefix").Element();
	if ( propertyNode )
	{
		String prefix = propertyNode->Attribute("value");
		String tmp = prefix;
		// transform to upper case
		transform( tmp.begin(), tmp.end(), prefix.begin(), toupper);
		grammarDesc->setSymbolNamePrefix(prefix);
	}

	// loading mesh library name
	propertyNode = propertiesNode->FirstChildElement("mesh_library").Element();
	if ( propertyNode )
	{
		String meshLibraryStr = propertyNode->Attribute("value");
		grammarDesc->setMeshLibrary(meshLibraryStr);
	}

	// loading mesh generation depth
	propertyNode = propertiesNode->FirstChildElement("generation_depth").Element();
	if ( propertyNode )
	{
		String depthStr = propertyNode->Attribute("value");
		std::istringstream strStream(depthStr);
		unsigned int depth = 0;
		strStream >> depth;
		grammarDesc->setGenerationDepth(depth);
	}
}

void XMLGrammarLoader::loadModuleAttributes( NodePtr moduleAttributesNode )
{
	GrammarDescriptor::ModuleType &moduleType = grammarDesc->getModuleType();
	AttributeName name;
	AttributeType type;
	String typeString;
	// loading predefined attributes
	for( TiXmlElement* element = moduleAttributesNode->FirstChildElement("predefined").Element(); element;
		element = element->NextSiblingElement("predefined") )
	{
		name = element->Attribute("name");
		if ( ! CodeGenerator::isPredefinedAttribute(name) )
		{
			ERROR_MSG("Bad input format",
				TO_C_STRING("Error: no such predefined attribute: \"" + name + "\"!"));
			exit(1);
		}
		if ( moduleType.count(name) > 0 )
		{
			ERROR_MSG("Bad input format",
				TO_C_STRING("Error: duplicated attribute name: \"" + name + "\"!"));
			exit(1);
		}
		CodeGenerator::getType(name,type);
		moduleType[name] = type;
	}

	// loading user-defined attributes
	for( TiXmlElement* element = moduleAttributesNode->FirstChildElement("userdefined").Element(); element;
		element = element->NextSiblingElement("userdefined") )
	{
		name = element->Attribute("name");
		// user-defined attributes cannot have the same name as any of the predefined attributes
		if ( CodeGenerator::isPredefinedAttribute(name) )
		{
			ERROR_MSG("Bad input format",
				TO_C_STRING("Error: attribute \"" + name + "\" is predefined! Rename it, or declare as predefined!"));
			exit(1);
		}
		if ( moduleType.count(name) > 0 )
		{
			ERROR_MSG("Bad input format",
				TO_C_STRING("Error: duplicated attribute name: \"" + name + "\"!"));
			exit(1);
		}

		typeString = element->Attribute("type");
		if ( ! stringToAttributeType(typeString,type) )
		{
			ERROR_MSG("Bad input format",
				TO_C_STRING("Error: no such type: \"" + typeString + "\"!"));
			exit(1);
		}
		moduleType[name] = type;
	}
}

void XMLGrammarLoader::loadModuleOperations( NodePtr successorNode, SuccessorSymbol &symbol )
{
	for( TiXmlElement* operation = successorNode->FirstChildElement("operation").Element(); operation;
		operation = operation->NextSiblingElement("operation") )
	{
		OperatorName opName = operation->Attribute("op");
		if ( ! CodeGenerator::isPredefinedOperator( opName ) )
		{
			ERROR_MSG("Bad input format",
				TO_C_STRING("Error: no such operator: \"" + opName + "\"!"));
			exit(1);
		}
		ModuleOperation moduleOp;
		moduleOp.operatorName = opName;
		OperatorSignature signature;
		CodeGenerator::getSignature( opName, signature );
		for ( unsigned int parameterIndex = 0; parameterIndex < signature.parameterNumber(); ++parameterIndex )
		{
			OperatorParameter& parameter = signature.getParameter( parameterIndex );
			if ( operation->Attribute(parameter.name.c_str()) == NULL && parameter.isRequired )
			{
				ERROR_MSG("Bad input format",
					TO_C_STRING("Error: missing required parameter of operator \"" + opName + "\" for successor symbol \"" +
					symbol.symbol + "\"!"));
				exit(1);
			}
			String parameterValue = operation->Attribute(parameter.name.c_str());
			std::istringstream strStream(parameterValue);
			strStream >> moduleOp.getParameter(parameterIndex);
		}

		moduleOp.animated = false;
		// load animation
		for ( TiXmlElement* animation = operation->FirstChildElement("animation"); animation;
			animation = animation->NextSiblingElement("animation") )
		{
			AttributeName attributeName = animation->Attribute("attribute");
			if ( ! signature.hasParameter(attributeName) )
			{
				ERROR_MSG("Bad input format",
					TO_C_STRING("Error: invalid animated attribute in \"" + opName + "\" for successor symbol \"" +
					symbol.symbol + "\"!"));
				exit(1);
			}
			unsigned int attributeIndex = signature.getParameterIndex(attributeName);
			moduleOp.getAnimationMin(attributeIndex) = animation->Attribute("min");
			moduleOp.getAnimationMax(attributeIndex) = animation->Attribute("max");
			moduleOp.getPeriodLength(attributeIndex) = animation->Attribute("period_length");
			moduleOp.animated = true;
		}
		moduleOp.randomized = false;
		// load randomization
		for ( TiXmlElement* random = operation->FirstChildElement("random"); random;
			random = random->NextSiblingElement("random") )
		{
			AttributeName attributeName = random->Attribute("attribute");
			if ( ! signature.hasParameter(attributeName) )
			{
				ERROR_MSG("Bad input format",
					TO_C_STRING("Error: invalid animated attribute in \"" + opName + "\" for successor symbol \"" +
					symbol.symbol + "\"!"));
				exit(1);
			}
			unsigned int attributeIndex = signature.getParameterIndex(attributeName);
			moduleOp.getRandomMin(attributeIndex) = random->Attribute("min");
			moduleOp.getRandomMax(attributeIndex) = random->Attribute("max");
			moduleOp.randomized = true;
		}

		symbol.addOperation( moduleOp );
	}
}

void XMLGrammarLoader::loadAxiom( NodePtr axiomNode )
{
	TiXmlElement *axiomElement = axiomNode->Element();
	if (axiomElement->Attribute("file") != NULL)
	{
		const String fileName = axiomElement->Attribute("file");
		TiXmlDocument axiomDoc( fileName.c_str() );
		bool loadOkay = axiomDoc.LoadFile();

		if ( !loadOkay )
		{

			ERROR_MSG( "Cannot open axiom file", 
				TO_C_STRING("Axiom file loading error: Cannot load file: " + fileName + "!"));
			exit(1);
		}

		TiXmlHandle axiomHandle = TiXmlHandle( &axiomDoc ).FirstChild( "axiom" );
		loadAxiomModules(&axiomHandle);
	} 
	else
		loadAxiomModules(axiomNode);
}

void XMLGrammarLoader::loadAxiomModules( NodePtr axiomNode )
{
	for( TiXmlElement* module = axiomNode->FirstChildElement("module").Element(); module;
		module = module->NextSiblingElement("module") )
	{
		Module m;
		String symbol = grammarDesc->getSymbolNamePrefix() + module->Attribute("symbol");
		std::ostringstream strStream;
		strStream << grammarDesc->nameToSymbolID(symbol);
		m.setAttribute("symbolID",strStream.str());
		for( TiXmlElement* attr = TiXmlHandle(module).FirstChildElement("attribute").Element(); attr;
			attr = attr->NextSiblingElement("attribute") )
		{
			String attrName = attr->Attribute("name");
			String attrValue = attr->Attribute("value");
			m.setAttribute(attrName, attrValue);
		}
		axiomDesc->addModule(m);
	}
}


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
#include "CodeGenerator.h"

CodeGenerator::CodeGenerator(GrammarDescriptor *aGrammarDesc):grammarDesc(aGrammarDesc)
{
}

CodeGenerator::~CodeGenerator(void)
{
}

void CodeGenerator::closeFiles()
{
	for ( OutFileVector::iterator it = outputs.begin(); it != outputs.end(); ++it )
	{
		if ( (*it) )
		{
			(*it)->close();
			delete (*it);
		}
	}
	outputs.clear();
}

bool CodeGenerator::isPredefinedAttribute(const AttributeName &attributeName)
{
	return predefinedAttributes.count(attributeName) > 0;
}

bool CodeGenerator::getType(const AttributeName &attributeName, AttributeType &type)
{
	if ( isPredefinedAttribute(attributeName) )
	{
		type = predefinedAttributes[attributeName];
		return true;
	}
	return false;
}

bool CodeGenerator::isPredefinedOperator(const OperatorName &operatorName)
{
	return predefinedOperators.count(operatorName) > 0;
}

bool CodeGenerator::getSignature(const OperatorName &operatorName, OperatorSignature &signature)
{
	if ( isPredefinedOperator(operatorName) )
	{
		signature = predefinedOperators[operatorName];
		return true;
	}
	return false;
}

void CodeGenerator::initPredefined()
{
	initPredefinedAttributes();
	initPredefinedOperators();
}

void CodeGenerator::initPredefinedAttributes()
{
	predefinedAttributes["symbolID"] = ATTRIBUTE_TYPE_UINT;
	predefinedAttributes["position"] = ATTRIBUTE_TYPE_FLOAT4;
	predefinedAttributes["size"] = ATTRIBUTE_TYPE_FLOAT;
	predefinedAttributes["size3"] = ATTRIBUTE_TYPE_FLOAT3;
	predefinedAttributes["orientation"] = ATTRIBUTE_TYPE_FLOAT4;
	predefinedAttributes["terminated"] = ATTRIBUTE_TYPE_UINT;
	predefinedAttributes["colorID"] = ATTRIBUTE_TYPE_UINT;
}

void CodeGenerator::initPredefinedOperators()
{
	OperatorSignature signature;
	OperatorParameter parameter;

	// termination - no input parameter
	signature.clear();
	signature.useParent = false;
	predefinedOperators["terminate"] = signature;

	// object resize
	signature.clear();
	parameter.isRequired = true;
	parameter.name = "value";
	parameter.type = ATTRIBUTE_TYPE_FLOAT;
	signature.addParameter(parameter);
	signature.useParent = true;
	predefinedOperators["resize"] = signature;
	predefinedOperators["resize_x"] = signature;
	predefinedOperators["resize_y"] = signature;
	predefinedOperators["resize_z"] = signature;

	signature.clear();
	parameter.isRequired = true;
	parameter.name = "x";
	parameter.type = ATTRIBUTE_TYPE_FLOAT;
	signature.addParameter(parameter);
	parameter.name = "y";
	signature.addParameter(parameter);
	parameter.name = "z";
	signature.addParameter(parameter);
	signature.useParent = true;
	predefinedOperators["resize3"] = signature;

	// object move
	signature.clear();
	parameter.isRequired = true;
	parameter.name = "x";
	parameter.type = ATTRIBUTE_TYPE_FLOAT;
	signature.addParameter(parameter);
	parameter.name = "y";
	signature.addParameter(parameter);
	parameter.name = "z";
	signature.addParameter(parameter);
	parameter.name = "scaler";
	parameter.isRequired = false;
	signature.addParameter(parameter);
	signature.useParent = true;
	predefinedOperators["scaled_move"] = signature;

	signature.clear();
	parameter.isRequired = true;
	parameter.name = "x";
	parameter.type = ATTRIBUTE_TYPE_FLOAT;
	signature.addParameter(parameter);
	parameter.name = "scaler";
	parameter.isRequired = false;
	signature.addParameter(parameter);
	signature.useParent = true;
	predefinedOperators["scaled_move_x"] = signature;

	signature.clear();
	parameter.isRequired = true;
	parameter.name = "y";
	parameter.type = ATTRIBUTE_TYPE_FLOAT;
	signature.addParameter(parameter);
	parameter.name = "scaler";
	parameter.isRequired = false;
	signature.addParameter(parameter);
	signature.useParent = true;
	predefinedOperators["scaled_move_y"] = signature;

	signature.clear();
	parameter.isRequired = true;
	parameter.name = "z";
	parameter.type = ATTRIBUTE_TYPE_FLOAT;
	signature.addParameter(parameter);
	parameter.name = "scaler";
	parameter.isRequired = false;
	signature.addParameter(parameter);
	signature.useParent = true;
	predefinedOperators["scaled_move_z"] = signature;

	// object rotation
	signature.clear();
	parameter.isRequired = true;
	parameter.name = "value";
	parameter.type = ATTRIBUTE_TYPE_FLOAT;
	signature.addParameter(parameter);
	signature.useParent = true;
	predefinedOperators["rotate_x"] = signature;
	predefinedOperators["rotate_y"] = signature;
	predefinedOperators["rotate_z"] = signature;

	signature.clear();
	parameter.isRequired = true;
	parameter.name = "x";
	parameter.type = ATTRIBUTE_TYPE_FLOAT;
	signature.addParameter(parameter);
	parameter.name = "y";
	signature.addParameter(parameter);
	parameter.name = "z";
	signature.addParameter(parameter);
	signature.useParent = true;
	predefinedOperators["rotate"] = signature;

	// perturbation of an attribute value
	signature.clear();
	parameter.isRequired = true;
	parameter.name = "attribute";
	parameter.type = ATTRIBUTE_TYPE_FLOAT;
	signature.addParameter(parameter);
	parameter.name = "min";
	signature.addParameter(parameter);
	parameter.name = "max";
	signature.addParameter(parameter);
	signature.useParent = true;
	predefinedOperators["add_noise"] = signature;
	predefinedOperators["add_noise_x"] = signature;
	predefinedOperators["add_noise_y"] = signature;
	predefinedOperators["add_noise_z"] = signature;
	predefinedOperators["add_noise_w"] = signature;

	// setting an attribute to the given value (any arithmetic expression of module attributes)
	//   NOTE: currently no error check is made, so an invalid value string will result an error in the
	//     HLSL shader compiler
	signature.clear();
	parameter.isRequired = true;
	parameter.name = "attribute";
	parameter.type = ATTRIBUTE_TYPE_FLOAT;
	signature.addParameter(parameter);
	parameter.name = "value";
	// NOTE: value is not a float, it is simply copied to the generated shader code
	signature.addParameter(parameter);
	signature.useParent = true;
	predefinedOperators["set"] = signature;
	predefinedOperators["set_x"] = signature;
	predefinedOperators["set_y"] = signature;
	predefinedOperators["set_z"] = signature;
	predefinedOperators["set_w"] = signature;
}

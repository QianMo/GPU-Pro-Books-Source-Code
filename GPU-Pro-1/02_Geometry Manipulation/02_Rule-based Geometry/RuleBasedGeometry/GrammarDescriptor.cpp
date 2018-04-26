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
#include "GrammarDescriptor.h"

#include <algorithm>
#include "CodeGenerator.h"

GrammarDescriptor::GrammarDescriptor(void)
{
	instancedCnt = 0;
	symbolNamePrefix = "";
	meshLibrary = "";
	generationDepth = 0;
}

GrammarDescriptor::~GrammarDescriptor(void)
{
}

void GrammarDescriptor::clear()
{
	instancedCnt = 0;
	symbolNamePrefix = "";
	symbols.clear();
	rules.clear();
	moduleType.clear();
	meshLibrary = "";
	symbolIDMap.clear();
	instancingTypes.clear();
	generationDepth = 0;
}

void GrammarDescriptor::prepare()
{
	// rendering code assumes that instanced modules have lower ID than non-instanced modules
	sortToInstancedNonInstanced();
	sortToInstancingTypes();
	addSymbolNamePrefixes();
	createIDMap();          // NOTE: this should be called after sorting
}

bool GrammarDescriptor::isAttributeDefined( const AttributeName &attributeName ) const
{
	return moduleType.count(attributeName) > 0;
}

unsigned int GrammarDescriptor::getInstancingTypeNumber() const
{
	return instancingTypes.size();
}

unsigned int GrammarDescriptor::getMaxRuleLength() const
{
	unsigned int maxLength = 0;
	for ( RuleMap::const_iterator rule_it = rules.begin(); rule_it != rules.end(); ++rule_it )
	{
		for ( SuccessorVector::const_iterator succ_it = rule_it->second.begin(); 
			succ_it != rule_it->second.end(); ++succ_it )
		{
			maxLength = max(maxLength,succ_it->symbolNumber());
		}
	}
	return maxLength;	
}

MeshName GrammarDescriptor::getMeshLibrary() const
{
	return meshLibrary;
}

unsigned int GrammarDescriptor::getGenerationDepth() const
{
	return generationDepth;
}

IDType GrammarDescriptor::getSymbolIDStart() const
{
	return CodeGenerator::SYMBOLID_START;
}

IDType GrammarDescriptor::nameToSymbolID(const SymbolName& name)
{
	return symbolIDMap[name];
}

SymbolName GrammarDescriptor::getSymbolNamePrefix() const
{
	return symbolNamePrefix;
}

InstancingTypeCounter& GrammarDescriptor::getInstancingType(unsigned int index)
{
	return instancingTypes[index];
}

bool GrammarDescriptor::isInstanced(GrammarSymbol symbol) const
{
	return symbol.instanced;
}

void GrammarDescriptor::sortToInstancedNonInstanced()
{
	GrammarSymbol tmp;
	instancedCnt = 0;

	for ( SymbolVector::iterator it = symbols.begin(); it != symbols.end(); ++it )
	{
		if ( isInstanced( (*it) ))
		{
			// assumption: first instancedCnt is instanced 
			//   => it's enough to swap the instancedCnt th element with the current
			tmp = (*it);
			(*it) = symbols[instancedCnt];
			symbols[instancedCnt] = tmp;
			++instancedCnt;
		}
	}
}

void GrammarDescriptor::sortToInstancingTypes()
{
	if ( symbols.empty() ) return;   // at least 1 symbol is assumed in the followings

	std::sort(symbols.begin(), symbols.begin()+instancedCnt);

	InstancingType type = symbols[0].instancingType;
	InstancingTypeCounter typeCounter;
	typeCounter.instancingType = type;
	typeCounter.number = 1;
	if ( instancedCnt > 0 )
		instancingTypes.push_back(typeCounter);

	for ( unsigned int i = 1; i < instancedCnt; ++i )
	{
		// if the next symbol has the same type
		if ( symbols[i].instancingType == type )
		{
			// increase the counter
			++(instancingTypes[instancingTypes.size()-1].number);
		}
		// if the next symbol has different type
		else
		{
			// store the found type as the current one, and reset the counter
			type = symbols[i].instancingType;
			typeCounter.instancingType = type;
			typeCounter.number = 1;
			instancingTypes.push_back(typeCounter);
		}
	}
}

void GrammarDescriptor::addSymbolNamePrefixes()
{
	for ( unsigned int i = 0; i != symbols.size(); ++i )
	{
		symbols[i].name = symbolNamePrefix + symbols[i].name;
	}

	for ( RuleMap::iterator rule_it = rules.begin(); rule_it != rules.end(); ++rule_it )
	{
			// const_cast: c++ does not allow to modify the iterator's firs element (map key), even if it does not
			//		affect the key (the modification does not affect the result of the comparison operator)
		Predecessor &pred = *(const_cast<Predecessor*>(&(rule_it->first)));

		pred.symbol = symbolNamePrefix + pred.symbol;

		SuccessorVector &succV = rule_it->second;
		for ( SuccessorVector::iterator succV_it = succV.begin(); succV_it != succV.end(); ++succV_it )
		{
			for ( SuccessorSymbolVector::iterator succ_it = succV_it->symbols.begin(); succ_it != succV_it->symbols.end();
				++succ_it )
			{
				succ_it->symbol = symbolNamePrefix + succ_it->symbol;
			}
		}
	}
}

void GrammarDescriptor::createIDMap()
{
	for ( unsigned int i = 0; i < symbols.size(); ++i )
	{
		symbolIDMap[symbols[i].name] = (CodeGenerator::SYMBOLID_START+i);
	}
}

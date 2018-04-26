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
#include "GrammarSymbol.h"


bool InstancingType::operator<(const InstancingType& other) const 
{ 
	return (meshName < other.meshName) || (meshName == other.meshName && technique < other.technique);
}
bool InstancingType::operator==(const InstancingType& other) const
{
	return meshName == other.meshName && technique == other.technique;
}
bool InstancingType::operator!=(const InstancingType& other) const
{
	return ! (*this == other);
}

GrammarSymbol::GrammarSymbol(void):
	instanced(false),
	ruleSelection(RULESELECTION_ALWAYSFIRST)
{
}

GrammarSymbol::~GrammarSymbol(void)
{
}

bool GrammarSymbol::operator<(const GrammarSymbol& other) const
{
	return this->instanced && this->instancingType < other.instancingType;
}

bool GrammarSymbol::operator==(const GrammarSymbol& other) const
{
	return (this->instanced && other.instanced) && this->instancingType == other.instancingType;
}

bool GrammarSymbol::operator!=(const GrammarSymbol& other) const
{
	return ! (*this == other);
}

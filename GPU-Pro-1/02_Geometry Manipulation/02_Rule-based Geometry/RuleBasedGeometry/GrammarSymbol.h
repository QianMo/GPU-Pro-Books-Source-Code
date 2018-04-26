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
#include <vector>
#pragma warning(default:4995)

#include "GrammarBasicTypes.h"

/////////////////////////////////////////////////////////
// InstancingType struct - represents an instancing type
/////////////////////////////////////////////////////////

struct InstancingType
{
	MeshName meshName;
	RenderingTechniqueName technique;

	bool operator<(const InstancingType& other) const;
	bool operator==(const InstancingType& other) const;
	bool operator!=(const InstancingType& other) const;
};

///////////////////////////////////////////////////////////////////////
// GrammarSymbol struct - symbols of the L-system and their properties
//
// - note that we do not distinguish between terminal and non-terminal
//   symbols
///////////////////////////////////////////////////////////////////////

struct GrammarSymbol
{
public:
	GrammarSymbol(void);
	~GrammarSymbol(void);

	SymbolName name; 
	bool instanced;
	InstancingType instancingType;
	RuleSelectionMethod ruleSelection;

	// NOTE: these operators are used for instancing types, symbol name is not taken into account
	bool operator<(const GrammarSymbol& other) const;
	bool operator==(const GrammarSymbol& other) const;
	bool operator!=(const GrammarSymbol& other) const;
};

// a sequence of GrammarSymbols is represented as a vector
typedef std::vector<GrammarSymbol> SymbolVector;

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
#include "ModuleOperation.h"


//////////////////////////////////////////////////////////////////
// SuccessorSymbol struct 
//
// - represents 1 symbol in a successor, including its operators
//////////////////////////////////////////////////////////////////

struct SuccessorSymbol
{
	SuccessorSymbol(void);
	~SuccessorSymbol(void);

	void addOperation( const ModuleOperation &op ) { operations.push_back(op); }

	SymbolName symbol;
	ModuleOperationVector operations;
};

// a sequence of SuccessorSymbols is represented as a vector
typedef std::vector<SuccessorSymbol> SuccessorSymbolVector;

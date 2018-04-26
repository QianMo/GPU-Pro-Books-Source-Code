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

#include "SuccessorSymbol.h"


//////////////////////////////////////////////////
// Successor struct - successors of grammar rules
//////////////////////////////////////////////////

struct Successor
{
public:
	Successor(void);
	~Successor(void);

	SuccessorSymbolVector symbols;
	float probability;
	String condition;
	unsigned int lodMinPixels;
	unsigned int lodMaxPixels;

	void addSymbol( const SuccessorSymbol &symbol ) { symbols.push_back( symbol ); }
	unsigned int symbolNumber() const               { return (unsigned int)symbols.size();}
	SuccessorSymbol getSymbol( unsigned int i )     { return symbols[i]; }
};

// a sequence of Successors is represented as a vector
typedef std::vector<Successor> SuccessorVector;

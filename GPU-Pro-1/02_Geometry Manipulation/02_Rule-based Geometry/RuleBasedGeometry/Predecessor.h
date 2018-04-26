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

#include "GrammarBasicTypes.h"


//////////////////////////////////////////////////////
// Predecessor struct - predecessors of grammar rules
//////////////////////////////////////////////////////

struct Predecessor
{
	Predecessor(const SymbolName& aSymbol);

	~Predecessor(void);

	bool operator==( const Predecessor &other ) const;
	bool operator!=( const Predecessor &other ) const;
	bool operator<( const Predecessor &other ) const;		// needed for storing rules in a map

	SymbolName symbol;
};

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
#include "Predecessor.h"

Predecessor::Predecessor(const SymbolName& aSymbol ):
	symbol(aSymbol)
{
}

Predecessor::~Predecessor(void)
{
}

bool Predecessor::operator==( const Predecessor &other ) const
{
		return symbol == other.symbol;
}

bool Predecessor::operator!=( const Predecessor &other ) const
{
		return symbol == other.symbol;
}

bool Predecessor::operator<( const Predecessor &other ) const
{
		return symbol < other.symbol;
}

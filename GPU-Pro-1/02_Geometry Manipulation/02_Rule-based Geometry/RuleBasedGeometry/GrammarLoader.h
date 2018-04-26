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
#include <string>
#pragma warning(default:4995)

/////////////////////////////////////////////////////////////////////////////////////////
// GrammarLoader class - abstract base class for loading a grammar description from file
/////////////////////////////////////////////////////////////////////////////////////////

class GrammarDescriptor;
class AxiomDescriptor;

class GrammarLoader
{
protected:
	typedef std::string FileName;
public:
	GrammarLoader(GrammarDescriptor *aGrammarDesc, AxiomDescriptor *anAxiomDesc);
	virtual ~GrammarLoader(void);

	virtual void loadGrammar(const FileName& fileName) = 0;
protected:
	GrammarDescriptor *grammarDesc;         // pointer to the L-system descriptor
	AxiomDescriptor		*axiomDesc;
};

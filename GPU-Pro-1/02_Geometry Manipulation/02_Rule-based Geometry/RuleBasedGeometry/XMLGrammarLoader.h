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
#include "grammarloader.h"


///////////////////////////////////////////////////////////////////////////
// XMLGrammarLoader class - loads an L-system description from an XML file
///////////////////////////////////////////////////////////////////////////

class TiXmlHandle;
struct SuccessorSymbol;

class XMLGrammarLoader :
	public GrammarLoader
{
	typedef TiXmlHandle* NodePtr;
public:
	XMLGrammarLoader(GrammarDescriptor *aGrammarDesc, AxiomDescriptor *anAxiomDesc);
	virtual ~XMLGrammarLoader(void);

	virtual void loadGrammar(const FileName& fileName);

protected:
	  // loads grammar symbols
	void loadSymbols( NodePtr symbolsNode );
	  // loads grammar rules
	void loadRules( NodePtr rulesNode );
      // loads properties, e.g. symbol name prefix
	void loadProperties( NodePtr propertiesNode );
	  // loads the module type (list of module attributes)
	void loadModuleAttributes( NodePtr moduleAttributesNode );
	  // loads the operators and their parameters assigned to a specific symbol in a successor
	void loadModuleOperations( NodePtr successorNode, SuccessorSymbol &symbol );
	  // loads grammar axioms
	void loadAxiom( NodePtr axiomNode );
	void loadAxiomModules( NodePtr axiomNode );
};

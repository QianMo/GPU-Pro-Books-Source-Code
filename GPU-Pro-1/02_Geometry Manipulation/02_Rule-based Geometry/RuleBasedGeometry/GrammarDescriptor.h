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
#include <map>
#pragma warning(default:4995)

#include "GrammarBasicTypes.h"
#include "Predecessor.h"
#include "Successor.h"
#include "GrammarSymbol.h"

struct InstancingTypeCounter
{
	InstancingType instancingType;
	unsigned int number;

	InstancingTypeCounter():number(0) {}
};

/////////////////////////////////////////////////////////////
// GrammarDescriptor class - the abstraction of an L-system
/////////////////////////////////////////////////////////////

class GrammarDescriptor
{
public:
	// Rules are stored as map of Predecessor->SuccessorVector
	// - note that in theory Predecessor->Successor is a 1:n mapping,
	//   instead of multimap, we used a map of vectors, this way
	//   iterating on successors for a specific predecessor becomes faster. 
	typedef std::map<Predecessor,SuccessorVector> RuleMap;

	// a module attribute has a name and a type
	typedef std::pair<AttributeName,AttributeType> ModuleAttribute;

	// we store the set of module attributes in a map to easier uniqueness check and type query
	typedef std::map<AttributeName,AttributeType> ModuleType;

	// symbol ID is used to store the symbol in the rendering codes, but it's more comfortable to use names when
	//   we build up an axiom buffer, so names should be mapped to IDs
	typedef std::map<SymbolName,IDType> SymbolIDMap;

	// symbols are sorted by instancing type for efficient instancing code. we store the instancing types,
	//   and the number of symbols with the same instancing type in a vector
	typedef std::vector<InstancingTypeCounter> InstancingTypeVector;
public:
	GrammarDescriptor(void);
	~GrammarDescriptor(void);

	void clear();                  // clears the whole L-system descriptor (for reuse)
	void prepare();                // prepares the descriptor for runtime use (should be called before code generation
	                               //   and before the L-system descriptor is used in the cpp rendering engine)

	// some get-set functions
	void setSymbolNamePrefix(const String &aString) { symbolNamePrefix = aString; }
	void setMeshLibrary(const MeshName &aMeshName)  { meshLibrary = aMeshName; }
	void setGenerationDepth(unsigned int depth)     { generationDepth = depth; }

	SymbolVector& getSymbols()  { return symbols; }
	RuleMap& getRules()         { return rules; }
	ModuleType& getModuleType() { return moduleType; }
                                   // return true if the module type contains the attribute
	bool isAttributeDefined( const AttributeName &attributeName ) const;
	unsigned int getInstancingTypeNumber() const;
	unsigned int getMaxRuleLength() const;
	MeshName getMeshLibrary() const;
	unsigned int getGenerationDepth() const;
	IDType getSymbolIDStart() const;
	IDType nameToSymbolID(const SymbolName& name);
	SymbolName getSymbolNamePrefix() const;
	                               // return the index_th instancing type (NOTE: no index validity check)
	InstancingTypeCounter& getInstancingType(unsigned int index);
protected:
	bool isInstanced( GrammarSymbol symbol ) const;
	                               // sorting: rendering code assumes that instanced symbols have lower ID
	void sortToInstancedNonInstanced();
	void sortToInstancingTypes();
	                               // adds a prefix to every symbol name 
	void addSymbolNamePrefixes();  //   (useful to avoid name collisions in the generated codes)
	                               // creates the mapping: symbol name -> symbol ID
	void createIDMap();            // NOTE: should be called after ID sorting
protected:
	SymbolVector symbols;         // the set of grammar symbols (excluding operators)
	unsigned int instancedCnt;    // number of instanced symbols (may be higher than the number of instancing types)
	RuleMap rules;                // rules of the grammar

	ModuleType moduleType;        // the set of module attributes stored as a map (see above for comment)
	String symbolNamePrefix;      // this prefix is added to every symbol name in method addSymbolNamePrefix()
	                              
	                              // name of the folder that stores the mesh files (used in instancing types)
	                              // - NOTE that this attribute has been put in the grammar description since
	                              //   the instancing type needs mesh names, thus, the mesh directory name becomes
	MeshName meshLibrary;         //   the property of the grammar
	         

	SymbolIDMap symbolIDMap;      // maps symbol names to IDs to more comfortable use in cpu codes

	                              // instancing types and the number of symbol with a type
	InstancingTypeVector instancingTypes;
	                              // generation depth of the production when generating the geometry
	unsigned int generationDepth; //   NOTE: the actual depth can be modified in run-time in the demo with numpad +/-
};

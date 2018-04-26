/*
**********************************************************************
 * Demo program for
 * Rule-based Geometry Synthesis in Real-time
 * ShaderX 8 article.
 *
 * @author: Gergely Klar
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

#include "Module.h"

////////////////////////////////////////////////////////////////////////////////////////
// AxiomDescriptor class - abstract for an Axiom of an L-system (a sequence of modules)
////////////////////////////////////////////////////////////////////////////////////////

class AxiomDescriptor
{
public:
	typedef std::vector<Module> ModuleVector;

	AxiomDescriptor(void);
	~AxiomDescriptor(void);

	void addModule(const Module& module);
	void clear();

	size_t getSize() const;
	const Module* getDataPtr() const;

protected:
	ModuleVector moduleVector;
};

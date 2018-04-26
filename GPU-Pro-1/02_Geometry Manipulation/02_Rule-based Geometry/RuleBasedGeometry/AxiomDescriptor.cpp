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

#include "DXUT.h"
#include "AxiomDescriptor.h"

AxiomDescriptor::AxiomDescriptor(void)
{
}

AxiomDescriptor::~AxiomDescriptor(void)
{
}

void AxiomDescriptor::addModule(const Module& module)
{ 
	moduleVector.push_back(module); 
}

void AxiomDescriptor::clear()
{
	moduleVector.clear();
}

size_t AxiomDescriptor::getSize() const
{
	return moduleVector.size();
}

const Module* AxiomDescriptor::getDataPtr() const
{
	if ( getSize() == 0 ) return NULL;
	return &(moduleVector[0]);
}

#include "DXUT.h"
#include "CopyResource.h"
#include "ScriptResourceVariable.h"
#include "Theatre.h"
#include "TaskContext.h"

CopyResource::CopyResource(	ScriptResourceVariable* source,	ScriptResourceVariable* destination)
{
	this->source = source;
	this->destination = destination;
}


void CopyResource::execute(const TaskContext& context)
{
	context.theatre->getDevice()->CopyResource(
		destination->getResource(),
		source->getResource());
}
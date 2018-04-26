#include "DXUT.h"
#include "DefineVariable.h"
#include "ScriptVariableClass.h"
#include "ResourceOwner.h"
#include "ResourceSet.h"
#include "TaskContext.h"

DefineVariable::DefineVariable(const ScriptVariableClass& type,	const wchar_t* name)
:type(type), name(name)
{
}

void DefineVariable::execute(const TaskContext& context)
{
	context.localResourceOwner->getResourceSet()->createVariable(type, name);
}

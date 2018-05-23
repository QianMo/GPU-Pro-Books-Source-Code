//////////////////////////////////////////////////////////////////////////////
//
//  Copyright (C) Tobias Zirr. All Rights Reserved.
//
//////////////////////////////////////////////////////////////////////////////

#include "stdafx.h"
#include "D3D11EffectsLite.h"
#include "Reflection.h"
#include "Allocator.h"
#include <cassert>

namespace D3DEffectsLite
{

PlainReflection::PlainReflection()
	: ReferenceCounter(1),
	Info()
{
}

const D3DEL_VARIABLE_INFO* D3DEFFECTSLITE_STDCALL PlainReflection::GetVariableByName(const char *name) const
{
	for (UINT i = 0; i < Info.VariableCount; ++i)
		if (strcmp(Info.Variables[i].Name, name) == 0)
			return &Info.Variables[i];

	return nullptr;
}
const D3DEL_VARIABLE_INFO* D3DEFFECTSLITE_STDCALL PlainReflection::GetVariableBySemantic(const char *semantic) const
{
	for (UINT i = 0; i < Info.VariableCount; ++i)
		if (Info.Variables[i].Semantic && strcmp(Info.Variables[i].Semantic, semantic) == 0)
			return &Info.Variables[i];

	return nullptr;
}

const D3DEL_GROUP_INFO* D3DEFFECTSLITE_STDCALL PlainReflection::GetGroup(const char *name) const
{
	for (UINT i = 0; i < Info.GroupCount; ++i)
		if (strcmp(Info.Groups[i].Name, name) == 0)
			return &Info.Groups[i];

	return nullptr;
}
const D3DEL_TECHNIQUE_INFO* D3DEFFECTSLITE_STDCALL PlainReflection::GetTechnique(const char *name) const
{
	for (UINT i = 0; i < Info.TechniqueCount; ++i)
		if (strcmp(Info.Techniques[i].Name, name) == 0)
			return &Info.Techniques[i];

	return nullptr;
}
const D3DEL_TECHNIQUE_INFO* D3DEFFECTSLITE_STDCALL PlainReflection::GetTechnique(const char *name, const D3DEL_GROUP_INFO *group) const
{
	if (!group)
		return GetTechnique(name);

	for (UINT i = 0; i < group->TechniqueCount; ++i)
		if (strcmp(group->Techniques[i].Name, name) == 0)
			return &group->Techniques[i];

	return nullptr;
}
const D3DEL_PASS_INFO* D3DEFFECTSLITE_STDCALL PlainReflection::GetPass(const char *name, const D3DEL_TECHNIQUE_INFO *technique) const
{
	if (!technique)
		return nullptr;

	for (UINT i = 0; i < technique->PassCount; ++i)
		if (strcmp(technique->Passes[i].Name, name) == 0)
			return &technique->Passes[i];

	return nullptr;
}

const D3DEL_EFFECT_INFO* D3DEFFECTSLITE_STDCALL PlainReflection::GetInfo() const
{
	return &Info;
}

com_ptr<PlainReflection> CreateEmptyPlainReflection()
{
	return new(*GetGlobalAllocator()) PlainReflection();
}

} // namespace

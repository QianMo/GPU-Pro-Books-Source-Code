//////////////////////////////////////////////////////////////////////////////
//
//  Copyright (C) Tobias Zirr.  All Rights Reserved.
//
//////////////////////////////////////////////////////////////////////////////

#pragma once

#include "D3D11EffectsLite.h"
#include "RefCounted.h"
#include "Heap.h"

namespace D3DEffectsLite
{

struct PlainReflection : public RefCounted<Reflection>
{
	EffectInfo Info;
	Heap Heap;
	ULONG ReferenceCounter;

	PlainReflection();

	const D3DEL_VARIABLE_INFO* D3DEFFECTSLITE_STDCALL GetVariableByName(const char *name) const;
	const D3DEL_VARIABLE_INFO* D3DEFFECTSLITE_STDCALL GetVariableBySemantic(const char *semantic) const;

	const D3DEL_GROUP_INFO* D3DEFFECTSLITE_STDCALL GetGroup(const char *name) const;
	const D3DEL_TECHNIQUE_INFO* D3DEFFECTSLITE_STDCALL GetTechnique(const char *name) const;
	const D3DEL_TECHNIQUE_INFO* D3DEFFECTSLITE_STDCALL GetTechnique(const char *name, const D3DEL_GROUP_INFO *group) const;
	const D3DEL_PASS_INFO* D3DEFFECTSLITE_STDCALL GetPass(const char *name, const D3DEL_TECHNIQUE_INFO *technique) const;

	const D3DEL_EFFECT_INFO* D3DEFFECTSLITE_STDCALL GetInfo() const;
};

com_ptr<PlainReflection> CreateEmptyPlainReflection();

} // namespace

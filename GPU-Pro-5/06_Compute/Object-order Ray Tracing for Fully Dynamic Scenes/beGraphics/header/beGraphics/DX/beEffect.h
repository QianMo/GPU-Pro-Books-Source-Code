/******************************************************/
/* breeze Engine Graphics Module (c) Tobias Zirr 2011 */
/******************************************************/

#pragma once
#ifndef BE_GRAPHICS_EFFECT_DX
#define BE_GRAPHICS_EFFECT_DX

#include "beGraphics.h"
#include "../beEffectCache.h"
#include <D3DCommon.h>
#include <lean/containers/dynamic_array.h>
#include <lean/strings/utility.h>

namespace beGraphics
{

namespace DX
{

/// Converts the given effect macros into a string representation.
template <class String>
LEAN_INLINE void ToString(const D3D_SHADER_MACRO *macros, const D3D_SHADER_MACRO *macrosEnd, String &string)
{
	bool bFirst = true;

	for (const D3D_SHADER_MACRO *macro = macros; macro != macrosEnd; ++macro)
	{
		if (macro->Name)
		{
			if (!bFirst)
				string.append(1, ';');
			bFirst = false;

			string.append(macro->Name);

			if (macro->Definition)
			{
				string.append(1, '=')
					.append(macro->Definition);
			}
		}
	}
}

/// Extracts the next macro.
template <class Char>
LEAN_INLINE const Char* ExtractNextMacro(const Char* it, lean::range<const Char*> &name, lean::range<const Char*> &definition)
{
	// Ignore redundant separators
	while (*it == ';' || *it == ' ')
		++it;

	name.begin() = it;

	// Find OPTIONAL definition
	while (*it != '=' && *it != ';' && *it)
		++it;

	name.end() = it;

	// Check for optional definition
	if (*it == '=')
	{
		definition.begin() = ++it;

		// Find definition end
		while (*it != ';' && *it)
			++it;

		definition.end() = it;
	}

	return it;
}

/// Counts the number of macros.
template <class String>
LEAN_INLINE size_t CountMacros(const String &macroString)
{
	size_t macroCount = 0;

	for (const char *it = macroString.c_str(); *it; )
	{
		lean::range<const char*> macroName;
		lean::range<const char*> macroDefinition;
		it = ExtractNextMacro(it, macroName, macroDefinition);

		macroCount += !macroName.empty();
	}

	return macroCount;
}

/// Converts the given string representation into an array of effect macros.
template <class String>
LEAN_INLINE lean::dynamic_array<EffectMacro> ToMacros(const String &macroString)
{
	lean::dynamic_array<EffectMacro> macros( CountMacros(macroString) );

	for (const char *it = macroString.c_str(); *it; )
	{
		lean::range<const char*> macroName;
		lean::range<const char*> macroDefinition;
		it = ExtractNextMacro(it, macroName, macroDefinition);

		if (!macroName.empty())
			macros.push_back(EffectMacro(macroName, macroDefinition));
	}

	return macros;
}

/// Converts the given string representation into an array of effect hooks.
template <class String>
LEAN_INLINE lean::dynamic_array<EffectHook> ToHooks(const String &hookString)
{
	lean::dynamic_array<EffectHook> hooks( CountMacros(hookString) );

	for (const char *it = hookString.c_str(); *it; )
	{
		lean::range<const char*> macroName;
		lean::range<const char*> macroDefinition;
		it = ExtractNextMacro(it, macroName, macroDefinition);

		if (!macroName.empty())
			hooks.push_back(EffectHook(macroName));
	}

	return hooks;
}

/// Converts the given effect macros to DirectX shader macros.
LEAN_INLINE lean::dynamic_array<D3D_SHADER_MACRO> ToAPI(const EffectMacro *macros, size_t macroCount, lean::dynamic_array<char> &backingStore)
{
	// NOTE: Include terminating null macro
	lean::dynamic_array<D3D_SHADER_MACRO> macrosDX(macroCount + 1);

	size_t storeLength = 0;
	
	for (size_t i = 0; i < macroCount; ++i)
	{
		const EffectMacro &macro = macros[i];
		// NOTE: Include terminating null
		storeLength += macro.Name.size() + 1;
		storeLength += macro.Definition.size() + 1;
	}

	backingStore.reset(storeLength);

	for (size_t i = 0; i < macroCount; ++i)
	{
		const EffectMacro &macro = macros[i];

		D3D_SHADER_MACRO &macroDX = macrosDX.push_back();
		macroDX.Name = lean::strcpy_range(
			backingStore.push_back_n(macro.Name.size() + 1),
			macro.Name );
		macroDX.Definition = lean::strcpy_range(
			backingStore.push_back_n(macro.Definition.size() + 1),
			macro.Definition );
	}

	D3D_SHADER_MACRO &endOfMacrosDX = macrosDX.push_back();
	endOfMacrosDX.Name = nullptr;
	endOfMacrosDX.Definition = nullptr;

	return macrosDX;
}

} // namespace

using DX::ToAPI;

} // namespace

#endif
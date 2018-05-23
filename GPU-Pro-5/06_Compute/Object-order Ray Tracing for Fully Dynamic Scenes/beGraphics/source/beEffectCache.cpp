/*******************************************************/
/* breeze Engine Graphics Module  (c) Tobias Zirr 2011 */
/*******************************************************/

#include "beGraphicsInternal/stdafx.h"
#include "beGraphics/beEffectCache.h"

#include "beGraphics/DX/beEffect.h"

#include <lean/io/filesystem.h>
#include <lean/io/numeric.h>
#include <lean/functional/bits.h>

#include <functional>

namespace beGraphics
{

namespace
{

template <class String, class Iterator>
void append_alnum(String &string, Iterator begin, Iterator end)
{
	// Replace non-alnum characters
	size_t beginPos = string.size();
	string.append(begin, end);
	std::replace_if(string.begin() + beginPos, string.end(), std::not1(std::ptr_fun(isalnum)), '_');
}

} // namespace

// Mangles the given file name & macros.
Exchange::utf8_string MangleFilename(const lean::utf8_ntri &file, const EffectMacro *pMacros, size_t macroCount, const EffectHook *pHooks, size_t hookCount)
{
	Exchange::utf8_string mangledFile;

	if (!pMacros) macroCount = 0;
	if (!pHooks) hookCount = 0;

	if (pMacros || pHooks)
	{
		size_t mangledLength = file.size() + 2;

		for (size_t i = 0; i < macroCount; ++i)
			mangledLength += 2 + pMacros->Name.size() + pMacros->Definition.size();

		for (size_t i = 0; i < hookCount; ++i)
			mangledLength += 1 + pHooks->File.size();
		
		mangledFile.reserve(mangledLength);
		
		mangledFile.append(file.begin(), file.end());
		mangledFile.append(1, '(');

		uint4 hash = lean::compute_hash<uint4>(file.begin(), file.end());

		for (size_t i = 0; i < macroCount; ++i)
		{
			const EffectMacro &macro = pMacros[i];

			if (!macro.Name.empty())
			{
				append_alnum(mangledFile.append(1, '-'), macro.Name.begin(), macro.Name.end());
				
				hash ^= lean::compute_hash<uint4>(macro.Name.begin(), macro.Name.end());

				if (!macro.Definition.empty())
				{
					append_alnum(mangledFile.append(1, '='), macro.Definition.begin(), macro.Definition.end());

					hash ^= lean::compute_hash<uint4>(macro.Definition.begin(), macro.Definition.end());
				}
			}
		}

		for (size_t i = 0; i < hookCount; ++i)
		{
			const EffectHook &hook = pHooks[i];

			if (!hook.File.empty())
			{
				append_alnum(mangledFile.append(1, '$'), lean::get_filename(hook.File.begin(), hook.File.end()), hook.File.end());

				hash ^= lean::compute_hash<uint4>(hook.File.begin(), hook.File.end());
			}
		}

		// NOTE: Don't hash absolute and/or unresolved paths, not portable across machines
		// TODO: Do something more intelligent?
/*		uint4 hashStringLen = (lean::size_info<uint4>::bits + 4) / 5;
		mangledFile.append(hashStringLen + 1, '!');
		lean::int_to_char(mangledFile.end() - hashStringLen, hash, 36U);
*/
		mangledFile.append(1, ')');
	}
	else
		mangledFile.assign(file.begin(), file.end());

	return mangledFile;
}

} // namespace
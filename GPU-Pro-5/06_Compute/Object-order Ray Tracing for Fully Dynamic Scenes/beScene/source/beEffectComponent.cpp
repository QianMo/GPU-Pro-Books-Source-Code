/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#include "beSceneInternal/stdafx.h"
#include <beGraphics/beEffect.h>

#include <beCore/beComponentReflector.h>
#include <beCore/beComponentTypes.h>
#include <beCore/beReflectionTypes.h>

#include "beScene/beSerializationParameters.h"
#include "beScene/beResourceManager.h"

#include <beGraphics/beEffectCache.h>

#include <lean/logging/log.h>
#include <lean/logging/errors.h>

namespace beScene
{

/// Reflects effects for use in component-based editing environments.
class EffectReflector : public beCore::ComponentReflector
{
	/// Gets principal component flags.
	uint4 GetComponentFlags() const LEAN_OVERRIDE
	{
		return bec::ComponentFlags::Filed;
	}
	/// Gets specific component flags.
	uint4 GetComponentFlags(const lean::any &component) const LEAN_OVERRIDE
	{
		uint4 flags = bec::ComponentFlags::NameMutable; // | bec::ComponentFlags::FileMutable

		if (const beg::Effect *effect = any_cast_default<beg::Effect*>(component))
			if (const beg::EffectCache *cache = effect->GetCache())
				if (!cache->GetFile(effect).empty())
					flags |= bec::ComponentState::Filed;

		return flags;
	}

	/// Gets information on the components currently available.
	bec::ComponentInfoVector GetComponentInfo(const beCore::ParameterSet &parameters) const LEAN_OVERRIDE
	{
		return GetSceneParameters(parameters).ResourceManager->EffectCache()->GetInfo();
	}
	
	/// Gets the component info.
	bec::ComponentInfo GetInfo(const lean::any &component) const LEAN_OVERRIDE
	{
		bec::ComponentInfo result;

		if (const beg::Effect *effect = any_cast_default<beg::Effect*>(component))
			if (const beg::EffectCache *cache = effect->GetCache())
				result = cache->GetInfo(effect);

		return result;
	}

	/// Sets the component name.
	void SetName(const lean::any &component, const utf8_ntri &name) const LEAN_OVERRIDE
	{
		if (beg::Effect *effect = any_cast_default<beg::Effect*>(component))
			if (beg::EffectCache *cache = effect->GetCache())
			{
				cache->SetName(effect, name);
				return;
			}

		LEAN_THROW_ERROR_CTX("Unknown effect cannot be renamed", name.c_str());
	}

	/// Gets a component by name.
	lean::cloneable_obj<lean::any, true> GetComponentByName(const utf8_ntri &name, const beCore::ParameterSet &parameters) const LEAN_OVERRIDE
	{
		SceneParameters sceneParameters = GetSceneParameters(parameters);

		return bec::any_resource_t<beGraphics::Effect>::t(
				sceneParameters.ResourceManager->EffectCache()->GetByName(name)
			);
	}

	/// Gets a fitting file extension, if available.
	utf8_ntr GetFileExtension() const LEAN_OVERRIDE
	{
		return utf8_ntr("fx");
	}
	/// Gets a list of creation parameters.
	beCore::ComponentParameters GetFileParameters(const utf8_ntri &file) const LEAN_OVERRIDE
	{
		static const beCore::ComponentParameter parameters[] = {
				beCore::ComponentParameter(utf8_ntr("Switches"), bec::GetReflectionType(bec::ReflectionType::String), bec::ComponentParameterFlags::Array),
				beCore::ComponentParameter(utf8_ntr("Hooks"), bec::GetReflectionType(bec::ReflectionType::File), bec::ComponentParameterFlags::Array)
			};

		return beCore::ComponentParameters(parameters, parameters + lean::arraylen(parameters));
	}
	/// Gets a component from the given file.
	lean::cloneable_obj<lean::any, true> GetComponentByFile(const utf8_ntri &file,
		const beCore::Parameters &fileParameters, const beCore::ParameterSet &parameters) const LEAN_OVERRIDE
	{
		SceneParameters sceneParameters = GetSceneParameters(parameters);

		std::vector<beg::EffectMacro> macros;
		if (const lean::any *macroParam = fileParameters.GetAnyValue(fileParameters.GetID("Switches")))
		{
			macros.resize(macroParam->size());

			for (uint4 i = 0, count = (uint4) macroParam->size(); i < count; ++i)
				macros[i].Name = lean::make_range_v(
						any_cast_checked<const bec::reflection_type<bec::ReflectionType::String>::type&>(macroParam, i)
					);
		}

		std::vector<beg::EffectHook> hooks;
		if (const lean::any *hookParam = fileParameters.GetAnyValue(fileParameters.GetID("Hooks")))
		{
			hooks.resize(hookParam->size());

			for (uint4 i = 0, count = (uint4) hookParam->size(); i < count; ++i)
				hooks[i].File = lean::make_range_v(
						any_cast_checked<const bec::reflection_type<bec::ReflectionType::File>::type&>(hookParam, i)
					);
		}

		return bec::any_resource_t<beGraphics::Effect>::t(
				sceneParameters.ResourceManager->EffectCache()->GetByFile(file, &macros[0], (uint4) macros.size(), &hooks[0], (uint4) hooks.size())
			);
	}

	/// Gets the component type reflected.
	const beCore::ComponentType* GetType() const LEAN_OVERRIDE
	{
		return beg::Effect::GetComponentType(); 
	}
};

static const beCore::ComponentReflectorPlugin<EffectReflector> EffectReflectorPlugin(beg::Effect::GetComponentType());

} // namespace

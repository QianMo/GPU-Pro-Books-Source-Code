/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#include "beSceneInternal/stdafx.h"
#include <beGraphics/beMaterial.h>
#include "beScene/beRenderableMaterial.h"
#include "beScene/beLightMaterial.h"

#include <beCore/beComponentReflector.h>
#include <beCore/beComponentTypes.h>
#include <beCore/beReflectionTypes.h>

#include "beScene/beSerializationParameters.h"
#include "beScene/beResourceManager.h"
#include "beScene/beEffectDrivenRenderer.h"

#include <beGraphics/beMaterialCache.h>
#include "beScene/beRenderableMaterialCache.h"
#include "beScene/beLightMaterialCache.h"

#include <beGraphics/beEffectCache.h>

#include <lean/logging/log.h>
#include <lean/logging/errors.h>

namespace beScene
{

extern const beCore::ComponentType RenderableMaterialType;
extern const beCore::ComponentType LightMaterialType;

namespace
{

template <class BoundMaterial>
BoundMaterial* GetBinding(const SceneParameters &sceneParameters, beg::Material *material);

/// Gets a renderable binding for the given material.
template <>
RenderableMaterial* GetBinding(const SceneParameters &sceneParameters, beg::Material *material)
{
	return sceneParameters.Renderer->RenderableMaterials()->GetMaterial(material);
}

/// Gets a light binding for the given material.
template <>
LightMaterial* GetBinding(const SceneParameters &sceneParameters, beg::Material *material)
{
	return sceneParameters.Renderer->LightMaterials()->GetMaterial(material);
}

} // namespace

/// Reflects materials for use in component-based editing environments.
template <class BoundMaterial>
class BoundMaterialReflector : public beCore::ComponentReflector
{
	/// Gets principal component flags.
	uint4 GetComponentFlags() const LEAN_OVERRIDE
	{
		return bec::ComponentFlags::Creatable | bec::ComponentFlags::Cloneable; // bec::ComponentFlags::Filed | 
	}
	/// Gets specific component flags.
	uint4 GetComponentFlags(const lean::any &component) const LEAN_OVERRIDE
	{
		return bec::ComponentFlags::NameMutable; // | bec::ComponentFlags::FileMutable
	}

	/// Gets information on the components currently available.
	bec::ComponentInfoVector GetComponentInfo(const beCore::ParameterSet &parameters) const LEAN_OVERRIDE
	{
		return GetSceneParameters(parameters).ResourceManager->MaterialCache()->GetInfo();
	}
	
	/// Gets the component info.
	bec::ComponentInfo GetInfo(const lean::any &component) const LEAN_OVERRIDE
	{
		bec::ComponentInfo result;

		if (const BoundMaterial *material = any_cast_default<BoundMaterial*>(component))
			if (const beg::MaterialCache *cache = material->GetMaterial()->GetCache())
				result = cache->GetInfo(material->GetMaterial());

		return result;
	}

	/// Gets a list of creation parameters.
	beCore::ComponentParameters GetCreationParameters() const LEAN_OVERRIDE
	{
		static const beCore::ComponentParameter parameters[] = {
				beCore::ComponentParameter(utf8_ntr("Effect"), beg::Effect::GetComponentType(), bec::ComponentParameterFlags::Deducible),
				beCore::ComponentParameter(utf8_ntr("Configuration"), beg::MaterialConfig::GetComponentType(), bec::ComponentParameterFlags::Array),
				beCore::ComponentParameter(utf8_ntr("New Config"), bec::GetReflectionType(bec::ReflectionType::Boolean), bec::ComponentParameterFlags::Optional)
			};

		return beCore::ComponentParameters(parameters, parameters + lean::arraylen(parameters));
	}
	/// Creates a component from the given parameters.
	lean::cloneable_obj<lean::any, true> CreateComponent(const utf8_ntri &name, const beCore::Parameters &creationParameters,
		const beCore::ParameterSet &parameters, const lean::any *pPrototype, const lean::any *pReplace) const LEAN_OVERRIDE
	{
		SceneParameters sceneParameters = GetSceneParameters(parameters);

		BoundMaterial *pPrototypeBoundMaterial = lean::any_cast_default<BoundMaterial*>(pPrototype);
		beg::Material *pPrototypeMaterial = (pPrototypeBoundMaterial) ? pPrototypeBoundMaterial->GetMaterial() : nullptr;

		lean::resource_ptr<beg::Material> material;
		{
			beg::Material::Effects effects;
			const beGraphics::Effect *pSpecifiedEffect = creationParameters.GetValueDefault<beGraphics::Effect*>("Effect");
			
			if (!pSpecifiedEffect && pPrototypeMaterial)
				effects = pPrototypeMaterial->GetEffects();
			else
			{
				LEAN_THROW_NULL(pSpecifiedEffect);
				effects = bec::MakeRangeN(&pSpecifiedEffect, 1);
			}

			material = beg::CreateMaterial(
					effects.Begin, Size4(effects),
					*sceneParameters.ResourceManager->EffectCache()
				);
		}

		{
			std::vector<beg::MaterialConfig*> configs;

			if (const lean::any *pConfiguration = creationParameters.GetAnyValue(creationParameters.GetID("Configuration")))
			{
				configs.resize(pConfiguration->size());

				for (uint4 configIdx = 0, configCount = (uint4) pConfiguration->size(); configIdx < configCount; ++configIdx)
					configs[configIdx] = LEAN_THROW_NULL(
							any_cast<beGraphics::MaterialConfig*>( any_cast_checked<const lean::any&>(pConfiguration, configIdx) )
						);
			}
/*			else if (pPrototypeMaterial)
			{
				beg::Material::Configurations configRange = pPrototypeMaterial->GetConfigurations();
				configs.assign(configRange.Begin, configRange.End);
				material->SetConfigurations(configs.data(), (uint4) configs.size());
			}
*/
			// SCOPE: Keep alive until set configurations succeeded!
			lean::resource_ptr<beg::MaterialConfig> newConfig;

			if (creationParameters.GetValueDefault<bool>("New Config", false))
			{
				if (!configs.empty())
					newConfig = beg::CreateMaterialConfig(*configs[0]);
				else
					newConfig = beg::CreateMaterialConfig();

				sceneParameters.ResourceManager->MaterialConfigCache()->SetName(
						newConfig,
						sceneParameters.ResourceManager->MaterialConfigCache()->GetUniqueName(name)
					);

				if (!configs.empty())
					configs[0] = newConfig;
				else
					configs.push_back(newConfig);
			}

			material->SetConfigurations(configs.data(), (uint4) configs.size());
		}

		// TODO?
//		if (pPrototypeMaterial)
//			Transfer(*material, *pPrototypeMaterial);

		if (BoundMaterial *pToBeReplaced = lean::any_cast_default<BoundMaterial*>(pReplace))
			sceneParameters.ResourceManager->MaterialCache()->Replace(pToBeReplaced->GetMaterial(), material);
		else
			sceneParameters.ResourceManager->MaterialCache()->SetName(material, name);


		return bec::any_resource_t<BoundMaterial>::t(
				GetBinding<BoundMaterial>( sceneParameters, material )
			);
	}

	/// Sets the component name.
	void SetName(const lean::any &component, const utf8_ntri &name) const LEAN_OVERRIDE
	{
		if (BoundMaterial *material = any_cast_default<BoundMaterial*>(component))
			if (beg::MaterialCache *cache = material->GetMaterial()->GetCache())
			{
				cache->SetName(material->GetMaterial(), name);
				return;
			}

		LEAN_THROW_ERROR_CTX("Unknown material cannot be renamed", name.c_str());
	}

	// Gets a list of creation parameters.
	void GetCreationInfo(const lean::any &component, bec::Parameters &creationParameters, bec::ComponentInfo *pInfo = nullptr) const LEAN_OVERRIDE
	{
		if (const BoundMaterial *material = any_cast_default<BoundMaterial*>(component))
			if (const beg::MaterialCache *cache = material->GetMaterial()->GetCache())
			{
				if (pInfo)
					*pInfo = cache->GetInfo(material->GetMaterial());

				creationParameters.SetValue<const beGraphics::Effect*>("Effect", *material->GetMaterial()->GetEffects().Begin);

				beg::Material::Configurations configRange = material->GetMaterial()->GetConfigurations();
				typedef std::vector< bec::any_resource_t<beg::MaterialConfig>::t > config_vector;
				creationParameters.SetAnyValue(
						creationParameters.Add("Configuration"),
						lean::any_vector< config_vector, lean::var_default<lean::any> >( config_vector(configRange.Begin, configRange.End) )
					);
			}
	}

	/// Gets a component by name.
	lean::cloneable_obj<lean::any, true> GetComponentByName(const utf8_ntri &name, const beCore::ParameterSet &parameters) const LEAN_OVERRIDE
	{
		SceneParameters sceneParameters = GetSceneParameters(parameters);

		return bec::any_resource_t<BoundMaterial>::t(
				GetBinding<BoundMaterial>( sceneParameters, 
					sceneParameters.ResourceManager->MaterialCache()->GetByName(name)
				)
			);
	}

	/// Gets a fitting file extension, if available.
	utf8_ntr GetFileExtension() const LEAN_OVERRIDE
	{
		return utf8_ntr("material.xml");
	}
	/// Gets a component from the given file.
	lean::cloneable_obj<lean::any, true> GetComponentByFile(const utf8_ntri &file,
		const beCore::Parameters &fileParameters, const beCore::ParameterSet &parameters) const LEAN_OVERRIDE
	{
		SceneParameters sceneParameters = GetSceneParameters(parameters);

		// TODO: Not what it is intended for!
		return bec::any_resource_t<BoundMaterial>::t(
				GetBinding<BoundMaterial>( sceneParameters, 
					sceneParameters.ResourceManager->MaterialCache()->NewByFile(file)
				)
			);
	}

	/// Gets the component type reflected.
	const beCore::ComponentType* GetType() const LEAN_OVERRIDE
	{
		return BoundMaterial::GetComponentType(); 
	}
};

static const beCore::ComponentReflectorPlugin< BoundMaterialReflector<RenderableMaterial> > RenderableMaterialReflectorPlugin(&RenderableMaterialType);
static const beCore::ComponentReflectorPlugin< BoundMaterialReflector<LightMaterial> > LightMaterialReflectorPlugin(&LightMaterialType);

/// Material configurations for use in component-based editing environments.
class MaterialConfigReflector : public beCore::ComponentReflector
{
	/// Gets principal component flags.
	uint4 GetComponentFlags() const LEAN_OVERRIDE
	{
		return bec::ComponentFlags::Creatable | bec::ComponentFlags::Cloneable; // bec::ComponentFlags::Filed | 
	}
	/// Gets specific component flags.
	uint4 GetComponentFlags(const lean::any &component) const LEAN_OVERRIDE
	{
		return bec::ComponentFlags::NameMutable; // | bec::ComponentFlags::FileMutable
	}

	/// Gets information on the components currently available.
	bec::ComponentInfoVector GetComponentInfo(const beCore::ParameterSet &parameters) const LEAN_OVERRIDE
	{
		return GetSceneParameters(parameters).ResourceManager->MaterialConfigCache()->GetInfo();
	}
	
	/// Gets the component info.
	bec::ComponentInfo GetInfo(const lean::any &component) const LEAN_OVERRIDE
	{
		bec::ComponentInfo result;

		if (const beg::MaterialConfig *material = any_cast_default<beg::MaterialConfig*>(component))
			if (const beg::MaterialConfigCache *cache = material->GetCache())
				result = cache->GetInfo(material);

		return result;
	}

	/// Creates a component from the given parameters.
	lean::cloneable_obj<lean::any, true> CreateComponent(const utf8_ntri &name, const beCore::Parameters &creationParameters,
		const beCore::ParameterSet &parameters, const lean::any *pPrototype, const lean::any *pReplace) const LEAN_OVERRIDE
	{
		SceneParameters sceneParameters = GetSceneParameters(parameters);

		lean::resource_ptr<beg::MaterialConfig> materialConfig = beg::CreateMaterialConfig();

		if (beg::MaterialConfig *pToBeReplaced = lean::any_cast_default<beg::MaterialConfig*>(pReplace))
			sceneParameters.ResourceManager->MaterialConfigCache()->Replace(pToBeReplaced, materialConfig);
		else
			sceneParameters.ResourceManager->MaterialConfigCache()->SetName(materialConfig, name);

		return bec::any_resource_t<beg::MaterialConfig>::t(materialConfig);
	}

	/// Sets the component name.
	void SetName(const lean::any &component, const utf8_ntri &name) const LEAN_OVERRIDE
	{
		if (beg::MaterialConfig *material = any_cast_default<beg::MaterialConfig*>(component))
			if (beg::MaterialConfigCache *cache = material->GetCache())
			{
				cache->SetName(material, name);
				return;
			}

		LEAN_THROW_ERROR_CTX("Unknown material config cannot be renamed", name.c_str());
	}

	/// Gets a component by name.
	lean::cloneable_obj<lean::any, true> GetComponentByName(const utf8_ntri &name, const beCore::ParameterSet &parameters) const LEAN_OVERRIDE
	{
		SceneParameters sceneParameters = GetSceneParameters(parameters);

		return bec::any_resource_t<beg::MaterialConfig>::t(
				sceneParameters.ResourceManager->MaterialConfigCache()->GetByName(name)
			);
	}

	/// Gets a fitting file extension, if available.
	utf8_ntr GetFileExtension() const LEAN_OVERRIDE
	{
		return utf8_ntr("materialconfig.xml");
	}

	/// Gets the component type reflected.
	const beCore::ComponentType* GetType() const LEAN_OVERRIDE
	{
		return beg::MaterialConfig::GetComponentType(); 
	}
};

static const beCore::ComponentReflectorPlugin<MaterialConfigReflector> MaterialConfigReflectorPlugin(beg::MaterialConfig::GetComponentType());

} // namespace

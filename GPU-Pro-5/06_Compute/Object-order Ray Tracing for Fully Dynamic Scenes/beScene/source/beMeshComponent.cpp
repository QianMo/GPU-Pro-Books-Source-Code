/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#include "beSceneInternal/stdafx.h"
#include <beScene/beAssembledMesh.h>

#include <beScene/beMesh.h>
#include <beScene/beRenderableMesh.h>
#include <beScene/beRenderableMaterial.h>

#include <beCore/beComponentReflector.h>
#include <beCore/beComponentTypes.h>

#include "beScene/beSerializationParameters.h"
#include "beScene/beResourceManager.h"
#include "beScene/beEffectDrivenRenderer.h"

#include <beScene/beMeshCache.h>
#include <beScene/beRenderableMeshCache.h>

#include <lean/logging/log.h>
#include <lean/logging/errors.h>

namespace beScene
{

extern const beCore::ComponentType AssembledMeshType;
extern const beCore::ComponentType RenderableMeshType;

/// Reflects meshes for use in component-based editing environments.
class MeshReflector : public beCore::ComponentReflector
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
		return GetSceneParameters(parameters).ResourceManager->MeshCache()->GetInfo();
	}
	
	/// Gets the component info.
	bec::ComponentInfo GetInfo(const lean::any &component) const LEAN_OVERRIDE
	{
		bec::ComponentInfo result;

		if (const AssembledMesh *mesh = any_cast_default<AssembledMesh*>(component))
			if (const MeshCache *cache = mesh->GetCache())
				result = cache->GetInfo(mesh);

		return result;
	}
	
	/// Sets the component name.
	void SetName(const lean::any &component, const utf8_ntri &name) const LEAN_OVERRIDE
	{
		if (AssembledMesh *mesh = any_cast_default<AssembledMesh*>(component))
			if (MeshCache *cache = mesh->GetCache())
			{
				cache->SetName(mesh, name);
				return;
			}

		LEAN_THROW_ERROR_CTX("Unknown mesh cannot be renamed", name.c_str());
	}

	/// Gets a component by name.
	lean::cloneable_obj<lean::any, true> GetComponentByName(const utf8_ntri &name, const beCore::ParameterSet &parameters) const LEAN_OVERRIDE
	{
		SceneParameters sceneParameters = GetSceneParameters(parameters);

		return bec::any_resource_t<AssembledMesh>::t(
				sceneParameters.ResourceManager->MeshCache()->GetByName(name)
			);
	}

	/// Gets a fitting file extension, if available.
	utf8_ntr GetFileExtension() const LEAN_OVERRIDE
	{
		return utf8_ntr("mesh");
	}
	/// Gets a component from the given file.
	lean::cloneable_obj<lean::any, true> GetComponentByFile(const utf8_ntri &file,
		const beCore::Parameters &fileParameters, const beCore::ParameterSet &parameters) const LEAN_OVERRIDE
	{
		SceneParameters sceneParameters = GetSceneParameters(parameters);

		return bec::any_resource_t<AssembledMesh>::t(
				sceneParameters.ResourceManager->MeshCache()->GetByFile(file)
			);
	}

	/// Gets the component type reflected.
	const beCore::ComponentType* GetType() const LEAN_OVERRIDE
	{
		return &AssembledMeshType; 
	}
};

static const beCore::ComponentReflectorPlugin<MeshReflector> MeshReflectorPlugin(&AssembledMeshType);

/// Reflects meshes for use in component-based editing environments.
class RenderableMeshReflector : public beCore::ComponentReflector
{
	/// Gets principal component flags.
	uint4 GetComponentFlags() const LEAN_OVERRIDE
	{
		return bec::ComponentFlags::Creatable | bec::ComponentFlags::Cloneable;
	}
	/// Gets specific component flags.
	uint4 GetComponentFlags(const lean::any &component) const LEAN_OVERRIDE
	{
		return bec::ComponentFlags::NameMutable;
	}

	/// Gets information on the components currently available.
	bec::ComponentInfoVector GetComponentInfo(const beCore::ParameterSet &parameters) const LEAN_OVERRIDE
	{
		return GetSceneParameters(parameters).Renderer->RenderableMeshes()->GetInfo();
	}
	
	/// Gets the component info.
	bec::ComponentInfo GetInfo(const lean::any &component) const LEAN_OVERRIDE
	{
		bec::ComponentInfo result;

		if (const RenderableMesh *mesh = any_cast_default<RenderableMesh*>(component))
			if (const RenderableMeshCache *cache = mesh->GetCache())
				result = cache->GetInfo(mesh);

		return result;
	}

	/// Gets a list of creation parameters.
	beCore::ComponentParameters GetCreationParameters() const LEAN_OVERRIDE
	{
		static const beCore::ComponentParameter parameters[] = {
				beCore::ComponentParameter(utf8_ntr("Source"), AssembledMesh::GetComponentType(), bec::ComponentParameterFlags::Deducible),
				beCore::ComponentParameter(utf8_ntr("Material"), RenderableMaterial::GetComponentType(), bec::ComponentParameterFlags::Optional)
			};

		return beCore::ComponentParameters(parameters, parameters + lean::arraylen(parameters));
	}
	/// Creates a component from the given parameters.
	lean::cloneable_obj<lean::any, true> CreateComponent(const utf8_ntri &name, const beCore::Parameters &creationParameters,
		const beCore::ParameterSet &parameters, const lean::any *pPrototype, const lean::any *pReplace) const LEAN_OVERRIDE
	{
		SceneParameters sceneParameters = GetSceneParameters(parameters);
		RenderableMesh *pPrototypeMesh = lean::any_cast_default<RenderableMesh*>(pPrototype);

		const AssembledMesh *sourceMesh = creationParameters.GetValueDefault<AssembledMesh*>("Source");
		if (!sourceMesh && pPrototypeMesh)
			sourceMesh = pPrototypeMesh->GetSource();

		lean::resource_ptr<RenderableMesh> mesh = ToRenderableMesh(*LEAN_THROW_NULL(sourceMesh), nullptr, true);

		if (pPrototypeMesh)
			TransferMaterials(*pPrototypeMesh, *mesh);

		if (RenderableMaterial *fillMaterial = creationParameters.GetValueDefault<RenderableMaterial*>("Material"))
		{
			// Construct subset-specific materials
			FillRenderableMesh(*mesh, fillMaterial, *sceneParameters.Renderer->RenderableMaterials);
			CacheMaterials(*mesh, *sceneParameters.ResourceManager->MaterialCache);
		}

		if (RenderableMesh *pToBeReplaced = lean::any_cast_default<RenderableMesh*>(pReplace))
			sceneParameters.Renderer->RenderableMeshes()->Replace(pToBeReplaced, mesh);
		else
			sceneParameters.Renderer->RenderableMeshes()->SetName(mesh, name);


		return bec::any_resource_t<RenderableMesh>::t(mesh);
	}
	
	// Gets a list of creation parameters.
	void GetCreationInfo(const lean::any &component, bec::Parameters &creationParameters, bec::ComponentInfo *pInfo = nullptr) const LEAN_OVERRIDE
	{
		if (const RenderableMesh *mesh = any_cast_default<RenderableMesh*>(component))
			if (const RenderableMeshCache *cache = mesh->GetCache())
			{
				if (pInfo)
					*pInfo = cache->GetInfo(mesh);

				creationParameters.SetValue<const AssembledMesh*>("Source", mesh->GetSource());
			}
	}

	/// Sets the component name.
	void SetName(const lean::any &component, const utf8_ntri &name) const LEAN_OVERRIDE
	{
		if (RenderableMesh *mesh = any_cast_default<RenderableMesh*>(component))
			if (RenderableMeshCache *cache = mesh->GetCache())
			{
				cache->SetName(mesh, name);
				return;
			}

		LEAN_THROW_ERROR_CTX("Unknown mesh cannot be renamed", name.c_str());
	}

	/// Gets a component by name.
	lean::cloneable_obj<lean::any, true> GetComponentByName(const utf8_ntri &name, const beCore::ParameterSet &parameters) const LEAN_OVERRIDE
	{
		SceneParameters sceneParameters = GetSceneParameters(parameters);

		return bec::any_resource_t<RenderableMesh>::t(
				sceneParameters.Renderer->RenderableMeshes()->GetByName(name)
			);
	}

	/// Gets the component type reflected.
	const beCore::ComponentType* GetType() const LEAN_OVERRIDE
	{
		return &RenderableMeshType; 
	}
};

static const beCore::ComponentReflectorPlugin<RenderableMeshReflector> RenderableMeshReflectorPlugin(&RenderableMeshType);

} // namespace

/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#include "beSceneInternal/stdafx.h"
#include "beScene/beInlineMaterialSerialization.h"

#include <beEntitySystem/beSerializationParameters.h>
#include <beEntitySystem/beSerializationTasks.h>

#include "beScene/beSerializationParameters.h"
#include "beScene/beResourceManager.h"

#include <beGraphics/beMaterialCache.h>
#include <beGraphics/beMaterialSerialization.h>

#include <lean/logging/errors.h>
#include <lean/xml/utility.h>
#include <lean/xml/numeric.h>
#include <set>

namespace beScene
{

namespace
{

/// Serializes a list of material configurations.
class MaterialConfigSerializer : public beCore::SaveJob
{
private:
	typedef std::set<const beg::MaterialConfig*> material_set;
	material_set m_materials;

public:
	/// Adds the given material for serialization.
	void AddMaterial(const beg::MaterialConfig *pMaterial)
	{
		LEAN_ASSERT_NOT_NULL( pMaterial );
		LEAN_ASSERT_NOT_NULL( pMaterial->GetCache() );
		m_materials.insert(pMaterial);
	}

	/// Saves anything, e.g. to the given XML root node.
	void Save(rapidxml::xml_node<lean::utf8_t> &root, beCore::ParameterSet &parameters, beCore::SerializationQueue<beCore::SaveJob> &queue) const
	{
		rapidxml::xml_document<utf8_t> &document = *root.document();

		rapidxml::xml_node<utf8_t> &materialsNode = *lean::allocate_node<utf8_t>(document, "materialconfigs");
		// ORDER: Append FIRST, otherwise parent document == nullptrs
		root.append_node(&materialsNode);

		for (material_set::const_iterator itMaterial = m_materials.begin(); itMaterial != m_materials.end(); itMaterial++)
		{
			const beg::MaterialConfig *material = *itMaterial;
			const beg::MaterialConfigCache *cache = material->GetCache();
			utf8_ntr name = cache->GetName(material);

			if (name.empty())
				LEAN_LOG_ERROR_MSG("Material configuration missing name, will be lost");
			else
			{
				rapidxml::xml_node<utf8_t> &materialNode = *lean::allocate_node<utf8_t>(document, "m");
				lean::append_attribute( document, materialNode, "name", name );
				// ORDER: Append FIRST, otherwise parent document == nullptr
				materialsNode.append_node(&materialNode);

				SaveConfig(*material, materialNode);
			}
		}
	}
};

/// Serializes a list of materials.
class MaterialSerializer : public beCore::SaveJob
{
private:
	typedef std::set<const beg::Material*> material_set;
	material_set m_materials;

public:
	/// Adds the given material for serialization.
	void AddMaterial(const beg::Material *pMaterial)
	{
		LEAN_ASSERT_NOT_NULL( pMaterial );
		LEAN_ASSERT_NOT_NULL( pMaterial->GetCache() );
		m_materials.insert(pMaterial);
	}

	/// Saves anything, e.g. to the given XML root node.
	void Save(rapidxml::xml_node<lean::utf8_t> &root, beCore::ParameterSet &parameters, beCore::SerializationQueue<beCore::SaveJob> &queue) const
	{
		rapidxml::xml_document<utf8_t> &document = *root.document();

		rapidxml::xml_node<utf8_t> &materialsNode = *lean::allocate_node<utf8_t>(document, "materials");
		// ORDER: Append FIRST, otherwise parent document == nullptrs
		root.append_node(&materialsNode);

		for (material_set::const_iterator itMaterial = m_materials.begin(); itMaterial != m_materials.end(); itMaterial++)
		{
			const beg::Material *material = *itMaterial;
			const beg::MaterialCache *cache = material->GetCache();
			utf8_ntr name = cache->GetName(material);

			if (name.empty())
				LEAN_LOG_ERROR_MSG("Material missing name, will be lost");
			else
			{
				rapidxml::xml_node<utf8_t> &materialNode = *lean::allocate_node<utf8_t>(document, "m");
				lean::append_attribute( document, materialNode, "name", name );
				// ORDER: Append FIRST, otherwise parent document == nullptr
				materialsNode.append_node(&materialNode);

				SaveMaterial(*material, materialNode);
			}
		}
	}
};

struct InlineSerializationToken;

} // namespace

// Schedules the given material for inline serialization.
void SaveMaterialConfig(const beg::MaterialConfig *pMaterial, beCore::ParameterSet &parameters, beCore::SerializationQueue<beCore::SaveJob> &queue)
{
	// Schedule material for serialization
	bees::GetOrMakeSaveJob<MaterialConfigSerializer, InlineSerializationToken>(
			parameters, "beScene.MaterialConfigSerializer", queue
		).AddMaterial(pMaterial);
}

// Schedules the given material for inline serialization.
void SaveMaterial(const beg::Material *pMaterial, beCore::ParameterSet &parameters, beCore::SerializationQueue<beCore::SaveJob> &queue)
{
	// Schedule material for serialization
	bees::GetOrMakeSaveJob<MaterialSerializer, InlineSerializationToken>(
			parameters, "beScene.MaterialSerializer", queue
		).AddMaterial(pMaterial);

	// Schedule configurations for serialization
	for (beg::Material::Configurations configs = pMaterial->GetConfigurations(); configs.Begin < configs.End; ++configs.Begin)
		SaveMaterialConfig(*configs.Begin, parameters, queue);
}

namespace
{

/// Loads a list of materials.
class MaterialConfigLoader : public beCore::LoadJob
{
public:
	/// Loads anything, e.g. to the given XML root node.
	void Load(const rapidxml::xml_node<lean::utf8_t> &root, beCore::ParameterSet &parameters, beCore::SerializationQueue<beCore::LoadJob> &queue) const
	{
		SceneParameters sceneParameters = GetSceneParameters(parameters);
		beg::TextureCache &textureCache = *LEAN_ASSERT_NOT_NULL(sceneParameters.ResourceManager)->TextureCache();
		beg::MaterialConfigCache &configCache = *LEAN_ASSERT_NOT_NULL(sceneParameters.ResourceManager)->MaterialConfigCache();

		bool bNoOverwrite = beEntitySystem::GetNoOverwriteParameter(parameters);

		for (const rapidxml::xml_node<utf8_t> *materialsNode = root.first_node("materialconfigs");
			materialsNode; materialsNode = materialsNode->next_sibling("materialconfigs"))
			for (const rapidxml::xml_node<utf8_t> *materialNode = materialsNode->first_node();
				materialNode; materialNode = materialNode->next_sibling())
			{
				utf8_ntr name = lean::get_attribute(*materialNode, "name");

				// Do not overwrite materials, if not permitted
				if (!bNoOverwrite || !configCache.GetByName(name))
				{
					lean::resource_ptr<beg::MaterialConfig> material = beg::CreateMaterialConfig();
					LoadNewConfig(*material, *materialNode, textureCache);

					try
					{
						configCache.SetName(material, name);
					}
					catch (const bec::ResourceCollision<beg::MaterialConfig> &e)
					{
						LEAN_ASSERT(!bNoOverwrite);
						configCache.Replace(e.Resource, material);
					}
				}
			}
	}
};

/// Loads a list of materials.
class MaterialLoader : public beCore::LoadJob
{
public:
	/// Loads anything, e.g. to the given XML root node.
	void Load(const rapidxml::xml_node<lean::utf8_t> &root, beCore::ParameterSet &parameters, beCore::SerializationQueue<beCore::LoadJob> &queue) const
	{
		SceneParameters sceneParameters = GetSceneParameters(parameters);
		beg::EffectCache &effectCache = *LEAN_ASSERT_NOT_NULL(sceneParameters.ResourceManager)->EffectCache();
		beg::MaterialConfigCache &configCache = *LEAN_ASSERT_NOT_NULL(sceneParameters.ResourceManager)->MaterialConfigCache();
		beg::MaterialCache &materialCache = *LEAN_ASSERT_NOT_NULL(sceneParameters.ResourceManager)->MaterialCache();

		bool bNoOverwrite = beEntitySystem::GetNoOverwriteParameter(parameters);

		for (const rapidxml::xml_node<utf8_t> *materialsNode = root.first_node("materials");
			materialsNode; materialsNode = materialsNode->next_sibling("materials"))
			for (const rapidxml::xml_node<utf8_t> *materialNode = materialsNode->first_node();
				materialNode; materialNode = materialNode->next_sibling())
			{
				utf8_ntr name = lean::get_attribute(*materialNode, "name");

				// Do not overwrite materials, if not permitted
				if (!bNoOverwrite || !materialCache.GetByName(name))
				{
					lean::resource_ptr<beg::Material> material = LoadMaterial(*materialNode, effectCache, configCache);

					try
					{
						materialCache.SetName(material, name);
					}
					catch (const bec::ResourceCollision<beg::Material> &e)
					{
						LEAN_ASSERT(!bNoOverwrite);
						materialCache.Replace(e.Resource, material);
					}

					// TODO: Move to utility function?
					// Fill in missing material configuration names
					for (beg::Material::Configurations configs = material->GetConfigurations(); configs.Begin < configs.End; ++configs.Begin)
						if (!configs.Begin[0]->GetCache())
							configCache.SetName(configs.Begin[0], configCache.GetUniqueName(name));
				}
			}
	}
};

} // namespace

const bec::LoadJob *CreateMaterialConfigLoader() { return new MaterialConfigLoader(); }
const bec::LoadJob *CreateMaterialLoader() { return new MaterialLoader(); }

} // namespace

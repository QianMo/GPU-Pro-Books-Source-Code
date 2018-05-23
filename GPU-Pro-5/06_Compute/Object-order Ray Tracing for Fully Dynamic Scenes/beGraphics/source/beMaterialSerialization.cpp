/*****************************************************/
/* breeze Engine Graphics Module  (c) Tobias Zirr 2011 */
/*****************************************************/

#include "beGraphicsInternal/stdafx.h"
#include "beGraphics/beMaterialSerialization.h"

#include "beGraphics/beTextureCache.h"
#include "beGraphics/beEffectCache.h"
#include "beGraphics/beMaterialConfigCache.h"

#include <beCore/bePropertySerialization.h>
#include <beCore/beValueTypes.h>

#include <lean/xml/numeric.h>
#include <lean/xml/xml_file.h>
#include <lean/functional/predicates.h>

#include <lean/logging/errors.h>
#include <lean/logging/log.h>

namespace beGraphics
{

// Saves the given effect to the given XML node.
void SaveEffect(const Effect &effect, rapidxml::xml_node<lean::utf8_t> &node)
{
	rapidxml::xml_document<utf8_t> &document = *LEAN_ASSERT_NOT_NULL(node.document());
	bool bIdentified = false;

	if (EffectCache *cache = effect.GetCache())
	{
		utf8_ntr file = cache->GetFile(&effect);
		bIdentified = !file.empty();

		if (bIdentified)
		{
			beCore::Exchange::utf8_string macroString, hookString;
			cache->GetParameters(&effect, &macroString, &hookString);

			lean::append_attribute(document, node, "effect", cache->GetPathResolver().Shorten(file));

			if (!macroString.empty())
				lean::append_attribute(document, node, "effectOptions", macroString);

			if (!hookString.empty())
				lean::append_attribute(document, node, "effectHooks", hookString);
		}
	}
	
	if (!bIdentified)
		LEAN_LOG_ERROR_CTX("Could not identify effect, effect will be lost", node.name());
}

// Loads an effect from the given XML node.
Effect* LoadEffect(const rapidxml::xml_node<lean::utf8_t> &node, EffectCache &cache, bool bThrow)
{
	utf8_ntr effectFile = lean::get_attribute(node, "effect");

	if (!effectFile.empty())
	{
		utf8_ntr effectMacros = lean::get_attribute(node, "effectOptions");
		utf8_ntr effectHooks = lean::get_attribute(node, "effectHooks");
		return cache.GetByFile(effectFile, effectMacros, effectHooks);
	}
	else if (bThrow)
		LEAN_THROW_ERROR_CTX("Node is missing a valid effect specification", node.name());
	else
		return nullptr;
}

// Loads an effect from the given XML node.
Effect* IdentifyEffect(const rapidxml::xml_node<lean::utf8_t> &node, const EffectCache &cache)
{
	utf8_ntr effectFile = lean::get_attribute(node, "effect");
	utf8_ntr effectMacros = lean::get_attribute(node, "effectOptions");
	utf8_ntr effectHooks = lean::get_attribute(node, "effectHooks");

	if (!effectFile.empty())
		return cache.IdentifyEffect(effectFile, effectMacros, effectHooks);
	else
		LEAN_THROW_ERROR_CTX("Node is missing a valid effect specification", node.name());
}

// Checks if the given XML node specifies an effect.
bool HasEffect(const rapidxml::xml_node<lean::utf8_t> &node)
{
	return !lean::get_attribute(node, "effect").empty();
}

// Saves the textures provided by the given object to the given XML node.
void SaveTextures(const TextureProvider &textures, rapidxml::xml_node<lean::utf8_t> &node)
{
	rapidxml::xml_document<utf8_t> &document = *LEAN_ASSERT_NOT_NULL(node.document());

	const uint4 textureCount = textures.GetTextureCount();

	if (textureCount > 0)
	{
		rapidxml::xml_node<utf8_t> &texturesNode = *lean::allocate_node<utf8_t>(document, "textures");
		// ORDER: Append FIRST, otherwise parent document == nullptrs
		node.append_node(&texturesNode);

		for (uint4 i = 0; i < textureCount; ++i)
		{
			rapidxml::xml_node<utf8_t> &textureNode = *lean::allocate_node(document, "t");
			// ORDER: Append FIRST, otherwise parent document == nullptrs
			texturesNode.append_node(&textureNode);

			utf8_ntr textureName = textures.GetTextureName(i);
			lean::append_attribute(document, textureNode, "n", textureName);

			if (textures.IsColorTexture(i))
				lean::append_attribute(document, textureNode, "color", "1");

			bool bIdentified = false;

			if (const TextureView *textureView = textures.GetTexture(i))
				if (const TextureCache *cache = textureView->GetCache())
					if (const Texture *texture = cache->GetTexture(textureView))
					{
						utf8_ntr file = cache->GetFile(texture);
						bIdentified = !file.empty();

						if (bIdentified)
							lean::append_attribute(document, textureNode, "file", cache->GetPathResolver().Shorten(file));
						else
						{
							utf8_ntr name = cache->GetName(texture);
							bIdentified = !name.empty();

							if (bIdentified)
								lean::append_attribute(document, textureNode, "name", name);
						}
					}

			if (!bIdentified)
				LEAN_LOG_ERROR_CTX("Could not identify texture, will be lost", textureName.c_str());
		}
	}
}

// Loads the textures provided by the given object from the given XML node.
void LoadTextures(TextureProvider &textures, const rapidxml::xml_node<lean::utf8_t> &node, TextureCache &cache)
{
	const uint4 textureCount = textures.GetTextureCount();

	uint4 nextTextureID = 0;

	for (const rapidxml::xml_node<lean::utf8_t> *texturesNode = node.first_node("textures");
		texturesNode; texturesNode = texturesNode->next_sibling("textures"))
		for (const rapidxml::xml_node<lean::utf8_t> *textureNode = texturesNode->first_node();
			textureNode; textureNode = textureNode->next_sibling())
		{
			utf8_ntr textureName = lean::get_attribute(*textureNode, "n");

			if (textureName.empty())
				textureName = utf8_ntr( textureNode->name() );

			utf8_ntr file = lean::get_attribute(*textureNode, "file");
			utf8_ntr name = lean::get_attribute(*textureNode, "name");
			bool bIsColor = lean::get_bool_attribute(*textureNode, "color", false);

			uint4 lowerTextureID = nextTextureID;
			uint4 upperTextureID = nextTextureID;

			if (!file.empty() || !name.empty())
				for (uint4 i = 0; i < textureCount; ++i)
				{
					// Perform bi-directional search: even == forward; odd == backward
					uint4 textureID = (lean::is_odd(i) | (upperTextureID == textureCount)) & (lowerTextureID != 0)
						? --lowerTextureID
						: upperTextureID++;

					if (textureName == textures.GetTextureName(textureID))
					{
						bool bSRGB = textures.IsColorTexture(textureID) || bIsColor;
						TextureView *texture = (!file.empty())
							? cache.GetViewByFile(file, bSRGB)
							: cache.GetView(cache.GetByName(name, true));
						textures.SetTexture(textureID, texture);

						// Start next search with next texture
						nextTextureID = textureID + 1;
						break;
					}
				}
			else
				LEAN_LOG_ERROR_CTX("Texture is missing both file and name specification", textureName.c_str());
		}
}

// Saves the given material configuration to the given XML node.
void SaveConfig(const MaterialConfig &config, rapidxml::xml_node<lean::utf8_t> &node)
{
	SaveProperties(config, node, true, true);
	SaveTextures(config, node);
}

// Load all properties from the given XML node.
void AddProperties(MaterialConfig &config, const rapidxml::xml_node<lean::utf8_t> &node)
{
	for (const rapidxml::xml_node<lean::utf8_t> *propertiesNode = node.first_node("properties");
		propertiesNode; propertiesNode = propertiesNode->next_sibling("properties"))
		for (const rapidxml::xml_node<lean::utf8_t> *propertyNode = propertiesNode->first_node();
			propertyNode; propertyNode = propertyNode->next_sibling())
		{
			bec::PropertyDeserializer serializer(config, *propertyNode);

			const utf8_ntr propertyName = serializer.Name();
			const utf8_ntr valueType = serializer.ValueType();
			uint4 valueCount = serializer.ValueCount();

			const bec::ValueTypeDesc *pDesc = bec::GetValueTypes().GetDesc(valueType);

			if (pDesc)
			{
				bec::PropertyDesc desc(*pDesc, valueCount, bec::Widget::Raw);
				uint4 propertyID = config.AddProperty(propertyName, desc);
				config.WriteProperty(propertyID, serializer);
			}
			else
				LEAN_LOG_ERROR_XCTX("Unknown property type", valueType.c_str(), propertyName.c_str());
		}
}

// Loads the textures provided by the given object from the given XML node.
void AddTextures(MaterialConfig &config, const rapidxml::xml_node<lean::utf8_t> &node, TextureCache &cache)
{
	for (const rapidxml::xml_node<lean::utf8_t> *texturesNode = node.first_node("textures");
		texturesNode; texturesNode = texturesNode->next_sibling("textures"))
		for (const rapidxml::xml_node<lean::utf8_t> *textureNode = texturesNode->first_node();
			textureNode; textureNode = textureNode->next_sibling())
		{
			utf8_ntr textureName = lean::get_attribute(*textureNode, "n");

			if (textureName.empty())
				textureName = utf8_ntr( textureNode->name() );

			bool bIsColor = lean::get_attribute(*textureNode, "color") == "1";
			uint4 textureID = config.AddTexture(textureName, bIsColor);

			utf8_ntr file = lean::get_attribute(*textureNode, "file");
			utf8_ntr name = lean::get_attribute(*textureNode, "name");
			TextureView *texture;

			if (!file.empty())
				texture = cache.GetViewByFile(file, bIsColor);
			else if (!name.empty())
				texture = cache.GetView(cache.GetByName(name, true));
			else
				LEAN_LOG_ERROR_CTX("Texture is missing both file and name specification", textureName.c_str());

			config.SetTexture(textureID, texture);
		}
}

// Load the given material configuration from the given (legacy) setup XML node.
void LoadNewConfig(MaterialConfig &config, const rapidxml::xml_node<lean::utf8_t> &node, TextureCache &textureCache)
{
	AddProperties(config, node);
	AddTextures(config, node, textureCache);
}

// Load the given material configuration from the given (legacy) setup XML node.
void LoadOverrideConfig(MaterialConfig &config, const rapidxml::xml_node<lean::utf8_t> &node, TextureCache &textureCache)
{
	LoadProperties(config, node);
	LoadTextures(config, node, textureCache);
}

// Saves the given material configuration to the given XML node.
void SaveConfig(const MaterialConfig &config, const utf8_ntri &file)
{
	lean::xml_file<lean::utf8_t> xml;
	rapidxml::xml_node<lean::utf8_t> &root = *lean::allocate_node<utf8_t>(xml.document(), "materialconfig");

	// ORDER: Append FIRST, otherwise parent document == nullptr
	xml.document().append_node(&root);
	SaveConfig(config, root);

	xml.save(file);
}

// Load the given material configuration from the given XML document.
void LoadConfig(MaterialConfig &config, const rapidxml::xml_document<lean::utf8_t> &document, TextureCache &textureCache)
{
	const rapidxml::xml_node<lean::utf8_t> *root = document.first_node("materialconfig");

	if (root)
		return LoadNewConfig(config, *root, textureCache);
	else
		LEAN_THROW_ERROR_MSG("Material configuration root node missing");
}

// Load the given material configuration from the given XML file.
void LoadConfig(MaterialConfig &config, const utf8_ntri &file, TextureCache &textureCache)
{
	LEAN_LOG("Attempting to load material configuration \"" << file.c_str() << "\"");
	LoadConfig( config, lean::xml_file<lean::utf8_t>(file).document(), textureCache );
	LEAN_LOG("Material configuration \"" << file.c_str() << "\" created successfully");
}



// Saves the textures provided by the given object to the given XML node.
void SaveMaterial(const Material &material, rapidxml::xml_node<lean::utf8_t> &node)
{
	rapidxml::xml_document<utf8_t> &document = *LEAN_ASSERT_NOT_NULL(node.document());

	for (beg::Material::Effects effects = material.GetEffects(); effects.Begin < effects.End; ++effects.Begin)
	{
		rapidxml::xml_node<utf8_t> &effectNode = *lean::allocate_node<utf8_t>(document, "effect");
		// ORDER: Append FIRST, otherwise parent document == nullptrs
		node.append_node(&effectNode);

		SaveEffect(**effects.Begin, effectNode);
	}

	for (beg::Material::Configurations configs = material.GetConfigurations(); configs.Begin < configs.End; ++configs.Begin)
	{
		rapidxml::xml_node<utf8_t> &configNode = *lean::allocate_node<utf8_t>(document, "config");
		// ORDER: Append FIRST, otherwise parent document == nullptrs
		node.append_node(&configNode);

		const MaterialConfig *materialConfig = *configs.Begin;
		bool bIdentified = false;

		if (const MaterialConfigCache *cache = materialConfig->GetCache())
		{
			utf8_ntr name = cache->GetName(materialConfig);
			bIdentified = !name.empty();

			if (bIdentified)
				lean::append_attribute(document, configNode, "name", name);
		}

		if (!bIdentified)
			LEAN_LOG_ERROR_MSG("Could not identify material config, will be lost");
	}
}

// Load the given material from the given (legacy) setup XML node.
void LoadSetup(Material &material, const rapidxml::xml_node<lean::utf8_t> &node, TextureCache &textureCache)
{
	lean::com_ptr<MaterialReflectionBinding> materialReflection = material.GetConfigBinding(0);
	LoadProperties(*materialReflection, node);
	LoadTextures(*materialReflection, node, textureCache);
}

// Load the given material from the given XML node.
void LoadMaterial(Material &material, const rapidxml::xml_node<lean::utf8_t> &node, MaterialConfigCache &configCache)
{
	// MONITOR: Support legacy material definitions
	if (node.first_node("setup"))
	{
		lean::resource_ptr<beg::MaterialConfig> materialConfig = beg::CreateMaterialConfig();
		material.SetConfigurations(&materialConfig.get(), 1);

		for (const rapidxml::xml_node<utf8_t> *setupNode = node.last_node("setup");
			setupNode; setupNode = setupNode->previous_sibling("setup"))
			LoadSetup(material, *setupNode, *configCache.GetTextureCache());
	}
	// New-style material definition
	else
	{
		std::vector<beg::MaterialConfig*> configs;

		for (const rapidxml::xml_node<lean::utf8_t> *configNode = node.first_node("config");
			configNode; configNode = configNode->next_sibling("config"))
		{
			utf8_ntr configName = lean::get_attribute(*configNode, "name");
		
			if (!configName.empty())
				configs.push_back( configCache.GetByName(configName, true) );
			else
				LEAN_LOG_ERROR_MSG("Material configuration reference is missing name specification");
		}

		material.SetConfigurations(&configs[0], (uint4) configs.size());
	}
}

// Creates a material from the given XML node.
lean::resource_ptr<Material, lean::critical_ref> LoadMaterial(const rapidxml::xml_node<lean::utf8_t> &node, EffectCache &effectCache, MaterialConfigCache &configCache)
{
	std::vector<beg::Effect*> effects;

	for (const rapidxml::xml_node<lean::utf8_t> *effectNode = node.first_node("effect");
		effectNode; effectNode = effectNode->next_sibling("effect"))
		effects.push_back( LoadEffect(*effectNode, effectCache) );

	if (beg::Effect *effect = LoadEffect(node, effectCache, effects.empty()))
		effects.push_back(effect);

	lean::resource_ptr<Material> material = beg::CreateMaterial(&effects[0], (uint4) effects.size(), effectCache);
	LoadMaterial(*material, node, configCache);
	return material.transfer();
}

// Saves the given material to the given XML file.
void SaveMaterial(const Material &material, const utf8_ntri &file)
{
	lean::xml_file<lean::utf8_t> xml;
	rapidxml::xml_node<lean::utf8_t> &root = *lean::allocate_node<utf8_t>(xml.document(), "material");

	// ORDER: Append FIRST, otherwise parent document == nullptr
	xml.document().append_node(&root);
	SaveMaterial(material, root);

	xml.save(file);
}

// Creates a material from the given XML document.
lean::resource_ptr<Material, lean::critical_ref> LoadMaterial(const rapidxml::xml_document<lean::utf8_t> &document, EffectCache &effectCache, MaterialConfigCache &configCache)
{
	const rapidxml::xml_node<lean::utf8_t> *root = document.first_node("material");

	if (root)
		return LoadMaterial(*root, effectCache, configCache);
	else
		LEAN_THROW_ERROR_MSG("Material root node missing");
}

// Creates a material from the given XML file.
lean::resource_ptr<Material, lean::critical_ref> LoadMaterial(const utf8_ntri &file, EffectCache &effectCache, MaterialConfigCache &configCache)
{
	LEAN_LOG("Attempting to load material \"" << file.c_str() << "\"");
	lean::resource_ptr<Material, lean::critical_ref> material = LoadMaterial( lean::xml_file<lean::utf8_t>(file).document(), effectCache, configCache );
	LEAN_LOG("Material \"" << file.c_str() << "\" created successfully");
	return material;
}

} // namespace

/******************************************************/
/* breeze Engine Graphics Module (c) Tobias Zirr 2011 */
/******************************************************/

#pragma once
#ifndef BE_GRAPHICS_MATERIAL_SERIALIZATION
#define BE_GRAPHICS_MATERIAL_SERIALIZATION

#include "beGraphics.h"
#include "beMaterial.h"
#include "beMaterialConfig.h"
#include <lean/rapidxml/rapidxml.hpp>

namespace beGraphics
{

class TextureCache;
class EffectCache;
class MaterialConfigCache;

/// Saves the given effect to the given XML node.
BE_GRAPHICS_API void SaveEffect(const Effect &effect, rapidxml::xml_node<lean::utf8_t> &node);
/// Loads an effect from the given XML node.
BE_GRAPHICS_API Effect* LoadEffect(const rapidxml::xml_node<lean::utf8_t> &node, EffectCache &cache, bool bThrow = true);
/// Loads an effect from the given XML node.
BE_GRAPHICS_API Effect* IdentifyEffect(const rapidxml::xml_node<lean::utf8_t> &node, const EffectCache &cache);
/// Checks if the given XML node specifies an effect.
BE_GRAPHICS_API bool HasEffect(const rapidxml::xml_node<lean::utf8_t> &node);

/// Saves the textures provided by the given object to the given XML node.
BE_GRAPHICS_API void SaveTextures(const TextureProvider &textures, rapidxml::xml_node<lean::utf8_t> &node);
/// Loads the textures provided by the given object from the given XML node.
BE_GRAPHICS_API void LoadTextures(TextureProvider &textures, const rapidxml::xml_node<lean::utf8_t> &node, TextureCache &cache);

/// Saves the given material configuration to the given XML node.
BE_GRAPHICS_API void SaveConfig(const MaterialConfig &config, rapidxml::xml_node<lean::utf8_t> &node);
/// Saves the given material configuration to the given XML file.
BE_GRAPHICS_API void SaveConfig(const MaterialConfig &config, const utf8_ntri &file);

/// Load the given material configuration from the given XML node.
BE_GRAPHICS_API void LoadOverrideConfig(MaterialConfig &config, const rapidxml::xml_node<lean::utf8_t> &node, TextureCache &textureCache);
/// Load the given material configuration from the given XML node.
BE_GRAPHICS_API void LoadNewConfig(MaterialConfig &config, const rapidxml::xml_node<lean::utf8_t> &node, TextureCache &textureCache);
/// Load the given material configuration from the given XML document.
BE_GRAPHICS_API void LoadConfig(MaterialConfig &config, const rapidxml::xml_document<lean::utf8_t> &document, TextureCache &textureCache);
/// Load the given material configuration from the given XML file.
BE_GRAPHICS_API void LoadConfig(MaterialConfig &config, const utf8_ntri &file, TextureCache &textureCache);


/// Saves the given material to the given XML node.
BE_GRAPHICS_API void SaveMaterial(const Material &material, rapidxml::xml_node<lean::utf8_t> &node);
/// Saves the given material to the given XML file.
BE_GRAPHICS_API void SaveMaterial(const Material &material, const utf8_ntri &file);

/// Load the given material from the given (legacy) XML node.
BE_GRAPHICS_API void LoadSetup(Material &material, const rapidxml::xml_node<lean::utf8_t> &node, TextureCache &textureCache);
/// Load the given material from the given XML node.
BE_GRAPHICS_API void LoadMaterial(Material &material, const rapidxml::xml_node<lean::utf8_t> &node, MaterialConfigCache &configCache);
/// Creates a material from the given XML node.
BE_GRAPHICS_API lean::resource_ptr<Material, lean::critical_ref> LoadMaterial(const rapidxml::xml_node<lean::utf8_t> &node, EffectCache &effectCache, MaterialConfigCache &configCache);
/// Creates a material from the given XML document.
BE_GRAPHICS_API lean::resource_ptr<Material, lean::critical_ref> LoadMaterial(const rapidxml::xml_document<lean::utf8_t> &document, EffectCache &effectCache, MaterialConfigCache &configCache);
/// Creates a material from the given XML file.
BE_GRAPHICS_API lean::resource_ptr<Material, lean::critical_ref> LoadMaterial(const utf8_ntri &file, EffectCache &effectCache, MaterialConfigCache &configCache);

} // namespace

#endif
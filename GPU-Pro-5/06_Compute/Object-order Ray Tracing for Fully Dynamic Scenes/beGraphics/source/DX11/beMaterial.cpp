/*****************************************************/
/* breeze Engine Graphics Module  (c) Tobias Zirr 2011 */
/*****************************************************/

#include "beGraphicsInternal/stdafx.h"
#include "beGraphics/DX11/beD3DXEffects11.h"
#include "beGraphics/DX11/beMaterial.h"
#include "beGraphics/DX11/beEffectConfig.h"
#include "beGraphics/DX11/beDeviceContext.h"
#include "beGraphics/DX11/beDevice.h"
#include "beGraphics/DX11/beEffect.h"

#include "beGraphics/beEffectCache.h"

#include "beGraphics/beTextureCache.h"
#include "beGraphics/DX11/beBuffer.h"

#include "beGraphics/DX/beError.h"

#include <beCore/beMany.h>

#include <beCore/bePropertyVisitor.h>
#include <beCore/beReflectionProperties.h>

#include <lean/functional/algorithm.h>
#include <lean/logging/log.h>
#include <lean/io/numeric.h>

namespace beGraphics
{

namespace DX11
{

struct Material::DataSource
{
	lean::resource_ptr<beg::MaterialConfig> config;
	const MaterialConfigRevision *configRevision;
	MaterialConfigRevision materialRevision;
	
	uint4 layerBaseIdx;
	uint4 layerMask;
	uint4 constantBufferMask;
	bec::Range<uint4> properties;
	bec::Range<uint4> constantDataLinks;
	bec::Range<uint4> textures;
	bec::Range<uint4> textureDataLinks;

	DataSource(beg::MaterialConfig *config, uint4 layerMask = -1, uint4 layerBaseIdx = -1)
		: layerBaseIdx(layerBaseIdx),
		layerMask(layerMask),
		config(config),
		configRevision(config->GetRevision()) { }
};

struct Material::Constants
{
	API::EffectConstantBuffer *constants;
	lean::com_ptr<API::Buffer> buffer;
	uint4 offset;

	bec::Range<uint4> dataLinks;

	Constants(API::EffectConstantBuffer *constants, API::Buffer* buffer, uint4 offset)
		: constants(constants),
		buffer(buffer),
		offset(offset) { }
};

struct Material::ConstantDataLink
{
	uint4 srcOffset;
	uint4 srcLength;
	uint4 destOffset;

	ConstantDataLink(uint4 srcOffset, uint4 srcLength, uint4 destOffset)
		: srcOffset(srcOffset),
		srcLength(srcLength),
		destOffset(destOffset) { }
};

struct Material::PropertyData
{
	uint4 offset;
	uint4 length;
	uint4 idInConfig;

	PropertyData(uint4 offset, uint4 length, uint4 idInConfig = -1)
		: offset(offset),
		length(length),
		idInConfig(idInConfig) { }
};

struct Material::TextureData
{
	lean::resource_ptr<const TextureView> pTexture;
	uint4 idInConfig;

	TextureData(uint4 idInConfig = -1)
		: idInConfig(idInConfig) { }
};

struct Material::TextureDataLink
{
	API::EffectShaderResource *variable;
	uint4 dataIdx;
	
	TextureDataLink(API::EffectShaderResource *variable, uint4 dataIdx = -1)
		: variable(variable),
		dataIdx(dataIdx) { }
};

namespace
{

/// Technique.
struct InternalMaterialTechnique
{
	beg::Material *material;
	lean::resource_ptr<beg::Technique> technique;

	/// Constructor.
	InternalMaterialTechnique(Material *material,
			Technique *technique)
		: material(material),
		technique(technique) { }
};

LEAN_LAYOUT_COMPATIBLE(MaterialTechnique, Material, InternalMaterialTechnique, material);
LEAN_LAYOUT_COMPATIBLE(MaterialTechnique, Technique, InternalMaterialTechnique, technique);
LEAN_SIZE_COMPATIBLE(MaterialTechnique, InternalMaterialTechnique);

}

struct Material::Technique
{
	InternalMaterialTechnique internal;

	bec::Range<uint4> constants;
	bec::Range<uint4> textures;

	Technique(Material *material, DX11::Technique *technique,
			bec::Range<uint4> constants,
			bec::Range<uint4> textures)
		: internal(material, technique),
		constants(constants),
		textures(textures) { }
};

namespace
{


struct CollectedEffectVariables
{
	const Effect *effect;
	uint4 layerBaseIdx;

	bec::Range<uint4> constants;
	bec::Range<uint4> textures;

	CollectedEffectVariables(const Effect *effect,
			uint4 layerBaseIdx,
			bec::Range<uint4> constants,
			bec::Range<uint4> textures)
		: effect(effect),
		layerBaseIdx(layerBaseIdx),
		constants(constants), 
		textures(textures) { }
};

struct CollectedPropertyDesc
{
	utf8_ntr name;
	PropertyDesc desc;

	CollectedPropertyDesc() { }
	CollectedPropertyDesc(utf8_ntr name,
		const PropertyDesc &desc)
			: name(name),
			desc(desc) { }
};

struct CollectedProperty : CollectedPropertyDesc
{
	uint4 layerMask;
	uint4 dataIdx;

	CollectedProperty(uint4 dataIdx,
		utf8_ntr name,
		const PropertyDesc &desc,
		uint4 layerMask)
			: CollectedPropertyDesc(name, desc),
			dataIdx(dataIdx),
			layerMask(layerMask) { }
};

struct CollectedPropertyData
{
	uint4 offset;
	uint4 length;
	uint4 cbufMask;

	CollectedPropertyData(uint4 cbufMask, uint4 offset, uint4 length)
			: cbufMask(cbufMask),
			offset(offset),
			length(length) { }
};

struct CollectedTextureDesc
{
	utf8_ntr name;
	bool bRaw;

	CollectedTextureDesc() : bRaw(false) { }
	CollectedTextureDesc(utf8_ntr name,
		bool bRaw)
			: name(name),
			bRaw(bRaw) { }
};

struct CollectedTexture : CollectedTextureDesc
{
	uint4 layerMask;
	uint4 textureIdx;

	CollectedTexture(uint4 textureIdx,
		utf8_ntr name,
		bool bRaw,
		uint4 layerMask)
			: CollectedTextureDesc(name, bRaw),
			textureIdx(textureIdx),
			layerMask(layerMask) { }
};

struct MaterialBindingContext
{
	typedef std::vector<CollectedProperty> properties_t;
	typedef std::vector<CollectedPropertyData> property_data_t;
	typedef std::vector<CollectedTexture> textures_t;

	const Material::Data *const data;
	uint4 cbIdx;
	uint4 texIdx;

	properties_t properties;
	property_data_t propertyData;
	textures_t textures;

	MaterialBindingContext(const Material::Data *data)
		: data(data),
		cbIdx(),
		texIdx() { }
};

template <class Type>
LEAN_INLINE bool less_or_equal_and(const Type &l, const Type &r, bool eq_order)
{
	return l < r || l == r && eq_order;
}

struct PropertySorter
{
	LEAN_INLINE bool operator ()(const CollectedPropertyDesc &l, const CollectedPropertyDesc &r) const
	{
		return less_or_equal_and(l.name, r.name,
			less_or_equal_and(l.desc.TypeDesc, r.desc.TypeDesc,
				l.desc.Count < r.desc.Count
			)
		);
	}
};

struct TextureSorter
{
	LEAN_INLINE bool operator ()(const CollectedTextureDesc &l, const CollectedTextureDesc &r) const
	{
		return less_or_equal_and(l.name, r.name,
				l.bRaw < r.bRaw
			);
	}
};

// Gets all properties.
uint4 CollectProperties(MaterialBindingContext &bindingContext, const EffectConfig &baseConfig, uint4 layerIdx)
{
	uint4 layerCount = 1;
	const Material::Data &data = *bindingContext.data;

	const EffectConfig::ConstantBufferRange effectCBs = baseConfig.GetConstantBufferInfo();
	const EffectConfig::ConstantRange effectConstants = baseConfig.GetConstantInfo();

	// Scan all constant buffers for properties
	for (const EffectConstantBufferInfo *itEffectCB = effectCBs.Begin; itEffectCB < effectCBs.End; ++itEffectCB)
	{
		uint4 cbufIdx = bindingContext.cbIdx++;
		LEAN_ASSERT(cbufIdx < lean::size_info<uint4>::bits);
		uint4 cbufMask = 1 << cbufIdx;

		const Material::Constants &constants = data.constants[cbufIdx];
		LEAN_ASSERT(constants.constants == itEffectCB->Variable);

		for (uint4 effectPropertyIdx = itEffectCB->Constants.Begin; effectPropertyIdx < itEffectCB->Constants.End; ++effectPropertyIdx)
		{
			utf8_ntr propertyName = baseConfig.GetPropertyName(effectPropertyIdx);
			const bec::PropertyDesc &propertyDesc = baseConfig.GetPropertyDesc(effectPropertyIdx);
			
			const EffectConstantInfo &constantInfo = effectConstants[effectPropertyIdx];
			uint4 propertyDataOffset = constants.offset + constantInfo.BufferOffset;
			uint4 propertyDataLength = propertyDesc.Count * propertyDesc.TypeDesc->Info.size;

			uint4 propertyLayerIdx = layerIdx + constantInfo.LayerOffset;
			LEAN_ASSERT(propertyLayerIdx < lean::size_info<uint4>::bits);
			uint4 propertyLayerMask = 1 << propertyLayerIdx;

			layerCount = max(constantInfo.LayerOffset + 1, layerCount);

			// Add new property
			uint4 propertyIdx = (uint4) bindingContext.propertyData.size();
			bindingContext.propertyData.push_back(
					CollectedPropertyData(cbufMask, propertyDataOffset, propertyDataLength)
				);
			bindingContext.properties.push_back(
					CollectedProperty(propertyIdx, propertyName, propertyDesc, propertyLayerMask)
				);
		}
	}

	return layerCount;
}

// Binds all properties.
void BindProperties(Material::Data &data, MaterialBindingContext &bindingContext, Material::DataSource &dataSource)
{
	// Initialize data source binding
	dataSource.constantBufferMask = 0;
	dataSource.properties.Begin = (uint4) data.properties.size();
	dataSource.constantDataLinks.Begin = (uint4) data.constantDataLinks.size();

	for (uint4 srcPropertyIdx = 0, srcPropertyCount = dataSource.config->GetPropertyCount(); srcPropertyIdx < srcPropertyCount; ++srcPropertyIdx)
	{
		utf8_ntr propertyName = dataSource.config->GetPropertyName(srcPropertyIdx);
		const bec::PropertyDesc &propertyDesc = dataSource.config->GetPropertyDesc(srcPropertyIdx);
		
		// Get range of properties matching data source
		CollectedPropertyDesc matchDesc(propertyName, propertyDesc);
		MaterialBindingContext::properties_t::iterator
			matchesBegin = std::lower_bound(bindingContext.properties.begin(), bindingContext.properties.end(), matchDesc, PropertySorter()),
			matchesEnd = std::upper_bound(bindingContext.properties.begin(), bindingContext.properties.end(), matchDesc, PropertySorter());

		const CollectedPropertyData *firstMatchData = nullptr;

		// Bind all matching properties
		for (MaterialBindingContext::properties_t::iterator match = matchesBegin; match < matchesEnd; ++match)
			if (match->layerMask & dataSource.layerMask)
			{
				const CollectedPropertyData &propertyData = bindingContext.propertyData[match->dataIdx];

				if (firstMatchData)
					// Emit link
					data.constantDataLinks.push_back(
							Material::ConstantDataLink(firstMatchData->offset, firstMatchData->length, propertyData.offset)
						);
				else
				{
					// Emit binding
					data.properties.push_back(
							Material::PropertyData(propertyData.offset, propertyData.length, srcPropertyIdx)
						);

					firstMatchData = &propertyData;
				}

				// Update dependencies
				dataSource.constantBufferMask |= propertyData.cbufMask;

				// Exclude from subsequent binding
				match->layerMask = 0;
			}
	}

	// Finalize data source binding range
	dataSource.properties.End = (uint4) data.properties.size();
	dataSource.constantDataLinks.End = (uint4) data.constantDataLinks.size();
}

// Gets all textures.
uint4 CollectTextures(MaterialBindingContext &bindingContext, const EffectConfig &baseConfig, uint4 layerIdx)
{
	uint4 layerCount = 1;
	const Material::Data &data = *bindingContext.data;

	const EffectConfig::ResourceRange effectResourceRange = baseConfig.GetResourceInfo();

	// Scan all variables for textures
	for (const EffectResourceInfo *itEffectRes = effectResourceRange.Begin; itEffectRes < effectResourceRange.End; ++itEffectRes)
	{
		uint4 textureIdx = bindingContext.texIdx++;
		const Material::TextureDataLink &texture = data.textureDataLinks[textureIdx];
		LEAN_ASSERT(texture.variable == itEffectRes->Variable);

		utf8_ntr textureName = baseConfig.GetTextureName(itEffectRes - effectResourceRange.Begin);

		uint4 textureLayerIdx = layerIdx + itEffectRes->LayerOffset;
		LEAN_ASSERT(textureLayerIdx < lean::size_info<uint4>::bits);
		uint4 textureLayerMask = 1 << textureLayerIdx;

		layerCount = max(itEffectRes->LayerOffset + 1, textureLayerIdx);

		bindingContext.textures.push_back(
				CollectedTexture(textureIdx, textureName, itEffectRes->IsRaw, textureLayerMask)
			);
	}

	return layerCount;
}

// Gets all textures.
void BindTextures(Material::Data &data, MaterialBindingContext &bindingContext, Material::DataSource &dataSource)
{
	// Initialize data source binding range
	dataSource.textures.Begin = (uint4) data.textures.size();
	dataSource.textureDataLinks.Begin = (uint4) data.textureDataLinks.size();

	MaterialConfig *srcConfig = &ToImpl(*dataSource.config);

	for (uint4 srcTextureIdx = 0, srcTextureCount = dataSource.config->GetTextureCount();
		srcTextureIdx < srcTextureCount; ++srcTextureIdx)
	{
		utf8_ntr textureName = srcConfig->GetTextureName(srcTextureIdx);
		bool bTextureRaw = !srcConfig->IsColorTexture(srcTextureIdx);

		CollectedTextureDesc matchDesc(textureName, bTextureRaw);
		MaterialBindingContext::textures_t::iterator
			matchesBegin = std::lower_bound(bindingContext.textures.begin(), bindingContext.textures.end(), matchDesc, TextureSorter()),
			matchesEnd = std::upper_bound(bindingContext.textures.begin(), bindingContext.textures.end(), matchDesc, TextureSorter());

		uint4 textureIdx = -1;

		// Bind all matching textures
		for (MaterialBindingContext::textures_t::iterator match = matchesBegin; match < matchesEnd; ++match)
			if (match->layerMask & dataSource.layerMask)
			{
				if (textureIdx == -1)
				{
					// Emit binding
					textureIdx  = (uint4) data.textures.size();
					data.textures.push_back(
							Material::TextureData(srcTextureIdx)
						);
				}

				// Emit link
				data.textureDataLinks[match->textureIdx].dataIdx = textureIdx;
				
				// Exclude from subsequent binding
				match->layerMask = 0;
			}
	}

	dataSource.textures.End = (uint4) data.textures.size();
	dataSource.textureDataLinks.Begin = (uint4) data.textureDataLinks.size();
}

void LoadBindingContext(MaterialBindingContext &bindingCtx, const Material::DataSource *sources, size_t sourceCount, bool bInitDS = false)
{
	uint4 currentLayerBaseIdx = 0;
	uint4 currentLayerIdx = 0;
	uint4 currentLayerCount = 1;

	// Recollect all properties & textures for binding
	for (const Material::DataSource *source = sources, *sourcesEnd = sources + sourceCount; source < sourcesEnd; ++source)
//	for (datasources_t::iterator it = m.dataSources.end() - m.immutableBaseConfigCount, itEnd = m.dataSources.end(); it < itEnd; ++it)
	{
		LEAN_ASSERT(currentLayerBaseIdx <= source->layerBaseIdx);

		if (currentLayerBaseIdx != source->layerBaseIdx)
		{
			currentLayerIdx += currentLayerCount;
			currentLayerCount = 1;
			currentLayerBaseIdx = source->layerBaseIdx;
		}

		currentLayerCount = max( CollectProperties(bindingCtx, static_cast<const DX11::EffectConfig&>(*source->config), currentLayerIdx), currentLayerCount );
		currentLayerCount = max( CollectTextures(bindingCtx, static_cast<const DX11::EffectConfig&>(*source->config), currentLayerIdx), currentLayerCount );
		
		uint4 layerMask = ((1 << currentLayerCount) - 1) << currentLayerIdx;
		if (bInitDS)
			const_cast<Material::DataSource*>(sources)->layerMask = layerMask;
		else
			LEAN_ASSERT(sources->layerMask == layerMask);
	}

	std::sort(bindingCtx.properties.begin(), bindingCtx.properties.end(), PropertySorter());
	std::sort(bindingCtx.textures.begin(), bindingCtx.textures.end(), TextureSorter());
}

// Collects all constant buffers.
bec::Range<uint4> AddConstants(Material::Data &data, const EffectConfig &baseConfig, API::Device *device)
{
	bec::Range<uint4> relevantConstants = bec::MakeRangeN((uint4) data.constants.size(), 0U);
	
	const EffectConfig::ConstantBufferRange effectCBs = baseConfig.GetConstantBufferInfo();
	data.constants.reserve_grow_by(Size(effectCBs));
	
	uint4 backingStoreOffset = (uint4) data.backingStore.size();

	// Scan all constant buffers for properties
	for (const EffectConstantBufferInfo *itEffectCB = effectCBs.Begin; itEffectCB < effectCBs.End; ++itEffectCB)
	{
		// Add constant buffer
		data.constants.push_back(
				Material::Constants(
					itEffectCB->Variable,
					CreateConstantBuffer(device, itEffectCB->Size).get(),
					backingStoreOffset
				)
			);

		backingStoreOffset += itEffectCB->Size;
	}

	// Allocate bytes for backing store
	data.backingStore.resize(backingStoreOffset);

	relevantConstants.End = (uint4) data.constants.size();
	return relevantConstants;
}

// Gets all textures.
bec::Range<uint4> AddTextures(Material::Data &data, const EffectConfig &baseConfig)
{
	bec::Range<uint4> relevantTextures = bec::MakeRangeN((uint4) data.textureDataLinks.size(), 0U);

	const EffectConfig::ResourceRange effectResourceRange = baseConfig.GetResourceInfo();
	data.textureDataLinks.reserve_grow_by(Size(effectResourceRange));

	// Scan all variables for textures
	for (const EffectResourceInfo *itEffectRes = effectResourceRange.Begin; itEffectRes < effectResourceRange.End; ++itEffectRes)
	{
		// Add texture
		data.textureDataLinks.push_back(
				Material::TextureDataLink(itEffectRes->Variable)
			);
	}

	relevantTextures.End = (uint4) data.textureDataLinks.size();
	return relevantTextures;
}

struct MaterialLoadContext
{
	Material *const material;
	Material::Data *const data;
	Material::datasources_t *const baseConfig;
	Material::techniques_t *const techniques;

	uint4 layerIdx;
	typedef std::vector<CollectedEffectVariables> effect_vars_t;
	effect_vars_t effectVars;

	MaterialLoadContext(Material *material, Material::Data *data, Material::datasources_t *baseConfig, Material::techniques_t *techniques)
		: material(material),
		data(data),
		baseConfig(baseConfig),
		techniques(techniques),
		layerIdx() { }
};

/// Adds the given effect to the given material.
CollectedEffectVariables GetEffectVariables(const Effect *effect, uint4 layerBaseIdx, MaterialLoadContext &context)
{
	LEAN_ASSERT( effect );
	EffectConfig *defaultConfig = LEAN_ASSERT_NOT_NULL( effect->GetConfig() );

	for (MaterialLoadContext::effect_vars_t::const_iterator it = context.effectVars.begin(), itEnd = context.effectVars.end(); it < itEnd; ++it)
		if (it->effect == effect)
			return *it;

	lean::com_ptr<API::Device> device;
	effect->Get()->GetDevice(device.rebind());

	// Add effect
	context.effectVars.push_back(
			CollectedEffectVariables(
				effect,
				layerBaseIdx,
				AddConstants(*context.data, *defaultConfig, device),
				AddTextures(*context.data, *defaultConfig)
			)
		);

	// Add default data
	context.baseConfig->push_back( Material::DataSource(defaultConfig, -1, layerBaseIdx) );

	return context.effectVars.back();
}

void LoadTechniques(MaterialLoadContext &creatctx, const CollectedEffectVariables &effectCtx, beg::EffectCache &effectCache);

/// Adds the given technique.
void AddTechnique(MaterialLoadContext &loadCtx, const CollectedEffectVariables &effectCtx, Technique *effectTechnique, beg::EffectCache &effectCache)
{
	const char *includeEffect = "";

	// Check for & load effect cross-references
	if (SUCCEEDED(effectTechnique->Get()->GetAnnotationByName("IncludeEffect")->AsString()->GetString(&includeEffect)) )
	{
		if (lean::char_traits<char>::equal(includeEffect, "#this"))
			includeEffect = bec::GetCachedFile<utf8_ntr>(effectTechnique->GetEffect()).c_str();
		
		bec::Exchange::vector_t<beGraphics::EffectMacro>::t includeMacros;
		bec::Exchange::vector_t<beGraphics::EffectHook>::t includeHooks;

		// Check for & load defines & hooks
		ID3DX11EffectStringVariable *pIncludeDefinitions = effectTechnique->Get()->GetAnnotationByName("IncludeDefinitions")->AsString();
		ID3DX11EffectStringVariable *pIncludeHooks = effectTechnique->Get()->GetAnnotationByName("IncludeHooks")->AsString();

		if (pIncludeDefinitions->IsValid())
		{
			typedef std::vector<const char*> string_pointers_t;
			string_pointers_t stringPtrs(beg::GetDesc(pIncludeDefinitions->GetType()).Elements);

			BE_THROW_DX_ERROR_MSG(
				pIncludeDefinitions->GetStringArray(&stringPtrs[0], 0, static_cast<UINT>(stringPtrs.size())),
				"ID3DX11EffectStringVariable::GetStringArray()" );
			includeMacros.reserve(stringPtrs.size());

			for (string_pointers_t::const_iterator it = stringPtrs.begin(); it != stringPtrs.end(); ++it)
			{
				utf8_ntr string(*it);
				// NOTE: Empty arrays unsupported, therefore check for empty dummy strings
				if (!string.empty())
				{
					if (string != "#this")
						includeMacros.push_back( beGraphics::EffectMacro(string, utf8_ntr("")) );
					else
					{
						beg::Effect const* effect = effectTechnique->GetEffect();
						if (beg::EffectCache* cache = effect->GetCache())
							cache->GetParameters(effect, &includeMacros, nullptr);
					}
				}
			}
		}

		if (pIncludeHooks->IsValid())
		{
			typedef std::vector<const char*> string_pointers_t;
			string_pointers_t stringPtrs(beg::GetDesc(pIncludeHooks->GetType()).Elements);

			BE_THROW_DX_ERROR_MSG(
				pIncludeHooks->GetStringArray(&stringPtrs[0], 0, static_cast<UINT>(stringPtrs.size())),
				"ID3DX11EffectStringVariable::GetStringArray()" );
			includeHooks.reserve(stringPtrs.size());

			for (string_pointers_t::const_iterator it = stringPtrs.begin(); it != stringPtrs.end(); ++it)
			{
				utf8_ntr string(*it);
				// NOTE: Empty arrays unsupported, therefore check for empty dummy strings
				if (!string.empty())
					includeHooks.push_back( beGraphics::EffectHook(string) );
			}
		}

		// Load linked effect
		const Effect *linkedEffect = ToImpl(
				effectCache.GetByFile(includeEffect, &includeMacros[0], includeMacros.size(), &includeHooks[0], includeHooks.size())
			);
		CollectedEffectVariables linkedEffectVars = GetEffectVariables(linkedEffect, effectCtx.layerBaseIdx, loadCtx);

		const char *includeTechnique = "";

		// Check for single technique
		if (SUCCEEDED(effectTechnique->Get()->GetAnnotationByName("IncludeTechnique")->AsString()->GetString(&includeTechnique)))
		{
			ID3DX11EffectTechnique *pLinkedTechniqueDX = linkedEffect->Get()->GetTechniqueByName(includeTechnique);

			if (!pLinkedTechniqueDX->IsValid())
				LEAN_THROW_ERROR_CTX("Unable to locate technique specified in IncludeTechnique", includeTechnique);

			// Add single linked technique
			lean::resource_ptr<Technique> linkedTechnique = new_resource Technique(linkedEffect, pLinkedTechniqueDX);
			AddTechnique(loadCtx, linkedEffectVars, linkedTechnique, effectCache);
		}
		else
			// Add all techniques
			LoadTechniques(loadCtx, linkedEffectVars, effectCache);
	}
	else
		// Simply add the technique
		loadCtx.techniques->push_back(
				Material::Technique(
					loadCtx.material, effectTechnique,
					effectCtx.constants,
					effectCtx.textures
				)
			);
}

/// Loads techniques & linked effects.
void LoadTechniques(MaterialLoadContext &loadCtx, const CollectedEffectVariables &effectCtx, beGraphics::EffectCache &effectCache)
{
	api::Effect *effectDX = *effectCtx.effect;
	D3DX11_EFFECT_DESC effectDesc = GetDesc(effectDX);

	loadCtx.techniques->reserve(effectDesc.Techniques);

	for (UINT id = 0; id < effectDesc.Techniques; ++id)
	{
		api::EffectTechnique *techniqueDX = effectDX->GetTechniqueByIndex(id);
		if (!techniqueDX->IsValid())
			LEAN_THROW_ERROR_MSG("ID3DX11Effect::GetTechniqueByIndex()");

		lean::resource_ptr<Technique> technique = new_resource Technique(effectCtx.effect, techniqueDX);
		AddTechnique(loadCtx, effectCtx, technique, effectCache);
	}
}

} // namespace

// Constructor.
Material::Material(const beg::Effect *const* effects, uint4 effectCount, beg::EffectCache &effectCache)
{
	LEAN_ASSERT(effects && effectCount > 0);

	// Load techniques & effect variables
	{
		MaterialLoadContext loadCtx(this, &m.data, &m.dataSources, &m.techniques);

		for (uint4 effectIdx = 0; effectIdx < effectCount; ++effectIdx)
		{
			const Effect *effect = LEAN_ASSERT_NOT_NULL( ToImpl(effects[effectIdx]) );
		
			// Add as primary effect
			CollectedEffectVariables effectVars = GetEffectVariables(effect, effectIdx, loadCtx);
			m.effects.push_back(effect);

			// Add techniques (and linked effects)
			LoadTechniques(loadCtx, effectVars, effectCache);
		}

		// All data sources collected so far are effect base configurations that may never change
		m.immutableBaseConfigCount = (uint4) m.dataSources.size();
	}

	// Collect effects that were linked in by other effects
	{
		m.hiddenEffectCount = 0;
	
		for (techniques_t::const_iterator it = m.techniques.begin(), itEnd = m.techniques.end(); it != itEnd; ++it)
		{
			bool bNew;
			lean::push_unique(m.effects, it->internal.technique->GetEffect(), &bNew);
			m.hiddenEffectCount += bNew;
		}
	}
	
	// NOTE: Bind lazily
	m.bBindingChanged = true;
}

// Copies the given material.
Material::Material(const Material &right)
	: m(right.m)
{
	// IMPORTANT: Relink techniques
	for (techniques_t::iterator it = m.techniques.begin(), itEnd = m.techniques.end(); it < itEnd; ++it)
		it->internal.material = this;

	// IMPORTANT: Clone constant buffers
	for (constants_t::iterator it = m.data.constants.begin(), itEnd = m.data.constants.end(); it < itEnd; ++it)
		it->buffer = CloneBuffer(*it->buffer);
}

// Destructor.
Material::~Material()
{
}

// Rebinds the data sources.
void Material::Rebind()
{
	MaterialBindingContext bindingCtx(&m.data);
	LoadBindingContext(
			bindingCtx,
			m.dataSources.data() + m.dataSources.size() - m.immutableBaseConfigCount, m.immutableBaseConfigCount,
			true
		);

	for (datasources_t::iterator it = m.dataSources.begin(), itEnd = m.dataSources.end(); it < itEnd; ++it)
	{
		DataSource &dataSource = *it;

		// Rebind data sources
		BindProperties(m.data, bindingCtx, dataSource);
		BindTextures(m.data, bindingCtx, dataSource);

		// Invalidate data, validate structure
		dataSource.materialRevision.Data = dataSource.configRevision->Data - 1;
		dataSource.materialRevision.Structure = dataSource.configRevision->Structure;
	}

	m.bBindingChanged = false;
}

// Applys the setup.
void Material::Apply(const MaterialTechnique *publicTechnique, const beGraphics::DeviceContext &context)
{
	bool bStructureChanged = m.bBindingChanged;
	bool bDataChanged = false;

	// TODO: Cache change detection via moving frame flag?
	for (uint4 dsIdx = 0, dsCount = (uint4) m.dataSources.size() - m.immutableBaseConfigCount; dsIdx < dsCount; ++dsIdx)
	{
		DataSource &dataSource = m.dataSources[dsIdx];
		
		// Check if bindings still up to date
		bStructureChanged |= (dataSource.materialRevision.Structure != dataSource.configRevision->Structure);
		bDataChanged |= (dataSource.materialRevision.Data != dataSource.configRevision->Data);
	}

	// Rebuild bindings, if outdated
	if (bStructureChanged)
	{
		Rebind();
		bDataChanged = true;
	}

	if (bDataChanged)
	{
		uint4 cbufChangedMask = 0;

		// Collect changed data from data sources
		for (uint4 dsIdx = 0, dsCount = (uint4) m.dataSources.size(); dsIdx < dsCount; ++dsIdx)
		{
			DataSource &dataSource = m.dataSources[dsIdx];

			// Check if data still up to date
			if (dataSource.materialRevision.Data != dataSource.configRevision->Data)
			{
				// Update property data from data source
				{
					for (bec::Range<uint4> dsProperties = dataSource.properties; dsProperties.Begin < dsProperties.End; ++dsProperties.Begin)
					{
						const PropertyData &data = m.data.properties[dsProperties.Begin];
						ToImpl(*dataSource.config).GetMaterialPropertyRaw(data.idInConfig, &m.data.backingStore[data.offset], data.length);
					}

					// Spread updated data to all constant buffers
					for (bec::Range<uint4> dsLinks = dataSource.constantDataLinks; dsLinks.Begin < dsLinks.End; ++dsLinks.Begin)
					{
						const ConstantDataLink &link = m.data.constantDataLinks[dsLinks.Begin];
						memcpy(&m.data.backingStore[link.destOffset], &m.data.backingStore[link.srcOffset], link.srcLength);
					}

					// Rebuild dependent constant buffers
					cbufChangedMask |= dataSource.constantBufferMask;
				}

				// Update textures from data source
				{
					for (bec::Range<uint4> dsTextures = dataSource.textures; dsTextures.Begin < dsTextures.End; ++dsTextures.Begin)
					{
						TextureData &data = m.data.textures[dsTextures.Begin];
						data.pTexture = ToImpl(*dataSource.config).GetTexture(data.idInConfig);
					}
				}

				// Data up to date
				dataSource.materialRevision.Data = dataSource.configRevision->Data;
			}
		}

		// Upload changed data to constant buffers
		if (cbufChangedMask)
			for (uint4 cbufIdx = 0, cbufCount = (uint4) m.data.constants.size(); cbufIdx < cbufCount; ++cbufIdx)
				// Update changed constant buffers
				if (cbufChangedMask & (1 << cbufIdx))
				{
					const Constants &cbuf = m.data.constants[cbufIdx];
					ToImpl(context)->UpdateSubresource(cbuf.buffer, 0, nullptr, &m.data.backingStore[cbuf.offset], 0, 0);
				}
	}
	
	const Technique &technique = *reinterpret_cast<const Technique*>(publicTechnique);
	LEAN_ASSERT(m.techniques.begin() <= &technique && &technique < m.techniques.end());

	// Bind technique-related constant buffers
	for (uint4 cbufIdx = technique.constants.Begin; cbufIdx < technique.constants.End; ++cbufIdx)
	{
		// Bind constant buffer
		const Constants &cbuf = m.data.constants[cbufIdx];
		cbuf.constants->SetConstantBuffer(cbuf.buffer);
	}

	// Bind technique-related textures
	for (uint4 texDataIdx = technique.textures.Begin; texDataIdx < technique.textures.End; ++texDataIdx)
	{
		const TextureDataLink &link = m.data.textureDataLinks[texDataIdx];
		const TextureData &data = m.data.textures[link.dataIdx];

		// Bind texture
		link.variable->SetResource( (data.pTexture) ? data.pTexture->GetView() : nullptr );
	}
}

/// Gets the effects.
Material::Effects Material::GetEffects() const
{
	return bec::MakeRangeN( &m.effects[0].get(), m.effects.size() - m.hiddenEffectCount );
}

/// Gets the effects.
Material::Effects Material::GetLinkedEffects() const
{
	return bec::MakeRangeN( &m.effects[0].get() + m.effects.size() - m.hiddenEffectCount, m.hiddenEffectCount );
}

// Gets the number of techniques.
uint4 Material::GetTechniqueCount() const
{
	return (uint4) m.techniques.size();
}

// Gets a technique by name.
uint4 Material::GetTechniqueIdx(const utf8_ntri &name)
{
	for (techniques_t::const_iterator it = m.techniques.begin(), itEnd = m.techniques.end(); it != itEnd; ++it)
		if (GetDesc(ToImpl(*it->internal.technique)).Name == name)
			return static_cast<uint4>(it - m.techniques.begin());

	return -1;
}

// Gets the number of techniques.
const MaterialTechnique* Material::GetTechnique(uint4 idx)
{
	return reinterpret_cast<const MaterialTechnique*>( &m.techniques[idx].internal );
}

// Gets the number of configurations.
uint4 Material::GetConfigurationCount() const
{
	return (uint4) m.dataSources.size() - m.immutableBaseConfigCount;
}

// Sets the given configuration.
void Material::SetConfiguration(uint4 idx, beg::MaterialConfig *config, uint4 layerMask)
{
	LEAN_ASSERT(config);
	LEAN_ASSERT(idx < m.dataSources.size() - m.immutableBaseConfigCount);
	m.dataSources[idx] = DataSource( ToImpl(config), layerMask );
	m.bBindingChanged = true;
}

// Gets the configurations.
MaterialConfig* Material::GetConfiguration(uint4 idx, uint4 *pLayerMask) const
{
	LEAN_ASSERT(idx < m.dataSources.size() - m.immutableBaseConfigCount);
	if (pLayerMask) *pLayerMask = m.dataSources[idx].layerMask;
	return ToImpl(m.dataSources[idx].config.get());
}

// Sets all layered material configurations (important first).
void Material::SetConfigurations(beg::MaterialConfig *const *config, uint4 configCount)
{
	m.dataSources.erase(m.dataSources.begin(), m.dataSources.end() - m.immutableBaseConfigCount);
	m.dataSources.insert_disjoint(m.dataSources.begin(), config, config + configCount);
	m.bBindingChanged = true;
}

// Gets all layered material configurations (important first).
Material::Configurations Material::GetConfigurations() const
{
	return bec::MakeRangeN(
			Configurations::iterator(&m.dataSources[0].config.get(), sizeof(m.dataSources[0])),
			m.dataSources.size() - m.immutableBaseConfigCount
		);
}
struct Material::ReflectionBinding : public beCore::ResourceAsRefCounted< beCore::PropertyFeedbackProvider<MaterialReflectionBinding> >
{
	static const beCore::ReflectionProperty AdditionalMaterialConfigProperties[1];

	lean::resource_ptr<Material> material;

	struct Binding
	{
		lean::resource_ptr<MaterialConfig> source;
		uint4 sourceIdx;

		Binding(MaterialConfig *binding, uint4 sourceIdx)
			: source(binding),
			sourceIdx(sourceIdx) { }
	};

	struct Property : CollectedPropertyDesc, Binding
	{
		Property() : Binding(nullptr, -1) { }
		Property(const CollectedPropertyDesc &desc, const Binding &binding)
			: CollectedPropertyDesc(desc), Binding(binding) { }
	};

	struct Texture : CollectedTextureDesc, Binding
	{
		Texture() : Binding(nullptr, -1) { }
		Texture(const CollectedTextureDesc &desc, const Binding &binding)
			: CollectedTextureDesc(desc), Binding(binding) { }
	};

	typedef std::vector<Property> properties_t;
	typedef std::vector<Texture> textures_t;
	properties_t propertySources;
	textures_t textureSources;

	uint4 targetSourceIdx;
	lean::resource_ptr<MaterialConfig> pTargetSource;

	ReflectionBinding(Material *material, uint4 targetSourceIdx, MaterialConfig *pTargetSource)
		: material( LEAN_ASSERT_NOT_NULL(material) ),
		targetSourceIdx( targetSourceIdx ),
		pTargetSource( pTargetSource )
	{
		const Material::M &m = material->m;
		MaterialBindingContext bindingCtx(&m.data);
		LoadBindingContext(
				bindingCtx,
				m.dataSources.data() + m.dataSources.size() - m.immutableBaseConfigCount, m.immutableBaseConfigCount,
				// TODO: HACK: UGLY: first-time initialization
				!m.dataSources.empty() && m.dataSources.back().layerMask == -1
			);

		std::vector<Property> properties;
		std::vector<Texture> textures;
		typedef std::vector< std::pair<uint4, uint4> > order_mapping_t;
		order_mapping_t propertyOrder, textureOrder;

		properties.reserve(m.data.properties.size());
		propertyOrder.reserve(m.data.properties.size());
		textures.reserve(m.data.textures.size());
		textureOrder.reserve(m.data.textures.size());

		MaterialConfig *pWaitForSource = pTargetSource;

		for (datasources_t::const_iterator it = m.dataSources.begin(), itEnd = m.dataSources.end(); it < itEnd; ++it)
		{
			const Material::DataSource &dataSource = *it;

			if (pWaitForSource && pWaitForSource != dataSource.config)
				continue;
			else
				pWaitForSource = nullptr;

			for (uint4 srcPropertyIdx = 0, srcPropertyCount = dataSource.config->GetPropertyCount();
				srcPropertyIdx < srcPropertyCount; ++srcPropertyIdx)
			{
				utf8_ntr propertyName = dataSource.config->GetPropertyName(srcPropertyIdx);
				const bec::PropertyDesc &propertyDesc = dataSource.config->GetPropertyDesc(srcPropertyIdx);

				// Get range of properties matching data source
				CollectedPropertyDesc matchDesc(propertyName, propertyDesc);
				MaterialBindingContext::properties_t::iterator
					matchesBegin = std::lower_bound(bindingCtx.properties.begin(), bindingCtx.properties.end(), matchDesc, PropertySorter()),
					matchesEnd = std::upper_bound(bindingCtx.properties.begin(), bindingCtx.properties.end(), matchDesc, PropertySorter());

				uint4 propertyIdx = -1;

				// Bind all matching properties
				for (MaterialBindingContext::properties_t::iterator match = matchesBegin; match < matchesEnd; ++match)
					if (match->layerMask & dataSource.layerMask)
					{
						if (propertyIdx == -1)
							properties.push_back( Property(*match, Binding(&ToImpl(*dataSource.config), srcPropertyIdx)) );
						propertyIdx = min(match->dataIdx, propertyIdx);

						// Exclude from subsequent binding
						match->layerMask = 0;
					}

				if (propertyIdx != -1)
					propertyOrder.push_back( std::make_pair((uint4) propertyOrder.size(), propertyIdx) );
			}

			for (uint4 srcTextureIdx = 0, srcTextureCount = dataSource.config->GetTextureCount();
				srcTextureIdx < srcTextureCount; ++srcTextureIdx)
			{
				utf8_ntr textureName = dataSource.config->GetTextureName(srcTextureIdx);
				bool bTextureRaw = !dataSource.config->IsColorTexture(srcTextureIdx);

				CollectedTextureDesc matchDesc(textureName, bTextureRaw);
				MaterialBindingContext::textures_t::iterator
					matchesBegin = std::lower_bound(bindingCtx.textures.begin(), bindingCtx.textures.end(), matchDesc, TextureSorter()),
					matchesEnd = std::upper_bound(bindingCtx.textures.begin(), bindingCtx.textures.end(), matchDesc, TextureSorter());

				uint4 textureIdx = -1;

				// Bind all matching textures
				for (MaterialBindingContext::textures_t::iterator match = matchesBegin; match < matchesEnd; ++match)
					if (match->layerMask & dataSource.layerMask)
					{
						if (textureIdx == -1)
							textures.push_back( Texture(*match, Binding(&ToImpl(*dataSource.config), srcTextureIdx)) );
						textureIdx = min(match->textureIdx, textureIdx);

						// Exclude from subsequent binding
						match->layerMask = 0;
					}

				if (textureIdx != -1)
					textureOrder.push_back( std::make_pair((uint4) textureOrder.size(), textureIdx) );
			}
		}

		struct SortBySecond
		{
			bool operator ()(order_mapping_t::value_type a, order_mapping_t::value_type b) const
			{
				return a.second < b.second;
			}
		};
		std::sort(propertyOrder.begin(), propertyOrder.end(), SortBySecond());
		std::sort(textureOrder.begin(), textureOrder.end(), SortBySecond());

		propertySources.resize(properties.size());
		textureSources.resize(textures.size());

		for (size_t i = 0, count = propertyOrder.size(); i < count; ++i)
			propertySources[i] = properties[propertyOrder[i].first];
		for (size_t i = 0, count = textureOrder.size(); i < count; ++i)
			textureSources[i] = textures[textureOrder[i].first];
	}

	/// Sets the layer mask.
	void SetLayerMask(uint4 layerMask)
	{
		material->SetConfiguration(targetSourceIdx, LEAN_ASSERT_NOT_NULL(pTargetSource), layerMask);
	}
	/// Gets the layer mask.
	uint4 GetLayerMask() const
	{
		LEAN_ASSERT_NOT_NULL(pTargetSource);
		uint4 layerMask;
		beg::MaterialConfig *config = material->GetConfiguration(targetSourceIdx, &layerMask);
		LEAN_ASSERT(config == pTargetSource);
		return layerMask;
	}

	/// Gets the number of properties.
	uint4 GetPropertyCount() const LEAN_OVERRIDE
	{
		return (uint4) (propertySources.size() + ((pTargetSource) ? lean::arraylen(AdditionalMaterialConfigProperties) : 0));
	}
	/// Gets the ID of the given property.
	uint4 GetPropertyID(const utf8_ntri &name) const LEAN_OVERRIDE
	{
		for (properties_t::const_iterator it = propertySources.begin(), itEnd = propertySources.end(); it != itEnd; ++it)
			if (it->name == name)
				return static_cast<uint4>(it - propertySources.begin());

		return bec::GetPropertyID((uint4)propertySources.size(), ToPropertyRange(AdditionalMaterialConfigProperties), name);
	}
	/// Gets the name of the given property.
	utf8_ntr GetPropertyName(uint4 id) const LEAN_OVERRIDE
	{
		uint4 propertyCount = (uint4) propertySources.size();
		return (id < propertyCount)
			? utf8_ntr(propertySources[id].name)
			: bec::GetPropertyName(propertyCount, ToPropertyRange(AdditionalMaterialConfigProperties), id);
	}
	/// Gets the type of the given property.
	PropertyDesc GetPropertyDesc(uint4 id) const LEAN_OVERRIDE
	{
		uint4 propertyCount = (uint4) propertySources.size();
		return (id < propertyCount)
			? propertySources[id].desc
			: bec::GetPropertyDesc(propertyCount, ToPropertyRange(AdditionalMaterialConfigProperties), id);
	}

	/// Adds and rebinds to the given property.
	void AddAndRebindProperty(uint4 id)
	{
		LEAN_ASSERT(id < propertySources.size());
		Property &binding = propertySources[id];

		binding.sourceIdx = pTargetSource->GetMaterialPropertyID(binding.name, binding.desc);
		if (binding.sourceIdx == -1)
			// TODO: Move desc to effect config
			binding.sourceIdx = pTargetSource->AddProperty(binding.name, binding.desc);
		binding.source = pTargetSource;
	}

	/// Sets the given (raw) values.
	bool SetProperty(uint4 id, const std::type_info &type, const void *values, size_t count) LEAN_OVERRIDE
	{
		uint4 propertyCount = (uint4) propertySources.size();
		if (id < propertyCount)
		{
			Binding &binding = propertySources[id];

			if (pTargetSource && binding.source != pTargetSource)
				AddAndRebindProperty(id);

			// TODO: Make effect configs immutable
			return binding.source->SetProperty(binding.sourceIdx, type, values, count);
		}
		else
			return bec::SetProperty(propertyCount, ToPropertyRange(AdditionalMaterialConfigProperties), *this, id, type, values, count);
	}
	/// Gets the given number of (raw) values.
	bool GetProperty(uint4 id, const std::type_info &type, void *values, size_t count) const LEAN_OVERRIDE
	{
		uint4 propertyCount = (uint4) propertySources.size();
		return (id < propertyCount)
			? propertySources[id].source->GetProperty(propertySources[id].sourceIdx, type, values, count)
			: bec::GetProperty(propertyCount, ToPropertyRange(AdditionalMaterialConfigProperties), *this, id, type, values, count);
	}

	/// Visits a property for modification.
	bool WriteProperty(uint4 id, beCore::PropertyVisitor &visitor, uint4 flags) LEAN_OVERRIDE
	{
		uint4 propertyCount = (uint4) propertySources.size();
		if (id < propertyCount)
		{
			Binding &binding = propertySources[id];

			if (pTargetSource && binding.source != pTargetSource)
				AddAndRebindProperty(id);

			// TODO: Make effect configs immutable
			return binding.source->WriteProperty(binding.sourceIdx, visitor, flags);
		}
		else
			return bec::WriteProperty(propertyCount, ToPropertyRange(AdditionalMaterialConfigProperties), *this, id, visitor, flags);
	}
	/// Visits a property for reading.
	bool ReadProperty(uint4 id, beCore::PropertyVisitor &visitor,  uint4 flags) const LEAN_OVERRIDE
	{
		uint4 propertyCount = (uint4) propertySources.size();
		return (id < propertyCount)
			? propertySources[id].source->ReadProperty(propertySources[id].sourceIdx, visitor, flags)
			: bec::ReadProperty(propertyCount, ToPropertyRange(AdditionalMaterialConfigProperties), *this, id, visitor, flags);
	}


	// Gets the number of textures.
	uint4 GetTextureCount() const LEAN_OVERRIDE
	{
		return (uint4) textureSources.size();
	}

	// Gets the ID of the given texture.
	uint4 GetTextureID(const utf8_ntri &name) const LEAN_OVERRIDE
	{
		for (textures_t::const_iterator it = textureSources.begin(), itEnd = textureSources.end(); it != itEnd; ++it)
			if (it->name == name)
				return static_cast<uint4>(it - textureSources.begin());

		return static_cast<uint4>(-1);
	}
	
	/// Adds and rebinds to the given texture.
	void AddAndRebindTexture(uint4 idx)
	{
		LEAN_ASSERT(idx < textureSources.size());

		Binding &binding = textureSources[idx];

		binding.sourceIdx = pTargetSource->GetTextureID(textureSources[idx].name);
		if (binding.sourceIdx == -1)
			binding.sourceIdx = pTargetSource->AddTexture(textureSources[idx].name, !textureSources[idx].bRaw);
		binding.source = pTargetSource;
	}

	// Gets the name of the given texture.
	utf8_ntr GetTextureName(uint4 id) const LEAN_OVERRIDE
	{
		return utf8_ntr(textureSources[id].name);
	}

	// Gets whether the texture is a color texture.
	bool IsColorTexture(uint4 id) const LEAN_OVERRIDE
	{
		return !textureSources[id].bRaw;
	}

	// Sets the given texture.
	void SetTexture(uint4 id, const beGraphics::TextureView *pView) LEAN_OVERRIDE
	{
		LEAN_ASSERT(id < textureSources.size());

		Binding &binding = textureSources[id];

		if (pTargetSource && binding.source != pTargetSource)
			AddAndRebindTexture(id);

		// TODO: Make effect configs immutable
		binding.source->SetTexture(binding.sourceIdx, pView );
	}

	// Gets the given texture.
	const TextureView* GetTexture(uint4 id) const LEAN_OVERRIDE
	{
		LEAN_ASSERT(id < textureSources.size());
		return textureSources[id].source->GetTexture(textureSources[id].sourceIdx);
	}


	/// Gets the number of child components.
	uint4 GetComponentCount() const LEAN_OVERRIDE
	{
		return (uint4) textureSources.size();
	}
	/// Gets the name of the n-th child component.
	beCore::Exchange::utf8_string GetComponentName(uint4 idx) const LEAN_OVERRIDE
	{
		LEAN_ASSERT(idx < textureSources.size());
		return lean::from_range<beCore::Exchange::utf8_string>( textureSources[idx].name );
	}
	/// Gets the n-th reflected child component, nullptr if not reflected.
	lean::com_ptr<const beCore::ReflectedComponent, lean::critical_ref> GetReflectedComponent(uint4 idx) const LEAN_OVERRIDE
	{
		return nullptr;
	}

	/// Gets the type of the n-th child component.
	const beCore::ComponentType* GetComponentType(uint4 idx) const LEAN_OVERRIDE
	{
		return beg::TextureView::GetComponentType();
	}
	/// Gets the n-th component.
	lean::cloneable_obj<lean::any, true> GetComponent(uint4 idx) const LEAN_OVERRIDE
	{
		return bec::any_resource_t<beg::TextureView>::t( const_cast<beGraphics::DX11::TextureView*>( GetTexture(idx) ) );
	}

	/// Returns true, if the n-th component can be replaced.
	bool IsComponentReplaceable(uint4 idx) const LEAN_OVERRIDE
	{
		// TODO: Make effect configs immutable
		return true;
	}
	/// Sets the n-th component.
	void SetComponent(uint4 idx, const lean::any &pComponent) LEAN_OVERRIDE
	{
		SetTexture( idx, any_cast<beg::TextureView*>(pComponent) );
	}

	// Gets the type of the n-th child component.
	const beCore::ComponentType* GetType() const
	{
		static const beCore::ComponentType type = { "Material.ReflectionBinding" };
		return &type;
	}
};

const beCore::ReflectionProperty Material::ReflectionBinding::AdditionalMaterialConfigProperties[] =
{
	beCore::MakeReflectionProperty<uint4>("layer mask", beCore::Widget::Raw)
		.set_setter( BE_CORE_PROPERTY_SETTER(&Material::ReflectionBinding::SetLayerMask) )
		.set_getter( BE_CORE_PROPERTY_GETTER(&Material::ReflectionBinding::GetLayerMask) ) 
};

// Gets a merged reflection binding.
lean::com_ptr<MaterialReflectionBinding, lean::critical_ref> Material::GetFixedBinding()
{
	return lean::bind_com( new ReflectionBinding(this, -1, nullptr) );
}

// Gets a reflection binding for the given configuration.
lean::com_ptr<MaterialReflectionBinding, lean::critical_ref> Material::GetConfigBinding(uint4 configIdx)
{
	LEAN_ASSERT(configIdx < m.dataSources.size() - m.immutableBaseConfigCount);
	MaterialConfig *config = ToImpl(m.dataSources[configIdx].config.get());
	return lean::bind_com( new ReflectionBinding(this, configIdx, config) );
}

// Gets the number of child components.
uint4 Material::GetComponentCount() const
{
	return (uint4) m.dataSources.size() - m.immutableBaseConfigCount + 1;
}

// Gets the name of the n-th child component.
beCore::Exchange::utf8_string Material::GetComponentName(uint4 idx) const
{
	beCore::Exchange::utf8_string name;

	LEAN_ASSERT(idx <= m.dataSources.size() - m.immutableBaseConfigCount);

	if (idx < m.dataSources.size() - m.immutableBaseConfigCount)
	{
		utf8_string num = lean::int_to_string(idx);
		name.reserve(lean::ntarraylen("Config ") + num.size());
		name.append("Config ");
		name.append(num.c_str(), num.c_str() + num.size());
	}
	else
		name = "Layered";

	return name;
}

// Returns true, if the n-th component is issential.
bool Material::IsComponentEssential(uint4 idx) const
{
	return idx == 0;
}

// Gets the n-th reflected child component, nullptr if not reflected.
lean::com_ptr<const beCore::ReflectedComponent, lean::critical_ref> Material::GetReflectedComponent(uint4 idx) const
{
	LEAN_ASSERT(idx <= m.dataSources.size() - m.immutableBaseConfigCount);
	return (idx < m.dataSources.size() - m.immutableBaseConfigCount)
		? const_cast<Material*>(this)->GetConfigBinding(idx)
		: const_cast<Material*>(this)->GetFixedBinding();
}

// Gets the type of the n-th child component.
const beCore::ComponentType* Material::GetComponentType(uint4 idx) const
{
	return beg::MaterialConfig::GetComponentType();
}

// Gets the n-th component.
lean::cloneable_obj<lean::any, true> Material::GetComponent(uint4 idx) const
{
	LEAN_ASSERT(idx <= m.dataSources.size() - m.immutableBaseConfigCount);
	return bec::any_resource_t<beg::MaterialConfig>::t(
			(idx < m.dataSources.size() - m.immutableBaseConfigCount) ? m.dataSources[idx].config : nullptr
		);
}

// Returns true, if the n-th component can be replaced.
bool Material::IsComponentReplaceable(uint4 idx) const
{
	return (idx < m.dataSources.size() - m.immutableBaseConfigCount);
}

// Sets the n-th component.
void Material::SetComponent(uint4 idx, const lean::any &pComponent)
{
	LEAN_ASSERT(idx <= m.dataSources.size() - m.immutableBaseConfigCount);
	// NOTE: Error-tolerant - actually, idx > 0 should be asserted
	if (idx < m.dataSources.size() - m.immutableBaseConfigCount)
		SetConfiguration(
				idx,
				// NOTE: Material configurations may not be unset
				LEAN_THROW_NULL(any_cast<beGraphics::MaterialConfig*>(pComponent))
			);
}

} // namespace

// Creates a new material.
lean::resource_ptr<Material, lean::critical_ref> CreateMaterial(const Effect *const* effects, uint4 effectCount, EffectCache &effectCache)
{
	return new_resource DX11::Material(effects, effectCount, effectCache);
}

// Creates a new material.
lean::resource_ptr<Material, lean::critical_ref> CreateMaterial(const Material &right)
{
	return new_resource DX11::Material(ToImpl(right));
}

} // namespace

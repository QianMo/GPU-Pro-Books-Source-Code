/*****************************************************/
/* breeze Engine Graphics Module  (c) Tobias Zirr 2011 */
/*****************************************************/

#include "beGraphicsInternal/stdafx.h"
#include "beGraphics/DX11/beEffectConfig.h"

#include "beGraphics/DX11/beDevice.h"
#include "beGraphics/DX11/beTextureCache.h"

#include "beGraphics/DX/beError.h"

#include <beCore/beBuiltinTypes.h>

#include <lean/properties/property_types.h>
#include <lean/functional/algorithm.h>

#include <lean/logging/log.h>

namespace beGraphics
{

namespace DX11
{

namespace
{

// Don't do this lazily, serialization might require type info before construction of first setup
const bec::ValueTypeDesc &BoolTypeDesc = beCore::GetBuiltinType<BOOL>();
const bec::ValueTypeDesc &IntTypeDesc = beCore::GetBuiltinType<INT>();
const bec::ValueTypeDesc &UintTypeDesc = beCore::GetBuiltinType<UINT>();
const bec::ValueTypeDesc &UlongTypeDesc = beCore::GetBuiltinType<UINT8>();
const bec::ValueTypeDesc &FloatTypeDesc = beCore::GetBuiltinType<FLOAT>();
const bec::ValueTypeDesc &DoubleTypeDesc = beCore::GetBuiltinType<DOUBLE>();

/// Gets a property description from the given effect type description. Returns count of zero if unknown.
PropertyDesc GetPropertyDesc(const D3DX11_EFFECT_TYPE_DESC &typeDesc, int2 widget)
{
	size_t componentCount = max<size_t>(typeDesc.Rows, 1)
		* max<size_t>(typeDesc.Columns, 1)
		* max<size_t>(typeDesc.Elements, 1);

	switch (typeDesc.Type)
	{
	case D3D_SVT_BOOL:
		return PropertyDesc(BoolTypeDesc, componentCount, widget);

	case D3D_SVT_INT:
		return PropertyDesc(IntTypeDesc, componentCount, widget);
	case D3D_SVT_UINT:
		return PropertyDesc(UintTypeDesc, componentCount, widget);
	case D3D_SVT_UINT8:
		return PropertyDesc(UlongTypeDesc, componentCount, widget);

	case D3D_SVT_FLOAT:
		return PropertyDesc(FloatTypeDesc, componentCount, widget);
	case D3D_SVT_DOUBLE:
		return PropertyDesc(DoubleTypeDesc, componentCount, widget);
	}

	return PropertyDesc();
}

/// Gets the widget by name.
int2 GetWidgetByName(const utf8_ntri &name)
{
	if (_stricmp(name.c_str(), "color") == 0)
		return beCore::Widget::Color;
	else if (_stricmp(name.c_str(), "slider") == 0)
		return beCore::Widget::Slider;
	else if (_stricmp(name.c_str(), "raw") == 0)
		return beCore::Widget::Raw;
	else if (_stricmp(name.c_str(), "angle") == 0)
		return beCore::Widget::Angle;
	else if (_stricmp(name.c_str(), "none") == 0)
		return beCore::Widget::None;
	else if (_stricmp(name.c_str(), "orientation") == 0)
		return beCore::Widget::Orientation;
	else
		// Default to raw value
		return beCore::Widget::Raw;
}

// Gets all properties.
void GetProperties(API::Effect *effect, MaterialConfig::backingstore_t &backingStore, MaterialConfig::properties_t &properties,
				   EffectConfig::constants_t &variables, EffectConfig::cbuffers_t &cbuffers)
{
	D3DX11_EFFECT_DESC effectDesc = GetDesc(effect);
	
	uint4 backingStoreOffset = 0;

	// Scan all constant buffers for properties
	for (uint4 cbIdx = 0; cbIdx < effectDesc.ConstantBuffers; ++cbIdx)
	{
		API::EffectConstantBuffer *cb = effect->GetConstantBufferByIndex(cbIdx);

		if (GetDesc(cb).Flags & D3DX11_EFFECT_VARIABLE_UNMANAGED)
			continue;

		D3DX11_EFFECT_TYPE_DESC cbDesc = GetDesc(cb->GetType());
		uint4 cbPropertyBegin = static_cast<uint4>(properties.size());

		for (uint4 varIdx = 0; varIdx < cbDesc.Members; ++varIdx)
		{
			API::EffectVariable *variable = cb->GetMemberByIndex(varIdx);
			
			const char *propertyName;

			// Get property name
			if (FAILED(variable->GetAnnotationByName("UIName")->AsString()->GetString(&propertyName)) &&
				FAILED(variable->GetAnnotationByName("Name")->AsString()->GetString(&propertyName)))
				continue;

			D3DX11_EFFECT_TYPE_DESC variableType = GetDesc(variable->GetType());

			// Build property description
			const char *widgetName = nullptr;
			SUCCEEDED(variable->GetAnnotationByName("UIWidget")->AsString()->GetString(&widgetName))
				|| SUCCEEDED(variable->GetAnnotationByName("Widget")->AsString()->GetString(&widgetName));

			int2 widget = (widgetName) ? GetWidgetByName(widgetName) : beCore::Widget::Raw;
			PropertyDesc propertyDesc = GetPropertyDesc(variableType, widget);

			// Check if description valid
			if (!propertyDesc.Count)
				continue;

			int layerOffset = 0;
			variable->GetAnnotationByName("LayerOffset")->AsScalar()->GetInt(&layerOffset);

			D3DX11_EFFECT_VARIABLE_DESC variableDesc = GetDesc(variable);
			
			// Add new property
			uint4 propertyIdx = static_cast<uint4>( properties.size() );
			variables.push_back(EffectConstantInfo(variable, variableDesc.BufferOffset, layerOffset));
			properties.push_back(
					MaterialConfig::Property(propertyName, propertyDesc),
					MaterialConfig::PropertyData(
						backingStoreOffset,
						static_cast<uint2>(propertyDesc.Count),
						static_cast<uint1>(variableType.PackedSize / propertyDesc.Count),
						propertyDesc.TypeDesc->Info.type)
				);
			
			backingStoreOffset += variableType.UnpackedSize;
			backingStore.resize(backingStoreOffset);

			// Read initial data
			MaterialConfig::PropertyData &data = properties(MaterialConfig::propertyData)[propertyIdx];
			variable->GetRawValue(&backingStore[data.offset], 0, data.count * data.elementSize);
			data.bSet = true;
		}

		uint4 cbPropertyEnd = static_cast<uint4>(properties.size());
		
		if (cbPropertyBegin < cbPropertyEnd)
			// Add constant buffer
			cbuffers.push_back(EffectConstantBufferInfo(cb, bec::MakeRange(cbPropertyBegin, cbPropertyEnd), cbDesc.UnpackedSize));
	}

	backingStore.shrink(backingStore.size());
}

// Gets all textures.
void GetTextures(API::Effect *effect, MaterialConfig::textures_t &textures, EffectConfig::resources_t &variables, TextureCache *pTextureCache)
{
	D3DX11_EFFECT_DESC effectDesc = GetDesc(effect);
	
	// Scan all variables for textures
	for (uint4 varIdx = 0; varIdx < effectDesc.GlobalVariables; ++varIdx)
	{
		API::EffectShaderResource *variable = effect->GetVariableByIndex(varIdx)->AsShaderResource();

		if (!variable->IsValid())
			continue;

		D3DX11_EFFECT_VARIABLE_DESC variableDesc = GetDesc(variable);

		// Not bound by semantic or unmanaged
		if (variableDesc.Semantic || (variableDesc.Flags & D3DX11_EFFECT_VARIABLE_UNMANAGED))
			continue;

		BOOL isColor = TRUE;
		variable->GetAnnotationByName("Color")->AsScalar()->GetBool(&isColor);
		BOOL isRaw = !isColor;
		variable->GetAnnotationByName("Raw")->AsScalar()->GetBool(&isRaw);

		const char *textureName = variableDesc.Name;
		SUCCEEDED(variable->GetAnnotationByName("UIName")->AsString()->GetString(&textureName))
			|| SUCCEEDED(variable->GetAnnotationByName("Name")->AsString()->GetString(&textureName));

		int layerOffset = 0;
		variable->GetAnnotationByName("LayerOffset")->AsScalar()->GetInt(&layerOffset);

		// Add new texture
		uint4 textureIdx = static_cast<uint4>( textures.size() );
		variables.push_back(EffectResourceInfo(variable, !!isRaw, layerOffset));
		textures.push_back( MaterialConfig::Texture(textureName), MaterialConfig::TextureData(!isRaw) );
		
		const char *pTextureFile = nullptr;
		SUCCEEDED(variable->GetAnnotationByName("UIFile")->AsString()->GetString(&pTextureFile))
			|| SUCCEEDED(variable->GetAnnotationByName("File")->AsString()->GetString(&pTextureFile));

		MaterialConfig::TextureData &data = textures(MaterialConfig::textureData)[textureIdx];
		data.pTexture = (pTextureCache && pTextureFile)
			? ToImpl(pTextureCache->GetViewByFile(pTextureFile, !isRaw))
			: nullptr;
		data.bSet = true;
	}
}

} // namespace

// Constructor.
EffectConfig::EffectConfig(Effect *effect, TextureCache *pTextureCache)
{
	GetProperties(*effect, m.backingStore, m.properties, m_constants, m_cbuffers);
	GetTextures(*effect, m.textures, m_resources, pTextureCache);
}

// Destructor.
EffectConfig::~EffectConfig()
{
}

} // namespace

} // namespace

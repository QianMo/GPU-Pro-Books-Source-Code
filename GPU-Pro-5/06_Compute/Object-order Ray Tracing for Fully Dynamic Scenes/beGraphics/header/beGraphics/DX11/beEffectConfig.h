/******************************************************/
/* breeze Engine Graphics Module (c) Tobias Zirr 2011 */
/******************************************************/

#pragma once
#ifndef BE_GRAPHICS_EFFECT_CONFIG_DX11
#define BE_GRAPHICS_EFFECT_CONFIG_DX11

#include "beGraphics.h"
#include "beMaterialConfig.h"
#include "beEffect.h"
#include "beD3DXEffects11.h"
#include <beCore/beMany.h>

namespace beGraphics
{

namespace DX11
{

class TextureCache;

struct EffectConstantBufferInfo
{
	API::EffectConstantBuffer *Variable;
	beCore::Range<uint4> Constants;
	uint4 Size;

	EffectConstantBufferInfo(API::EffectConstantBuffer *variable, beCore::Range<uint4> constants, uint4 size)
		: Variable(variable),
		Constants(constants),
		Size(size) { }
};

struct EffectConstantInfo
{
	API::EffectVariable *Variable;
	uint4 BufferOffset;
	uint4 LayerOffset;

	EffectConstantInfo(API::EffectVariable *variable, uint4 bufferOffset, uint4 layerOffset = 0)
		: Variable(variable),
		BufferOffset(bufferOffset),
		LayerOffset(layerOffset) { }
};

struct EffectResourceInfo
{
	API::EffectShaderResource *Variable;
	bool IsRaw;
	uint4 LayerOffset;

	EffectResourceInfo(API::EffectShaderResource *variable, bool isRaw, uint4 layerOffset = 0)
		: Variable(variable),
		IsRaw(isRaw),
		LayerOffset(layerOffset) { }
};

/// Setup implementation.
class EffectConfig : public MaterialConfig
{
public:
	typedef lean::simple_vector<EffectConstantBufferInfo, lean::containers::vector_policies::inipod> cbuffers_t;
	typedef lean::simple_vector<EffectConstantInfo, lean::containers::vector_policies::inipod> constants_t;
	typedef lean::simple_vector<EffectResourceInfo, lean::containers::vector_policies::inipod> resources_t;

protected:
	cbuffers_t m_cbuffers;
	constants_t m_constants;
	resources_t m_resources;

public:
	/// Constructor.
	BE_GRAPHICS_DX11_API EffectConfig(Effect *effect, TextureCache *pTextureCache);
	/// Destructor.
	BE_GRAPHICS_DX11_API ~EffectConfig();

	typedef beCore::Range<const EffectConstantBufferInfo*> ConstantBufferRange;
	typedef beCore::Range<const EffectConstantInfo*> ConstantRange;
	typedef beCore::Range<const EffectResourceInfo*> ResourceRange;

	/// Gets the constant buffers.
	ConstantBufferRange GetConstantBufferInfo() const { return beCore::MakeRangeN(&m_cbuffers[0], m_cbuffers.size()); }
	/// Gets the constants.
	ConstantRange GetConstantInfo() const { return beCore::MakeRangeN(&m_constants[0], m_constants.size()); }
	/// Gets the resources.
	ResourceRange GetResourceInfo() const { return beCore::MakeRangeN(&m_resources[0], m_resources.size()); }
};

} // namespace

} // namespace

#endif
#pragma once

#include "Tracing.h"

#include <beScene/beResourceManager.h>
#include <beScene/beEffectDrivenRenderer.h>
#include <beScene/beRenderContext.h>

#include <beGraphics/Any/beAPI.h>
#include <beGraphics/Any/beEffectsAPI.h>

#include <lean/smart/resource_ptr.h>
#include <lean/smart/com_ptr.h>
#include <vector>

#include <beMath/beVectorDef.h>

namespace app
{

class IncrementalGPUTimer;

namespace tracing
{

/// Voxel representation constants.
struct VoxelRepLayout
{
	bem::vector<uint4, 3> Resolution;
	uint4 RastResolution;
	bem::fvec3 VoxelWidth;
	float RastVoxelWidth;

	bem::fvec3 Min;
	float _pad3;
	bem::fvec3 Max;
	float _pad4;

	bem::fvec3 Center;
	float _pad5;
	bem::fvec3 Ext;
	float _pad6;
	bem::fvec3 UnitScale;
	float _pad7;

	bem::fvec3 VoxelSize;
	float _pad8;
	bem::fvec3 VoxelScale;
	float _pad9;
};

/// Voxel representation.
class VoxelRep
{
public:
	struct Level
	{
		lean::com_ptr<beg::api::ShaderResourceView> SRV;
		lean::com_ptr<beg::api::UnorderedAccessView> UAV;
	};
	typedef std::vector<Level> level_vector;

private:
	bem::fvec3 m_min;
	bem::fvec3 m_max;

	bem::vector<uint4, 3> m_resolution;
	uint4 m_rasterResolution;

	lean::com_ptr<beg::api::Buffer> m_constBuffer;

	lean::com_ptr<beg::api::Texture3D> m_voxels;
	lean::com_ptr<beg::api::ShaderResourceView> m_voxelSRV;
	lean::com_ptr<beg::api::UnorderedAccessView> m_voxelUAV;

public:
	/// Constructor.
	VoxelRep(bem::vector<uint4, 3> resolution, besc::EffectDrivenRenderer *renderer, besc::ResourceManager *resourceManager);
	/// Destructor.
	~VoxelRep();

	/// Updates the voxel representation constant buffer.
	beg::api::Buffer *const& UpdateConstants(beg::api::DeviceContext *context);
	/// Gets the voxel representation constant buffer.
	beg::api::Buffer *const& Constants() { return m_constBuffer.get(); }

	/// Sets the voxel field.
	void SetVoxelTexture(beg::api::Texture3D *tex, beg::api::ShaderResourceView *srv, beg::api::UnorderedAccessView *uav)
	{
		m_voxels = tex;
		m_voxelSRV = srv;
		m_voxelUAV = uav;
	}
	/// Gets the voxel field.
	beg::api::Texture3D *const& VoxelTexture() const { return m_voxels.get(); }
	/// Gets the voxel field.
	beg::api::ShaderResourceView *const& Voxels() const { return m_voxelSRV.get(); }
	/// Gets the voxel field.
	beg::api::UnorderedAccessView *const& VoxelUAV() { return m_voxelUAV.get(); }

	/// Gets the resolution.
	bem::vector<uint4, 3> Resolution() const { return m_resolution; }
	/// Gets the voxelization resolution.
	uint4 RasterResolution() const { return m_rasterResolution; }

	/// Sets the bounds.
	void SetBounds(const bem::fvec3 &min, const bem::fvec3 &max) { m_min = min; m_max = max; }
	/// Gets the min point.
	const bem::fvec3& Min() const { return m_min; }
	/// Gets the max point.
	const bem::fvec3& Max() const { return m_max; }
};


/// Voxel representation generator.
class VoxelRepGen
{
public:
	struct MipLevel
	{
		lean::com_ptr<ID3D11ShaderResourceView> SRV;
		lean::com_ptr<ID3D11UnorderedAccessView> UAV;
	};
	typedef std::vector<MipLevel> mip_vector;

private:
	lean::resource_ptr<beg::Material> m_material;

	beg::api::Effect *m_effect;
	beg::api::EffectTechnique *m_voxelMip;

	beg::api::EffectConstantBuffer *m_constVar;

	beg::api::EffectShaderResource *m_voxelSRVVar;
	beg::api::EffectUnorderedAccessView *m_voxelUAVVar;

	lean::com_ptr<beg::api::Texture3D> m_voxels;
	lean::com_ptr<beg::api::ShaderResourceView> m_voxelSRV;
	lean::com_ptr<beg::api::UnorderedAccessView> m_voxelUAV;
	mip_vector m_voxelLevels;

	/// Binds the given material.
	void BindMaterial(beg::Material *material);

public:
	/// Constructor.
	VoxelRepGen(const lean::utf8_ntri &file, bem::vector<uint4, 3> resolution, 
		besc::EffectDrivenRenderer *renderer, besc::ResourceManager *resourceManager);
	/// Destructor.
	~VoxelRepGen();
	
	/// Processes / commits changes.
	void Commit();

	/// Computes slice map mip levels.
	void MipVoxels(VoxelRep &voxelRep, besc::RenderContext &context);

	/// Binds the given voxel representation to this generator.
	void Bind(VoxelRep &voxelRep)
	{
		voxelRep.SetVoxelTexture(m_voxels, m_voxelSRV, m_voxelUAV);
	}
};

} // namespace

} // namespace
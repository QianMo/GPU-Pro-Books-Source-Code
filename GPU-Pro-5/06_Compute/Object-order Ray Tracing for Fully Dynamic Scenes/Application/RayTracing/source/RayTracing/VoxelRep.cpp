/******************************************************/
/* Object-order Ray Tracing Demo (c) Tobias Zirr 2013 */
/******************************************************/

#include "stdafx.h"

#include "RayTracing/VoxelRep.h"

#include <beGraphics/beMaterialCache.h>
#include <beGraphics/beMaterial.h>

#include <beGraphics/Any/beBuffer.h>
#include <beGraphics/Any/beTexture.h>
#include <beGraphics/Any/beDevice.h>
#include <beGraphics/Any/beEffect.h>
#include <beGraphics/Any/beDeviceContext.h>

#include <beGraphics/DX/beError.h>
#include <lean/logging/errors.h>

namespace app
{

namespace tracing
{

inline uint4 ceil_div(uint4 x, uint4 d)
{
	return (x + d - 1) / d;
}

// Constructor.
VoxelRep::VoxelRep(bem::vector<uint4, 3> resolution, besc::EffectDrivenRenderer *renderer, besc::ResourceManager *resourceManager)
	: m_resolution( resolution ),
	m_rasterResolution( max(max(resolution[0], resolution[1]), resolution[2]) )
{
	beg::api::Device *device = ToImpl(*renderer->Device());

	m_constBuffer = beg::Any::CreateConstantBuffer(device, sizeof(VoxelRepLayout));
}

// Destructor.
VoxelRep::~VoxelRep()
{
}

// Updates the voxel representation constant buffer.
beg::api::Buffer *const& VoxelRep::UpdateConstants(beg::api::DeviceContext *context)
{
	VoxelRepLayout constants;
	constants.Resolution = m_resolution;
	constants.RastResolution = m_rasterResolution;
	constants.VoxelWidth = 1.0f / bem::fvec3(constants.Resolution);
	constants.RastVoxelWidth = 1.0f / m_rasterResolution;
	constants.Min = m_min;
	constants.Max = m_max;
	constants.Center = 0.5f * m_min + 0.5f * m_max;
	constants.Ext = m_max - m_min;
	constants.UnitScale = 1.0f / constants.Ext;
	constants.VoxelSize = constants.Ext * constants.VoxelWidth;
	constants.VoxelScale = bem::fvec3(constants.Resolution) / constants.Ext;

	context->UpdateSubresource(m_constBuffer, 0, nullptr, &constants, 0, 0);
	return m_constBuffer.get();
}

// Binds the given material.
void VoxelRepGen::BindMaterial(beg::Material *material)
{
	m_material = material;
	m_effect = ToImpl(m_material->GetEffects()[0])->Get();

	m_voxelMip = beg::Any::ValidateEffectVariable(m_effect->GetTechniqueByName("VoxelMip"), LSS);

	m_constVar = beg::Any::ValidateEffectVariable(m_effect->GetConstantBufferByName("VoxelRepConstants"), LSS);

	m_voxelSRVVar = beg::Any::ValidateEffectVariable(m_effect->GetVariableBySemantic("VoxelField")->AsShaderResource(), LSS);
	m_voxelUAVVar = beg::Any::ValidateEffectVariable(m_effect->GetVariableBySemantic("VoxelFieldUAV")->AsUnorderedAccessView(), LSS);
}

// Constructor.
VoxelRepGen::VoxelRepGen(const lean::utf8_ntri &file, bem::vector<uint4, 3> resolution,
	besc::EffectDrivenRenderer *renderer, besc::ResourceManager *resourceManager)
{
	beg::api::Device *device = ToImpl(*renderer->Device());
	
	// Effect
	BindMaterial( resourceManager->MaterialCache()->NewByFile(file, "VoxelRep") );

	// Voxels
	{
		m_voxels = beg::Any::CreateTexture3D(device,
				D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE,
				DXGI_FORMAT_R32_UINT,
				resolution[0], resolution[1], resolution[2], 0
			);
		m_voxelSRV = beg::Any::CreateSRV(m_voxels);
		m_voxelUAV = beg::Any::CreateUAV(m_voxels);

		const uint4 levelCount = beg::Any::GetDesc(m_voxels).MipLevels;
		m_voxelLevels.resize(levelCount);

		for (uint4 levelIdx = 0; levelIdx < levelCount; ++levelIdx)
		{
			MipLevel &level = m_voxelLevels[levelIdx];

			D3D11_SHADER_RESOURCE_VIEW_DESC SRVDesc;
			SRVDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE3D;
			SRVDesc.Format = DXGI_FORMAT_R32_UINT;
			SRVDesc.Texture3D.MostDetailedMip = levelIdx;
			SRVDesc.Texture3D.MipLevels = 1;

			BE_THROW_DX_ERROR_MSG(
				device->CreateShaderResourceView(m_voxels, &SRVDesc, level.SRV.rebind()),
				"ID3D11Device::CreateShaderResourceView()" );

			D3D11_UNORDERED_ACCESS_VIEW_DESC UAVDesc;
			UAVDesc.ViewDimension = D3D11_UAV_DIMENSION_TEXTURE3D;
			UAVDesc.Format = DXGI_FORMAT_R32_UINT;
			UAVDesc.Texture3D.MipSlice = levelIdx;
			UAVDesc.Texture3D.FirstWSlice = 0;
			UAVDesc.Texture3D.WSize = -1;

			BE_THROW_DX_ERROR_MSG(
				device->CreateUnorderedAccessView(m_voxels, &UAVDesc, level.UAV.rebind()),
				"ID3D11Device::CreateUnorderedAccessView()" );
		}
	}
}

// Destructor.
VoxelRepGen::~VoxelRepGen()
{
}

// Processes / commits changes.
void VoxelRepGen::Commit()
{
	// Hot swap
	if (beg::Material *successor = m_material->GetSuccessor())
		BindMaterial(successor);
}

// Computes slice map mip levels.
void VoxelRepGen::MipVoxels(VoxelRep &voxelRep, besc::RenderContext &context)
{
	beg::api::DeviceContext *deviceContext = ToImpl(context.Context());

	m_constVar->SetConstantBuffer(voxelRep.Constants());

	bem::vector<uint4, 3> mipRes = voxelRep.Resolution() / 2;

	for (uint4 levelIdx = 1; levelIdx < 6; ++levelIdx, mipRes /= 2)
	{
		m_voxelSRVVar->SetResource(m_voxelLevels[levelIdx - 1].SRV);
		m_voxelUAVVar->SetUnorderedAccessView(m_voxelLevels[levelIdx].UAV);

		m_voxelMip->GetPassByIndex(0)->Apply(0, deviceContext);

		static const uint4 GroupSize = 8;
		deviceContext->Dispatch(ceil_div(mipRes[0], GroupSize), ceil_div(mipRes[1], GroupSize), ceil_div(mipRes[2], GroupSize));

		beg::Any::UnbindAllComputeTargets(deviceContext);
	}
}

} // namespace

} // namespace

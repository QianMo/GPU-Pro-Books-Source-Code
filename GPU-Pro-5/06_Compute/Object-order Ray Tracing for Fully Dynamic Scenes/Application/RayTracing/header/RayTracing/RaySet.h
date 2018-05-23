#pragma once

#include "Tracing.h"

#include <beScene/beResourceManager.h>
#include <beScene/beEffectDrivenRenderer.h>
#include <beScene/beRenderContext.h>

#include <beGraphics/Any/beAPI.h>
#include <beGraphics/Any/beEffectsAPI.h>

#include "CUDAUtils.h"
#include "B40Calls.h"

#include <lean/smart/resource_ptr.h>
#include <lean/smart/com_ptr.h>
#include <vector>

#include <beMath/beMatrixDef.h>

namespace app
{

class IncrementalGPUTimer;

namespace tracing
{

class VoxelRep;

/// Ray set constants.
struct RaySetLayout
{
	uint4 RayCount;
	uint4 RayLinkCount;
	uint4 ActiveRayCount;
	uint4 MaxRayLinkCount;
};

/// Ray set.
class RaySet
{
private:
	uint4 m_maxRayCount;

	lean::com_ptr<beg::api::Buffer> m_constBuffer;

	lean::com_ptr<beg::api::Buffer> m_rayDescBuffer;
	lean::com_ptr<beg::api::ShaderResourceView> m_rayDescSRV;
	lean::com_ptr<beg::api::UnorderedAccessView> m_rayDescUAV;

	lean::com_ptr<beg::api::Buffer> m_rayGeometryBuffer;
	lean::com_ptr<beg::api::ShaderResourceView> m_rayGeometrySRV;
	lean::com_ptr<beg::api::UnorderedAccessView> m_rayGeometryUAV;

	lean::com_ptr<beg::api::Buffer> m_rayLightBuffer;
	lean::com_ptr<beg::api::ShaderResourceView> m_rayLightSRV;
	lean::com_ptr<beg::api::UnorderedAccessView> m_rayLightUAV;

	lean::com_ptr<beg::api::Buffer> m_rayDebugBuffer;
	lean::com_ptr<beg::api::ShaderResourceView> m_rayDebugSRV;
	lean::com_ptr<beg::api::UnorderedAccessView> m_rayDebugUAV;

	lean::com_ptr<beg::api::Buffer> m_rayLinks;
	lean::com_ptr<beg::api::ShaderResourceView> m_rayLinkSRV;
	lean::com_ptr<beg::api::UnorderedAccessView> m_rayLinkUAV;

	lean::com_ptr<beg::api::Buffer> m_rayList;
	lean::com_ptr<beg::api::ShaderResourceView> m_rayListSRV;
	lean::com_ptr<beg::api::UnorderedAccessView> m_rayListUAV;

	lean::com_ptr<beg::api::Texture3D> m_rayGridBegin, m_rayGridEnd;
	lean::com_ptr<beg::api::ShaderResourceView> m_rayGridBeginSRV, m_rayGridEndSRV;
	lean::com_ptr<beg::api::UnorderedAccessView> m_rayGridBeginUAV, m_rayGridEndUAV;

public:
	/// Constructor.
	RaySet(uint4 maxRayCount, besc::EffectDrivenRenderer *renderer, besc::ResourceManager *resourceManager);
	/// Destructor.
	~RaySet();

	/// Gets the set constant buffer.
	beg::api::Buffer *const& Constants() { return m_constBuffer.get(); }

	/// Gets the ray description buffer.
	beg::api::Buffer *const& RayDescBuffer() const { return m_rayDescBuffer.get(); }
	/// Gets the ray description buffer.
	beg::api::ShaderResourceView *const& RayDescs() const { return m_rayDescSRV.get(); }
	/// Gets the ray description buffer.
	beg::api::UnorderedAccessView *const& RayDescUAV() { return m_rayDescUAV.get(); }

	/// Gets the ray geometry buffer.
	beg::api::ShaderResourceView *const& RayGeometry() const { return m_rayGeometrySRV.get(); }
	/// Gets the ray geometry buffer.
	beg::api::UnorderedAccessView *const& RayGeometryUAV() { return m_rayGeometryUAV.get(); }

	/// Gets the ray light buffer.
	beg::api::ShaderResourceView *const& RayLight() const { return m_rayLightSRV.get(); }
	/// Gets the ray light buffer.
	beg::api::UnorderedAccessView *const& RayLightUAV() { return m_rayLightUAV.get(); }

	/// Gets the ray debug buffer.
	beg::api::ShaderResourceView *const& RayDebug() const { return m_rayDebugSRV.get(); }
	/// Gets the ray debug buffer.
	beg::api::UnorderedAccessView *const& RayDebugUAV() { return m_rayDebugUAV.get(); }

	/// Sets the ray grid.
	void SetRayGrid(beg::api::Buffer *linkBuf, beg::api::ShaderResourceView *linkSRV, beg::api::UnorderedAccessView *linkUAV,
		beg::api::Buffer *listBuf, beg::api::ShaderResourceView *listSRV, beg::api::UnorderedAccessView *listUAV,
		beg::api::Texture3D *gridBTex, beg::api::ShaderResourceView *gridBSRV, beg::api::UnorderedAccessView *gridBUAV,
		beg::api::Texture3D *gridETex, beg::api::ShaderResourceView *gridESRV, beg::api::UnorderedAccessView *gridEUAV)
	{
		m_rayLinks = linkBuf;
		m_rayLinkSRV = linkSRV;
		m_rayLinkUAV = linkUAV;
		m_rayList = listBuf;
		m_rayListSRV = listSRV;
		m_rayListUAV = listUAV;
		m_rayGridBegin = gridBTex;
		m_rayGridBeginUAV = gridBUAV;
		m_rayGridBeginSRV = gridBSRV;
		m_rayGridEnd = gridETex;
		m_rayGridEndUAV = gridEUAV;
		m_rayGridEndSRV = gridESRV;
	}

	/// Gets the ray list node buffer.
	beg::api::Buffer *const& RayListBuffer() const { return m_rayList.get(); }
	/// Gets the ray list node buffer.
	beg::api::ShaderResourceView *const& RayList() const { return m_rayListSRV.get(); }
	/// Gets the ray list node buffer.
	beg::api::UnorderedAccessView *const& RayListUAV() { return m_rayListUAV.get(); }

	/// Gets the ray link buffer.
	beg::api::Buffer *const& RayLinkBuffer() const { return m_rayLinks.get(); }
	/// Gets the ray link buffer.
	beg::api::ShaderResourceView *const& RayLinks() const { return m_rayLinkSRV.get(); }
	/// Gets the ray link buffer.
	beg::api::UnorderedAccessView *const& RayLinkUAV() { return m_rayLinkUAV.get(); }

	/// Gets the ray grid texture.
	beg::api::Texture3D *const& RayGridBeginTexture() const { return m_rayGridBegin.get(); }
	/// Gets the ray grid texture.
	beg::api::ShaderResourceView *const& RayGridBegin() const { return m_rayGridBeginSRV.get(); }
	/// Gets the ray grid texture.
	beg::api::UnorderedAccessView *const& RayGridBeginUAV() { return m_rayGridBeginUAV.get(); }

	/// Gets the ray grid texture.
	beg::api::Texture3D *const& RayGridEndTexture() const { return m_rayGridEnd.get(); }
	/// Gets the ray grid texture.
	beg::api::ShaderResourceView *const& RayGridEnd() const { return m_rayGridEndSRV.get(); }
	/// Gets the ray grid texture.
	beg::api::UnorderedAccessView *const& RayGridEndUAV() { return m_rayGridEndUAV.get(); }

	/// Gets the maximum number of rays.
	uint4 RayCount() const { return m_maxRayCount; }
};

/// Ray grid generator.
class RayGridGen
{
private:
	bem::vector<uint4, 3> m_resolution;
	uint4 m_maxRayCount;
	uint4 m_maxRayLinkCount;

	lean::resource_ptr<beg::Material> m_material;

	beg::api::Effect *m_effect;
	beg::api::EffectTechnique *m_march;
	beg::api::EffectTechnique *m_inject;
	beg::api::EffectTechnique *m_inline;
	beg::api::EffectTechnique *m_pFilter;

	beg::api::EffectVector *m_boundedTraversalLimitsVar;

	beg::api::EffectShaderResource *m_activeRayListVar;
	beg::api::EffectUnorderedAccessView *m_activeRayListUAVVar;

	beg::api::EffectConstantBuffer *m_voxelRepConstVar;
	beg::api::EffectConstantBuffer *m_raySetConstVar;

	beg::api::EffectUnorderedAccessView *m_rayDescUAVVar;
	beg::api::EffectShaderResource *m_rayDescVar;
	beg::api::EffectShaderResource *m_rayGeometryVar;
	beg::api::EffectUnorderedAccessView *m_rayGeometryUAVVar;
	beg::api::EffectUnorderedAccessView *m_rayDebugUAVVar;

	beg::api::EffectShaderResource *m_voxelRepVar;

	beg::api::EffectShaderResource *m_gridIdxListVar;
	beg::api::EffectUnorderedAccessView *m_gridIdxListUAVVar;
	beg::api::EffectShaderResource *m_rayLinkListVar;
	beg::api::EffectUnorderedAccessView *m_rayLinkListUAVVar;
	beg::api::EffectShaderResource *m_rayInlineListVar;
	beg::api::EffectUnorderedAccessView *m_rayInlineListUAVVar;

	beg::api::EffectScalar * m_rayGridIdxBaseAddressVar;
	beg::api::EffectScalar * m_rayLinkBaseAddressVar;

	beg::api::EffectShaderResource *m_gridBeginVar, *m_gridEndVar;
	beg::api::EffectUnorderedAccessView *m_gridBeginUAVVar, *m_gridEndUAVVar;

	beg::api::EffectUnorderedAccessView *m_counterUAVVar;
	beg::api::EffectUnorderedAccessView *m_groupDispatchUAVVar;

	lean::scoped_ptr<cuda::B40Sorter> m_sorter;
	uint4 m_rayLinkCount;

	struct DoubleBuffered
	{
		lean::com_ptr<ID3D11Buffer> m_rayListBuffer;
		scoped_cgr_ptr m_rayListCGR;

		lean::com_ptr<ID3D11ShaderResourceView> m_rayLinkSRV;
		lean::com_ptr<ID3D11UnorderedAccessView> m_rayLinkUAV;
		uint4 m_rayLinkOffset;
	
		lean::com_ptr<ID3D11ShaderResourceView> m_rayGridIdxSRV;
		lean::com_ptr<ID3D11UnorderedAccessView> m_rayGridIdxUAV;
		uint4 m_rayGridIdxOffset;

		lean::com_ptr<ID3D11UnorderedAccessView> m_rayGridIdxLinkUAV;
	};
	DoubleBuffered m_main, m_aux;

	lean::com_ptr<ID3D11Buffer> m_rayListBuffer;
	lean::com_ptr<ID3D11ShaderResourceView> m_rayListSRV;
	lean::com_ptr<ID3D11UnorderedAccessView> m_rayListUAV;

	lean::com_ptr<beg::api::Texture3D> m_rayGridBegin, m_rayGridEnd;
	lean::com_ptr<beg::api::ShaderResourceView> m_rayGridBeginSRV, m_rayGridEndSRV;
	lean::com_ptr<beg::api::UnorderedAccessView> m_rayGridBeginUAV, m_rayGridEndUAV;

	lean::com_ptr<beg::api::Buffer> m_counterBuffer;
	lean::com_ptr<beg::api::UnorderedAccessView> m_counterUAV;
	
	lean::com_ptr<beg::api::Buffer> m_stagingBuffer;

	lean::com_ptr<beg::api::Buffer> m_dispatchBuffer;
	lean::com_ptr<beg::api::UnorderedAccessView> m_dispatchUAV;

	/// Binds the given material.
	void BindMaterial(beg::Material *material);

	/// Computes the number of groups to dispatch.
	void ComputeDispatchGroupCount(beg::api::EffectTechnique *technique, uint4 passID,
		beg::api::UnorderedAccessView *dispatchUAV, beg::api::DeviceContext *context) const;

	/// Scatters the rays into the ray grid.
	void March(RaySet &raySet, VoxelRep &voxelRep, uint4 traceStep, besc::RenderContext &context);
	/// Scatters the rays into the ray grid.
	void Sort(RaySet &raySet, besc::RenderContext &context);
	/// Scatters the rays into the ray grid.
	void Inject(RaySet &raySet, besc::RenderContext &context);
	/// Inlines the rays with the list of ray links.
	void Inline(RaySet &raySet, besc::RenderContext &context);

public:
	/// Constructor.
	RayGridGen(const lean::utf8_ntri &file, uint4 maxRayCount, bem::vector<uint4, 3> resolution, uint4 avgRayLength,
		besc::EffectDrivenRenderer *renderer, besc::ResourceManager *resourceManager);
	/// Destructor.
	~RayGridGen();

	/// Processes / commits changes.
	void Commit();

	/// Scatters the rays into the ray grid.
	void ConstructRayGrid(RaySet &raySet, VoxelRep &voxelRep, uint4 traceStep, besc::RenderContext &context,
		IncrementalGPUTimer *pMarch, IncrementalGPUTimer *pSort, IncrementalGPUTimer *pInject, IncrementalGPUTimer *pInline);

	/// Binds the given ray set to this generator.
	void Bind(RaySet &raySet)
	{
		raySet.SetRayGrid(m_main.m_rayListBuffer, m_main.m_rayLinkSRV, m_main.m_rayLinkUAV,
			m_aux.m_rayListBuffer, m_rayListSRV, m_rayListUAV,
			m_rayGridBegin, m_rayGridBeginSRV, m_rayGridBeginUAV,
			m_rayGridEnd, m_rayGridEndSRV, m_rayGridEndUAV);
	}

	/// Gets the number of tracing steps.
	uint4 GetTraceStepCount() const;
	/// Gets the number of ray links.
	uint4 GetRayLinkCount() const { return m_rayLinkCount; }
};

} // namespace

} // namespace
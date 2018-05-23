#pragma once

#include "Tracing.h"

#include <beCore/beShared.h>
#include <beGraphics/Any/beAPI.h>
#include <lean/smart/com_ptr.h>

#include <vector>

namespace app
{

namespace tracing
{

class RaySet;

struct TracingBinderConstants
{
	uint4 InputCount;
	uint4 InputOffset;
	uint4 MaxOutputCount;
	uint4 BatchBase;
};

class TracingEffectBinderPool : public beCore::Resource
{
public:
	struct IntermediatePipelineBuffer
	{
		lean::com_ptr<beg::api::Buffer> buffer;
		lean::com_ptr<beg::api::ShaderResourceView> srv;
		lean::com_ptr<beg::api::UnorderedAccessView> uav;
		uint4 size;
		uint4 used;

		IntermediatePipelineBuffer() : size(0), used(0) { }
	};
	typedef std::vector<IntermediatePipelineBuffer> buffer_vector;

	struct IntermediateBuffer
	{
		beg::api::Buffer *buffer;
		beg::api::ShaderResourceView *srv;
		beg::api::UnorderedAccessView *uav;

		IntermediateBuffer() : buffer(), srv(), uav() { }
		IntermediateBuffer(const IntermediatePipelineBuffer &right) : buffer(right.buffer), srv(right.srv), uav(right.uav) { }
	};

private:
	lean::com_ptr<beg::api::Device> m_device;

	RaySet *m_pRaySet;

	uint4 m_usedCounter;

	buffer_vector m_triangles;
	buffer_vector m_geometry;
	buffer_vector m_voxels;

	uint4 m_primitiveVoxelMultiplier;

	buffer_vector m_constants;
	buffer_vector m_dispatch;
	buffer_vector m_draw;

	lean::com_ptr<beg::api::UnorderedAccessView> m_debugUAV;

public:
	/// Constructor.
	TracingEffectBinderPool(beg::api::Device *device, 
		RaySet *pRaySet = nullptr);
	/// Destructor.
	~TracingEffectBinderPool();

	/// Sets the ray hierarchy. May be null.
	void SetRaySet(RaySet *pSet) { m_pRaySet = pSet; }
	/// Gets the ray hierarchy. May be null.
	RaySet* GetRaySet() { return m_pRaySet; }
	/// Gets the ray hierarchy. May be null.
	const RaySet* GetRaySet() const { return m_pRaySet; }
	
	/// Resizes the buffers.
	void SetupBuffers(uint4 primitiveCount, uint4 geometryBufferStride, uint4 geometryBufferElements, uint4 voxelMulti);

	/// Invalidates cached values.
	void InvalidateCaches();

	/// Gets a fitting geometry buffer.
	IntermediateBuffer GetTriangleBuffer(uint4 triangleCount);
	/// Gets a fitting geometry buffer.
	IntermediateBuffer GetGeometryBuffer(uint4 bytes);
	/// Gets a fitting voxel buffer.
	IntermediateBuffer GetVoxelBuffer(uint4 primitiveCount);

	/// Gets a constant buffer.
	IntermediateBuffer GetConstantBuffer();
	/// Gets a dispatch buffer.
	IntermediateBuffer GetDispatchBuffer();
	/// Gets a dispatch buffer.
	IntermediateBuffer GetSingleInstanceBuffer();

	/// Gets the UAV.
	void SetDebugUAV(beg::api::UnorderedAccessView *uav) { m_debugUAV = uav; }
	/// Gets the UAV.
	beg::api::UnorderedAccessView* GetDebugUAV() { return m_debugUAV; }
};

} // namespace

} // namespace
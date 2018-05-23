/******************************************************/
/* Object-order Ray Tracing Demo (c) Tobias Zirr 2013 */
/******************************************************/

#include "stdafx.h"

#include "RayTracing/TracingEffectBinderPool.h"
#include "RayTracing/Pipeline.h"
#include "RayTracing/Ray.h"

#include <beGraphics/Any/beDevice.h>
#include <beGraphics/Any/beDeviceContext.h>
#include <beGraphics/Any/beBuffer.h>
#include <beGraphics/Any/beTexture.h>

#include <beGraphics/DX/beError.h>

#define SINGLE_BUFFER_ONE(x) x

namespace app
{

namespace tracing
{

namespace
{

TracingEffectBinderPool::buffer_vector CreateGeometryBuffers(beg::api::Device *device, uint4 bufferSize)
{
	const uint4 bufferSizes[] = { bufferSize, bufferSize / 2, bufferSize / 2, bufferSize / 4 };

	const uint4 bufferCount = SINGLE_BUFFER_ONE( lean::arraylen(bufferSizes) );
	TracingEffectBinderPool::buffer_vector buffers(bufferCount);
	
	for (uint4 i = 0; i < bufferCount; ++i)
	{
		buffers[i].size = bufferSizes[i];
		buffers[i].buffer = beg::Any::CreateStructuredBuffer(device,
				D3D11_BIND_STREAM_OUTPUT | D3D11_BIND_SHADER_RESOURCE,
				buffers[i].size, 1, D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS
			);
		buffers[i].srv = beg::Any::CreateRawSRV(buffers[i].buffer);
	}

	return buffers;
}

TracingEffectBinderPool::buffer_vector CreateStructuredBuffers(beg::api::Device *device, uint4 structSize, uint4 bufferSize, bool bVarySize = false, uint4 minSize = 0)
{
	const uint4 bufferSizes[] = { max(bufferSize, minSize),
		bVarySize ? max(bufferSize / 2, minSize) : bufferSize, 
		bVarySize ? max(bufferSize / 2, minSize) : bufferSize, 
		bVarySize ? max(bufferSize / 4, minSize) : bufferSize }; 

	const uint4 bufferCount = SINGLE_BUFFER_ONE( lean::arraylen(bufferSizes) );
	TracingEffectBinderPool::buffer_vector buffers(bufferCount);
	
	for (uint4 i = 0; i < bufferCount; ++i)
	{
		buffers[i].size = bufferSizes[i];
		buffers[i].buffer = beg::Any::CreateStructuredBuffer(device,
				D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE,
				structSize, buffers[i].size
			);
		buffers[i].srv = beg::Any::CreateSRV(buffers[i].buffer);
		buffers[i].uav = beg::Any::CreateCountingUAV(buffers[i].buffer);
	}

	return buffers;
}

TracingEffectBinderPool::buffer_vector CreateConstantBuffers(beg::api::Device *device, uint4 bufferSize)
{
	const uint4 bufferCount = SINGLE_BUFFER_ONE(8);
	TracingEffectBinderPool::buffer_vector buffers(bufferCount);
	
	for (uint4 i = 0; i < bufferCount; ++i)
	{
		buffers[i].size = bufferSize;
		buffers[i].buffer = beg::Any::CreateConstantBuffer(device, bufferSize);
	}

	return buffers;
}

TracingEffectBinderPool::buffer_vector CreateDispatchBuffers(beg::api::Device *device)
{
	const uint4 bufferCount = SINGLE_BUFFER_ONE(8);
	TracingEffectBinderPool::buffer_vector buffers(bufferCount);
	
	for (uint4 i = 0; i < bufferCount; ++i)
	{
		buffers[i].size = sizeof(uint4) * 4;
		buffers[i].buffer = beg::Any::CreateStructuredBuffer(device, D3D11_BIND_UNORDERED_ACCESS,
			sizeof(uint4), 4, D3D11_RESOURCE_MISC_DRAWINDIRECT_ARGS | D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS);
		buffers[i].uav = beg::Any::CreateRawUAV(buffers[i].buffer);
	}

	return buffers;
}

const uint4 SingleInstanceBufferData[4] = { 0, 1, 0, 0 };

TracingEffectBinderPool::buffer_vector CreateDrawBuffers(beg::api::Device *device)
{
	const uint4 bufferCount = SINGLE_BUFFER_ONE(8);
	TracingEffectBinderPool::buffer_vector buffers(bufferCount);
	
	for (uint4 i = 0; i < bufferCount; ++i)
	{
		buffers[i].size = sizeof(uint4) * 4;
		buffers[i].buffer = beg::Any::CreateStructuredBuffer(device, 0,
			sizeof(uint4), 4, D3D11_RESOURCE_MISC_DRAWINDIRECT_ARGS, SingleInstanceBufferData);
	}

	return buffers;
}

} // namespace

// Constructor.
TracingEffectBinderPool::TracingEffectBinderPool(beg::api::Device *device,
		RaySet *pRaySet)
	: m_device( LEAN_ASSERT_NOT_NULL(device) ),
	m_pRaySet( pRaySet ),

	m_usedCounter( 0 ),

	m_constants( CreateConstantBuffers(device, sizeof(TracingBinderConstants)) ),
	m_dispatch( CreateDispatchBuffers(device) ),
	m_draw( CreateDrawBuffers(device) )
{
}

// Destructor.
TracingEffectBinderPool::~TracingEffectBinderPool()
{
}

// Resizes the buffers.
void TracingEffectBinderPool::SetupBuffers(uint4 primitiveCount, uint4 geometryBufferStride, uint4 geometryBufferElements, uint4 voxelMulti)
{
	m_primitiveVoxelMultiplier = voxelMulti;

	m_triangles = CreateGeometryBuffers(m_device, (sizeof(float) * 3 * 3) * geometryBufferElements);
	m_geometry = CreateGeometryBuffers(m_device, geometryBufferStride * geometryBufferElements);
	m_voxels = CreateStructuredBuffers(m_device, sizeof(VoxelTriangle), primitiveCount * m_primitiveVoxelMultiplier, true, 512U * 1024U);
}

// Invalidates cached values.
void TracingEffectBinderPool::InvalidateCaches()
{
}

namespace
{

template <class Collection>
TracingEffectBinderPool::IntermediateBuffer GetBuffer(Collection &buffers, uint4 minSize, uint4 request)
{
	TracingEffectBinderPool::IntermediateBuffer result;

	const size_t bufferCount = buffers.size();

	size_t bufferIdx = -1;
	uint4 bufferUsed = -1;

	for (size_t j = 0; j < bufferCount; ++j)
		if (buffers[j].size >= minSize && buffers[j].used < bufferUsed)
		{
			bufferIdx = j;
			bufferUsed = buffers[j].used;
		}

	if (bufferIdx < bufferCount)
	{
		result.buffer = buffers[bufferIdx].buffer;
		result.srv = buffers[bufferIdx].srv;
		result.uav = buffers[bufferIdx].uav;

		buffers[bufferIdx].used = request;
	}

	return result;
}

} // namespace

// Gets a fitting geometry buffer.
TracingEffectBinderPool::IntermediateBuffer TracingEffectBinderPool::GetGeometryBuffer(uint4 bytes)
{
	return GetBuffer(m_geometry, bytes, ++m_usedCounter);
}

// Gets a fitting geometry buffer.
TracingEffectBinderPool::IntermediateBuffer TracingEffectBinderPool::GetTriangleBuffer(uint4 triCount)
{
	return GetBuffer(m_triangles, (sizeof(float) * 3 * 3) * triCount, ++m_usedCounter);
}

// Gets a fitting voxel buffer.
TracingEffectBinderPool::IntermediateBuffer TracingEffectBinderPool::GetVoxelBuffer(uint4 primitiveCount)
{
	return GetBuffer(m_voxels, primitiveCount * m_primitiveVoxelMultiplier, ++m_usedCounter);
}

TracingEffectBinderPool::IntermediateBuffer TracingEffectBinderPool::GetConstantBuffer()
{
	return GetBuffer(m_constants, 0, ++m_usedCounter);
}

// Gets a dispatch buffer.
TracingEffectBinderPool::IntermediateBuffer TracingEffectBinderPool::GetDispatchBuffer()
{
	return GetBuffer(m_dispatch, 0, ++m_usedCounter);
}

// Gets a dispatch buffer.
TracingEffectBinderPool::IntermediateBuffer TracingEffectBinderPool::GetSingleInstanceBuffer()
{
	return GetBuffer(m_draw, 0, ++m_usedCounter);
}

} // namespace

} // namespace

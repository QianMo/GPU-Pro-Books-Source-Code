#ifndef BE_TRACING_SCENE_H
#define BE_TRACING_SCENE_H

#include "Engine/BindPoints.fx"
#include "Pipelines/Tracing/Ray.fx"
#include "Pipelines/Tracing/RaySet.fx"

#ifdef BE_TRACING_SETUP
	#define BE_TRACING_PREBOUND_S(semantic) semantic
#else
	#define BE_TRACING_PREBOUND_S(semantic) prebound_s(semantic)
#endif

struct VoxelTriangle
{
	uint CellID;
	uint TriangleID;
};

struct TracingConstantLayout
{
	uint InputCount;
	uint InputOffset;
	uint MaxOutputCount;
	uint BatchBase;
};

cbuffer TracingConstants
{
	TracingConstantLayout TracingConstants;
}

RWStructuredBuffer<TracedGeometry> TracedGeometryUAV : bindpoint_s(TracedGeometryUAV, u0);
/// Auxiliary buffer for debug data per ray.
RWStructuredBuffer<RayDebug> DebugUAV : bindpoint_s(DebugUAV, u1);
// Pairs of triangles & ray grid cells to be tested for intersection.
RWStructuredBuffer<VoxelTriangle> TracingVoxelOut : bindpoint_s(TracingVoxelOut, u2);
StructuredBuffer<VoxelTriangle> TracingVoxelIn : TracingVoxelIn;

/// Ray light.
RWStructuredBuffer<TracedLight> TracedLightUAV : bindpoint_s(TracedLightUAV, u0);

/// GPU-based CS dispatching.
RWByteAddressBuffer GroupDispatchUAV : bindpoint_s(GroupDispatchUAV, u2);

/// Auxiliary buffers for transformed geometry of current batch.
ByteAddressBuffer TracingTriangles : TracingTriangles;
ByteAddressBuffer TracingGeometry : TracingGeometry;

/// Ray grid w/ cell ranges of ray link indices.
Texture3D<uint> RayGridBegin : prebound_s(bindpoint_s(RayGridBegin, t13));
Texture3D<uint> RayGridEnd : prebound_s(bindpoint_s(RayGridEnd, t12));
/// Ray links.
ByteAddressBuffer RayLinkBuffer : prebound_s(bindpoint_s(RayLinks, t11));
/// Ray descriptions.
StructuredBuffer<RayDesc> RayDescriptionBuffer : prebound_s(bindpoint_s(RayDescriptions, t10));
/// Ray geometry.
StructuredBuffer<TracedGeometry> TracedGeometryBuffer : prebound_s(bindpoint_s(TracedGeometry, t9));
/// Compacted ray descs.
ByteAddressBuffer RayInlinedBuffer : prebound_s(bindpoint_s(RayListNodes, t8));

#endif
/******************************************************/
/* Object-order Ray Tracing Demo (c) Tobias Zirr 2013 */
/******************************************************/

#define BE_RENDERABLE_INCLUDE_WORLD

// #define TRI_COLOR // Display scene triangles.

#hookincl "2.0/Materials/Textured.fx" ...

#include "Pipelines/Tracing/Ray.fx"
#include "Pipelines/Tracing/CompactRay.fx"
#include "Pipelines/Tracing/Scene.fx"
#include "Pipelines/Tracing/VoxelRep.fx"

#include <Utility/Conservative.fx>
#include <Utility/Math.fx>

// #define RECORD_INTERSECTION_STATISTICS // Slow, enable together with SHOW_INTERSECTION_STATISTICS in Pipeline.cpp to see statistics
// #define COUNT_TESTS_PER_RAY // Debug switch, results can be displayed via color coding in RayGen.fx

#define COMPACT_RAYS // Use inlined compacted rays to achieve good read coherency.
#define LOCKFREE // Right now, it appears that ray locking is unnecessary. This might change with an increased number of cores.
// #define PLAINUPDATE // Use normal writes rather than InterlockedExchange to update rays.

#ifndef LOCKFREE
	// Prechecking the ray distance accelerates ray update when using locks
	#define PRECHECK_DIST
#endif

// Relevant triangle information
struct TransformedTraceVertex
{
	float3 Position : Position1;
	float3 Normal : Normal1;
	float2 TexCoord : TexCoord4;
};
static const uint TransformedTraceVertexSize = 8;

// Triangle data + vertex pos transformed to grid space
struct RasterizedTraceVertex : TransformedTraceVertex
{
	float4 VPosition : Position;
};

RasterizedTraceVertex VSRenderToRayGrid(Vertex v)
{
	#hookcall transformState = Transform(v);
	TransformedVertex r = VSMain(v);

	// Triangle data
	RasterizedTraceVertex o;
	float3 worldPos = GetWorldPosition(transformState).xyz;
	o.Position = worldPos;
	o.Normal = r.NormalDepth.xyz;
	o.TexCoord = r.TexCoord;

	// Grid-space triangle
	o.VPosition = float4( (worldPos - VoxelRep.Min) * VoxelRep.UnitScale, 1.0f );
	
	return o;
}

// Triangle as stored by intermediate storage buffers:
// --
// Separate buffers for vertex positions (-> intersection tests)
// and tex coords etc. (-> interpolation / hit point evaluation)
struct TransformedTracePosTriangle
{
	float3 Vertices[3];
};
static const uint TransformedTracePositionVertexSize = 3;

struct TransformedTraceTriangle
{
	TransformedTraceVertex Vertices[3];
};
static const uint TransformedTraceGeometryVertexSize = TransformedTraceVertexSize - TransformedTracePositionVertexSize;

// Load positions from vertex position buffer
TransformedTracePosTriangle ExtractPosTriangle(ByteAddressBuffer triangles, ByteAddressBuffer geometry, uint triangleIdx)
{
	TransformedTracePosTriangle tri;

	uint triangleAddress = (TransformedTracePositionVertexSize * 4) * 3 * triangleIdx;
	[unroll]
	for (uint j = 0; j < 3; ++j)
	{
		tri.Vertices[j] = asfloat( triangles.Load3(triangleAddress) );
		triangleAddress += (TransformedTracePositionVertexSize * 4);
	}

	return tri;
}

// Load additional triangle data from secondary buffer
TransformedTraceTriangle ExtractTriangle(ByteAddressBuffer triangles, ByteAddressBuffer geometry, uint triangleIdx)
{
	TransformedTracePosTriangle posTri = ExtractPosTriangle(triangles, geometry, triangleIdx);
	TransformedTraceTriangle tri;

	static const uint triangleDataWords = (4 * TransformedTraceVertexSize - 1) / 4;
	float4 triangleData[triangleDataWords];

	uint triangleAddress = (TransformedTraceGeometryVertexSize * 4) * 3 * triangleIdx;
	[unroll]
	for (uint j = 0; j < triangleDataWords; ++j)
	{
		triangleData[j] = asfloat( geometry.Load4(triangleAddress) );
		triangleAddress += 4 * 4;
	}

	// Convert to more convenient triangle structure
	[unroll]
	for (uint j = 0, f = 0; j < 3; ++j)
	{
		tri.Vertices[j].Position = posTri.Vertices[j];
		[unroll]
		for (uint k = 0; k < 3; ++k, ++f)
			tri.Vertices[j].Normal[k] = triangleData[f / 4][f % 4];
		[unroll]
		for (uint k = 0; k < 2; ++k, ++f)
			tri.Vertices[j].TexCoord[k] = triangleData[f / 4][f % 4];
	}

	return tri;
}

struct ConservativeVoxelizationVertex
{
	float4 Position : SV_Position;
	nointerpolation float4 AABB : TexCoord0;
	nointerpolation uint TriangleID : TexCoord1;
	nointerpolation float2 Range : TexCoord2;
};

struct ConservativeVoxelizationWithStreamOutVertex : ConservativeVoxelizationVertex
{
	TransformedTraceVertex TriangleVertex;
};

[MaxVertexCount(3)]
void GSRenderToRayGrid(triangle RasterizedTraceVertex tri[3],
	inout TriangleStream<ConservativeVoxelizationWithStreamOutVertex> triStream,
	uint triangleID : SV_PrimitiveID)
{
	// Dominant axis
	float3 n = cross(tri[1].VPosition.xyz - tri[0].VPosition.xyz, tri[2].VPosition.xyz - tri[0].VPosition.xyz);
	float3 an = abs(n);
	float man = max3(an);

	// Project triangle to dominant axis
	float4 v[3] = { tri[0].VPosition, tri[1].VPosition, tri[2].VPosition };
	float2 maxis = 1.0f;
	float2 halfPixelOffset = VoxelRep.RastVoxelWidth;
	uint2 resolution = VoxelRep.Resolution.xy;
	
	if (an.x >= man)
	{
		[unroll] for (int i = 0; i < 3; ++i) v[i] = v[i].yzxw;
		maxis.x = -1.0f;
		resolution = VoxelRep.Resolution.yz;
	}
	else if (an.y >= man)
	{
		[unroll] for (int i = 0; i < 3; ++i) v[i] = v[i].zxyw;
		maxis.y = -1.0f;
		resolution = VoxelRep.Resolution.zx;
	}
	
	// Transform projected triangle to normalized device coordinates
	[unroll] for (int i = 0; i < 3; ++i)
		v[i].xy *= resolution * VoxelRep.RastVoxelWidth;
	[unroll] for (int i = 0; i < 3; ++i)
		v[i].xy = v[i].xy * float2(2.0f, -2.0f) - float2(1.0f, -1.0f);

	// Compute conservative hull
	float4 p[3];
	float4 aabb;
	ConservativeTriBoundsOrtho(p, aabb, v, 1.005f * halfPixelOffset);
	
	ConservativeVoxelizationWithStreamOutVertex o;
	// Transform AABB to normalized device coords & encode projection axis in sign (-> maxis)
	o.AABB = max( saturate(aabb * float2(0.5f, -0.5f).xyxy + 0.5f).xwzy * VoxelRep.RastResolution, 0.000001f ) * maxis.xyxy;
	o.TriangleID = triangleID;
	float3 zv = float3(v[0].z, v[1].z, v[2].z);
	o.Range = float2(min3(zv), max3(zv));

	// Add conservative voxelization vertices & stream-out data
	[unroll]
	for (int i = 0; i < 3; ++i)
	{
		o.Position = p[i];
		o.TriangleVertex = (TransformedTraceVertex) tri[i];
		triStream.Append(o);
	}
}

void PSRenderToRayGrid(in ConservativeVoxelizationVertex p, uniform bool bLite = false)
{
	float4 aabb = abs(p.AABB);
	clip( float4(p.Position.xy - aabb.xy, aabb.zw - p.Position.xy) );

	// Compute conservative Z range
	float depthDelta = fwidth(p.Position.z);
	float2 minMaxZ = p.Position.z + 0.505f * float2(-depthDelta, depthDelta);
	minMaxZ = clamp(minMaxZ, p.Range.x, p.Range.y);

	uint3 gridIdx = uint3((uint2) p.Position.xy, 0);
	uint3 sliceStep = uint3(0, 0, 1);
	
	// Reconstruct main axis
	if (p.AABB.x < 0.0f)
	{
		gridIdx = gridIdx.zxy; // from yzx
		sliceStep = uint3(1, 0, 0);
	}
	else if (p.AABB.y < 0.0f)
	{
		gridIdx = gridIdx.yzx; // from zxy
		sliceStep = uint3(0, 1, 0);
	}

	// Reconstruct original position
	uint resolutionZ = dot(sliceStep, VoxelRep.Resolution);
	uint2 minMaxSlice = min( (uint2) (minMaxZ * resolutionZ), resolutionZ - 1 );
	gridIdx += minMaxSlice.x * sliceStep;

	// Loop over ray grid cells in current voxel column
	for (uint slice = minMaxSlice.x; slice <= minMaxSlice.y; ++slice, gridIdx += sliceStep)
	{
		// Skip cells w/o rays
		if (RayGridEnd[gridIdx] != 0)
		{
			// Schedule triangle to be tested w/ grid cell
			uint cellID = bitpack(gridIdx);
			VoxelTriangle vt = { cellID, p.TriangleID };
#ifdef APPEND_CONSUME_TRAVERSAL
			TracingVoxelOut.Append(vt);
#else
			uint taskID = TracingVoxelOut.IncrementCounter();
			TracingVoxelOut[taskID] = vt;
#endif
		}
	}
}

static const uint2 TraceViewportDimension = uint2(4096, 4096);
static const uint TraceTaskHeight = 2;

// Nothing done in VS
struct NoInput { };
NoInput VSIntersectCellRays(NoInput v)
{
	NoInput o;
	return o;
}

struct TaskControlPoint
{
	float4 Position : SV_Position;
	nointerpolation uint4 Triangle_Cell_RayLinkBase_Idx__Ray_Count : TexCoord0;
	uint Viewport : SV_ViewportArrayIndex;
};

[MaxVertexCount(3)]
void GSIntersectCellRays(point NoInput nothing[1], inout TriangleStream<TaskControlPoint> triStream, uint taskID : SV_PrimitiveID)
{
	VoxelTriangle task = TracingVoxelIn[taskID];

	uint3 gridIdx = bitunpack3(task.CellID);
	uint listNodeIdx = RayGridBegin[gridIdx];
	uint listNodeEndIdx = RayGridEnd[gridIdx];
	uint rayCount = (listNodeIdx < listNodeEndIdx) ? listNodeEndIdx - listNodeIdx : 0;
	
	// The following code creates a triangle that forms a quad with the viewport boundaries
	// The number of pixels in the quad is greater or equal than the number of rays in the
	// ray grid cell currently processed.
	// Thus, when the quad is rasterized, a sufficient number of pixel shaders are executed
	// that allows for each thread to intersect exactly one ray with the triangle currently
	// processed (-> task.TriangleID).

	// Pass relevant info to pixel shader thread group
	TaskControlPoint o;
	o.Viewport = 1;
	o.Triangle_Cell_RayLinkBase_Idx__Ray_Count = uint4(task.TriangleID, task.CellID, listNodeIdx, rayCount); 

	// Adapt height of pixel shader thread group
	uint taskHeightU = max(TraceTaskHeight, ceil_div(rayCount, TraceViewportDimension.x * 2) * 2);

	// Compute extents pixel shader thread group
	float taskHeight = taskHeightU;
	float taskWidth = ceil_div(rayCount, taskHeightU);

	// Optimize small thread groups
	if (taskWidth < 2.0f) taskHeight = rayCount;

	// Compute thread group extents & origin in normalized device coordinates
	float2 pixelWidth = 2.0f / TraceViewportDimension;
	float2 taskViewportExt = float2(taskWidth, taskHeight) * pixelWidth;
	float taskMaxViewportExt = max2(taskViewportExt);
	float2 taskViewportOrig = float2(-1.0f + taskViewportExt.x, 1.0f - taskViewportExt.y);

	// Create triangle that forms a quad of computed extent with the viewport bounds
	o.Position = float4(-2.0f * taskMaxViewportExt + taskViewportOrig.x, taskViewportOrig.y, 0.5f, 1.0f);
	triStream.Append(o);
	o.Position = float4(taskViewportOrig.x, 2.0f * taskMaxViewportExt + taskViewportOrig.y, 0.5f, 1.0f);
	triStream.Append(o);
	o.Position = float4(taskViewportOrig, 0.5f, 1.0f);
	triStream.Append(o);
}

// Evaluates and writes hit point data to the closest hit record of the given ray.
void UpdateRayGeometry(uint rayIdx, RayDesc ray, float distance, TransformedTraceTriangle tri, float3 baryCoords, float3 edge1, float3 edge2)
{
	// Interpolate
	float3 normal = normalize( interpolate(tri.Vertices[0].Normal, tri.Vertices[1].Normal, tri.Vertices[2].Normal, baryCoords) );
	float2 texCoord = interpolate(tri.Vertices[0].TexCoord, tri.Vertices[1].TexCoord, tri.Vertices[2].TexCoord, baryCoords);
	
	float mipDistance = (length(ray.Orig - Perspective.CamPos) + distance) / abs(dot(normal, ray.Dir));

	float2 tedge1 = tri.Vertices[1].TexCoord - tri.Vertices[0].TexCoord;
	float2 tedge2 = tri.Vertices[2].TexCoord - tri.Vertices[0].TexCoord;

	float3 triTangent = (edge2 * tedge1.y - edge1 * tedge2.y);
	float3 triBitangent = (edge1 * tedge2.x - edge2 * tedge1.x);

	// TODO: Use actual screen resolution
	float2 ddux = float2( 1.0f / 1024 * abs(tedge2.x * tedge1.y - tedge1.x * tedge2.y) / length(triTangent) * mipDistance, 0.0f );
	float2 ddvx = float2( 0.0f, 1.0f / 1024 * abs(tedge1.y * tedge2.x - tedge2.y * tedge1.x) / length(triBitangent) * mipDistance );
	
	// Compute fragment data
	float4 diffuse = DiffuseColor;
	IF_COLORMAP(
		diffuse.xyz *= DiffuseTexture.SampleGrad(LinearSampler, texCoord, ddux, ddvx).xyz;
	)

	float4 specular = SpecularColor;
	IF_SPECULARMAP(
		specular.xyz *= SpecularTexture.SampleGrad(LinearSampler, texCoord, ddux, ddvx).xyz;
	)

	// Note: Could also insert normal map here, since we've already got tangent & bitangent vectors

#if defined(LOCKFREE) || defined(PLAINUPDATE)
	// So far, it looks like everything is written to global memory in time,
	// so we don't need to use InterlockedExchange for ordered update?
	TracedGeometryUAV[rayIdx].Normal.x = PackTracedNormal(normal).x;
	TracedGeometryUAV[rayIdx].Normal.y = PackTracedNormal(normal).y;
	TracedGeometryUAV[rayIdx].Diffuse = PackTracedColor(diffuse);
	TracedGeometryUAV[rayIdx].Specular = PackTracedColor(specular);
#else
	uint unusedBits1, unusedBits2, unusedBits3, unusedBits4;
	InterlockedExchange(TracedGeometryUAV[rayIdx].Normal.x, PackTracedNormal(normal).x, unusedBits1);
	InterlockedExchange(TracedGeometryUAV[rayIdx].Normal.y, PackTracedNormal(normal).y, unusedBits2);
	InterlockedExchange(TracedGeometryUAV[rayIdx].Diffuse, PackTracedColor(diffuse), unusedBits3);
	InterlockedExchange(TracedGeometryUAV[rayIdx].Specular, PackTracedColor(specular), unusedBits4);
#endif
}

// Updates the given ray with the information of the given hit point, if it is closer than the current hit point.
int UpdateRay(uint rayIdx, RayDesc ray, float distance, TransformedTraceTriangle tri, float3 baryCoords, float3 edge1, float3 edge2)
{
	// Distance of the new hit point,
	// asuint() retains the order of positive floats
	int newDist = asuint(distance);

	int lastDistBitsOrLocked = 0;
	int lockedCounter = 0;

#ifndef LOCKFREE
	[allow_uav_condition]
	do
#endif
	{
#ifdef PRECHECK_DIST
		int prelimDistBits = TracedGeometryUAV[rayIdx].Depth;

		// The absolute value always corresponds to the current
		// closest hit. Immediately discard farther hits
		if (newDist >= abs(prelimDistBits))
			return lockedCounter;
#endif
		// Atomically compare new hit to the current closest hit
		// and check if the ray is locked at the same time
		InterlockedMin(TracedGeometryUAV[rayIdx].Depth, newDist, lastDistBitsOrLocked);

		// Only entered if ray is unlocked (lastDistOrLocked >= 0)	
		// and new distance is less than old distance
		if (newDist < lastDistBitsOrLocked)
		{
#ifndef LOCKFREE
			// Atomically lock ray via the distance buffer
			// (= set distance to a negative value)
			int lastDistBits;
			InterlockedCompareExchange(TracedGeometryUAV[rayIdx].Depth, newDist, -newDist, lastDistBits);

			// Check if exchange successful and new distance still closest
			if (lastDistBits == newDist)
#endif
			{
				UpdateRayGeometry(rayIdx, ray, distance, tri, baryCoords, edge1, edge2);

#ifndef LOCKFREE
				// Unlock the ray by updating the distance buffer
				uint unusedBits;
				InterlockedExchange(TracedGeometryUAV[rayIdx].Depth, newDist, unusedBits);
#endif
			}
		}

		++lockedCounter;
	}
#ifndef LOCKFREE
	// Re-iterate until the ray has been unlocked
	while (lastDistBitsOrLocked < 0);
#endif

	return lockedCounter;
}

// Gets a description of the ray referenced by the given ray link.
RayDesc GetRay(uint rayLinkIdx, out uint rayIdx, out bool rayLinkLoaded)
{
	RayDesc ray;

#ifdef COMPACT_RAYS
	rayIdx = -1; // Not needed here, load later
	rayLinkLoaded = false;

	uint3 compactRay = RayInlinedBuffer.Load3(12 * rayLinkIdx);
	ray.Dir = ExtractDirection(compactRay.x);
	ray.Orig = ExtractCellOrigin(compactRay.yz);
#else
	rayIdx = RayLinkBuffer.Load(4 * rayLinkIdx);
	rayLinkLoaded = true;

	ray = RayDescriptionBuffer[rayLink.RayID];
#endif

	return ray;
}

// Gets the index of the ray referenced by the given ray link.
uint GetRayLinkAfterRay(uint rayLinkIdx, uint rayIdxIfLoaded, inout bool rayLinkLoaded)
{
#ifdef COMPACT_RAYS
	rayIdxIfLoaded = RayLinkBuffer.Load(4 * rayLinkIdx);
	rayLinkLoaded = true;
#endif
	return rayIdxIfLoaded;
}

bool IntersectRayWithTriangle(RayDesc ray, out float dist, out float3 baryCoords,
							  float3 v0, float3 edge1, float3 edge2, float3 normal,
							  float3 gridIdx)
{
	dist = 0.0f;
	baryCoords = 0.0f;
	
	float det = -dot(normal, ray.Dir);
	if (det < 0.0f)
		return false;

	float3 toOrig = ray.Orig - v0;
	dist = dot(normal, toOrig);

	float3 dirCrossToOrig = cross(ray.Dir, toOrig);
	
	float u = -dot(dirCrossToOrig, edge2);
	if (u < 0.0f || u > det || dist < 0.0f)
		return false;

	float v = dot(dirCrossToOrig, edge1);
	if (v < 0.0f || v > det - u)
		return false;

	// Hit detected, compute distance & point
	float oneOverDet = rcp(det);
	dist *= oneOverDet;
	float3 hitPoint = ray.Orig + dist * ray.Dir;
	
	// Make sure hit is IN cell
	float3 relativeHitPoint = (hitPoint - VoxelRep.Min) * VoxelRep.VoxelScale - gridIdx;
	float eps = 0.005f;
	if (any( floor(relativeHitPoint * (1.0f - 2.0f * eps) + eps) ))
		return false;

	// Compute barycentric coords
	u *= oneOverDet;
	v *= oneOverDet;
	baryCoords = float3(1.0f - u - v, u, v);

	return true;
}

void PSIntersectCellRays(in TaskControlPoint p)
{
	uint rayCount = p.Triangle_Cell_RayLinkBase_Idx__Ray_Count.w;
	uint triangleIdx = p.Triangle_Cell_RayLinkBase_Idx__Ray_Count.x;
	uint rayLinkBaseIdx = p.Triangle_Cell_RayLinkBase_Idx__Ray_Count.z;
	uint cellID = p.Triangle_Cell_RayLinkBase_Idx__Ray_Count.y;

	///// Triangle /////

	TransformedTracePosTriangle posTri = ExtractPosTriangle(TracingTriangles, TracingGeometry, triangleIdx);
	float3 v0 = posTri.Vertices[0];
	float3 edge1 = posTri.Vertices[1] - posTri.Vertices[0];
	float3 edge2 = posTri.Vertices[2] - posTri.Vertices[0];
	float3 normal = cross(edge1, edge2);

	///// Batch /////

	// Adapt height of intersection testing thread group
	uint intersectionTaskHeight = max(TraceTaskHeight, ceil_div(rayCount, TraceViewportDimension.x * 2) * 2);

	// Compute ray link index
	uint localRayLinkIdx = intersectionTaskHeight * (uint) p.Position.x + (uint) p.Position.y;
	if (localRayLinkIdx >= rayCount) 
		return;
	uint rayLinkIdx = rayLinkBaseIdx + localRayLinkIdx;

	///// Intersection testing /////

	// DEBUG: Count number of intersection tests
#ifdef RECORD_INTERSECTION_STATISTICS
	DebugUAV.IncrementCounter();
#endif
#ifdef COUNT_TESTS_PER_RAY
	InterlockedAdd(DebugUAV[RayLinkBuffer.Load(4 * rayLinkIdx)].RayLength, 1);
#endif

	// Get ray by ray link
	uint rayIdx;
	bool rayLinkLoaded;
	RayDesc ray = GetRay(rayLinkIdx, rayIdx, rayLinkLoaded);

	// Compute hit point
	float dist;
	float3 baryCoords;
	if (!IntersectRayWithTriangle(ray, dist, baryCoords, v0, edge1, edge2, normal, bitunpack3(cellID)))
		return;

	///// Hit point evaluation /////

	// DEBUG: Count number of intersections
#ifdef RECORD_INTERSECTION_STATISTICS
	TracedGeometryUAV.IncrementCounter();
#endif

	// Fetch additional triangle data
	rayIdx = GetRayLinkAfterRay(rayLinkIdx, rayIdx, rayLinkLoaded);
	TransformedTraceTriangle tri = ExtractTriangle(TracingTriangles, TracingGeometry, triangleIdx);

	// Evalute hit point & update ray hit record
	UpdateRay(rayIdx, ray, dist, tri, baryCoords, edge1, edge2);
}

// Note: PARTIAL_INTERSECTION_TESTING allows you to experiment with the shaders in this file
// in ways that break the proper detection of hit points, without completely breaking subsequent
// tracing passes. If partial tracing is enabled in the tweak bar, the tracing pipeline first
// executes the shaders with PARTIAL_INTERSECTION_TESTING defined and then redoes tracing without
// PARTIAL_INTERSECTION_TESTING defined to properly detect all hit points. This allows for normal
// ray termination and ensures sensible input in subsequent passes.

technique11 Tracing <
#ifndef PARTIAL_INTERSECTION_TESTING
	string PipelineStage = "TracingPipelineStage";
#else
	string PipelineStage = "PartialTracingPipelineStage";
#endif
	string RenderQueue = "DefaultRenderQueue";
	bool EnableTracing = true;
>
{
	pass <
		string IntersectCellRaysPass = "IntersectCellRays";
	>
	{
		SetVertexShader( CompileShader(vs_5_0, VSRenderToRayGrid()) );
		SetGeometryShader( ConstructGSWithSO(CompileShader(gs_5_0, GSRenderToRayGrid()), "0:Position1.xyz; 1:Normal1.xyz; 1:Texcoord4.xy") );
		SetPixelShader( CompileShader(ps_5_0, PSRenderToRayGrid()) ); 

#ifdef PARTIAL_INTERSECTION_TESTING
//		SetGeometryShader( NULL ); 
//		SetPixelShader( NULL ); 
#endif
	}
	pass IntersectCellRays < bool Normal = false;
			bool DynamicDispatch = true;
			#ifdef PARTIAL_INTERSECTION_TESTING
//				bool NullSkip = true;
			#endif
		>
	{
		SetVertexShader( CompileShader(vs_5_0, VSIntersectCellRays()) );
		SetGeometryShader( CompileShader(gs_5_0, GSIntersectCellRays()) );
		SetPixelShader( CompileShader(ps_5_0, PSIntersectCellRays()) );

#ifdef PARTIAL_INTERSECTION_TESTING
//		SetGeometryShader( NULL ); 
//		SetPixelShader( NULL ); 
#endif
	}
}

technique11 <
	string IncludeEffect = "#this";
	string IncludeTechnique = "Tracing";
	string IncludeDefinitions[] = { "#this", "PARTIAL_INTERSECTION_TESTING" };
	string IncludeHooks[] = #hookarray;
> { }

technique11 <
	string IncludeEffect = "Prototypes/SceneApprox.fx";
	string IncludeHooks[] = #hookarray;
> { }
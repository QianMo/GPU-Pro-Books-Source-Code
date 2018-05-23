/******************************************************/
/* Object-order Ray Tracing Demo (c) Tobias Zirr 2013 */
/******************************************************/

#define BE_RENDERABLE_INCLUDE_WORLD

#include <Engine/Perspective.fx>
#include <Engine/Renderable.fx>
#include <Pipelines/Tracing/VoxelRep.fx>
#include <Utility/Conservative.fx>
#include <Utility/Bits.fx>

// #define COUNT_TRIANGLES // Debug switch. Results may be displayed via color coding in RayGen.fx.

cbuffer SetupConstants
{
	#hookinsert SetupConstants
}

#hookincl "Hooks/Transform.fx"
#hookincl ...

struct Vertex
{
	float4 Position	: Position;
};

typedef Vertex TransformedVertex;

struct Voxel
{
	float4 Position : SV_Position;
	nointerpolation float4 AABB : TexCoord0;
	nointerpolation float2 Range : TexCoord1;
};

TransformedVertex VSDepth(Vertex v)
{
	#hookcall transformState = Transform(TransformHookPositionOnly(v.Position));

	TransformedVertex o;
	o.Position = float4( (GetWorldPosition(transformState).xyz - VoxelRep.Min) * VoxelRep.UnitScale, 1.0f );
	return o;
}

[MaxVertexCount(3)]
void GSTracing(triangle TransformedVertex tri[3], inout TriangleStream<Voxel> triStream)
{
	float2 halfPixelOffset = VoxelRep.RastVoxelWidth;
	float2 pixelOffset = 2.0f * halfPixelOffset;

	float4 v[3] = { tri[0].Position, tri[1].Position, tri[2].Position };

	// Dominant axis
	float3 n = cross(v[1].xyz - v[0].xyz, v[2].xyz - v[0].xyz);
	float3 an = abs(n);
	float man = max3(an);

	// Project triangle to dominant axis
	uint2 resolution = VoxelRep.Resolution.xy;
	float2 maxis = 1.0f;

	if (an.x >= man)
	{
		for (int i = 0; i < 3; ++i) v[i] = v[i].yzxw;
		resolution = VoxelRep.Resolution.yz;
		maxis.x = -1.0f;
	}
	else if (an.y >= man)
	{
		for (int i = 0; i < 3; ++i) v[i] = v[i].zxyw;
		resolution = VoxelRep.Resolution.zx;
		maxis.y = -1.0f;
	}

	// Transform projected triangle to normalized device coordinates
	for (int i = 0; i < 3; ++i)
		v[i].xy *= resolution * VoxelRep.RastVoxelWidth;
	for (int i = 0; i < 3; ++i)
		v[i].xy = v[i].xy * float2(2.0f, -2.0f) - float2(1.0f, -1.0f);

	// Compute conservative hull
	float4 p[3];
	float4 aabb;
	ConservativeTriBoundsOrtho(p, aabb, v, 1.005f * halfPixelOffset);

	Voxel o;
	// Transform AABB to normalized device coords & encode projection axis in sign (-> maxis)
	o.AABB = max( saturate(aabb * float2(0.5f, -0.5f).xyxy + 0.5f).xwzy * VoxelRep.RastResolution, 0.000001f ) * maxis.xyxy;
	float3 zv = float3(v[0].z, v[1].z, v[2].z);
	o.Range = float2(min3(zv), max3(zv));

	[unroll]
	for (int i = 0; i < 3; ++i)
	{
		o.Position = p[i];
		triStream.Append(o);
	}
}

void PSDepth(Voxel p)
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
	float2 minMaxSliceF = minMaxZ * resolutionZ;

	uint2 minMaxSlice = min( (uint2) minMaxSliceF, resolutionZ - 1 );
	gridIdx += minMaxSlice.x * sliceStep;

	for (uint slice = minMaxSlice.x; slice <= minMaxSlice.y; ++slice, gridIdx += sliceStep)
	{
#ifndef COUNT_TRIANGLES
		VoxelRepUAV[gridIdx] = 1;
#else
		InterlockedAdd(VoxelRepUAV[gridIdx], 1);
#endif
	}
}

technique11 Approximation <
	string PipelineStage = "VoxelRepPipelineStage";
	string RenderQueue = "DefaultRenderQueue";
>
{
	pass
	{
		SetVertexShader( CompileShader(vs_5_0, VSDepth()) );
		SetGeometryShader( CompileShader(gs_5_0, GSTracing()) );
		SetPixelShader( CompileShader(ps_5_0, PSDepth()) );
	}
}

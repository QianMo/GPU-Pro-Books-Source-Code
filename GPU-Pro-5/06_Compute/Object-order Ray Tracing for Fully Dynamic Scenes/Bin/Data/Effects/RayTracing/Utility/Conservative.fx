#include <Utility/Math.fx>

float ConservativeTriBoundsOrtho(out float4 cv[3], out float4 cb, in float4 v[3], in float2 offset)
{
	cb = float2(1.0f, -1.0f).xxyy;

	// Build bounding box from vertices
	[unroll]
	for (int i = 0; i < 3; ++i)
	{
		cb.xy = min(v[i].xy, cb.xy);
		cb.zw = max(v[i].xy, cb.zw);
	}

	// Enhance box by conservative pixel offset
	cb.xy -= offset;
	cb.zw += offset;

	float3 edgePlanes[3];

	// Construct edge planes
	[unroll]
	for (int i = 0; i < 3; ++i)
		edgePlanes[i] = cross(v[i].xyw, v[mod3(i + 1)].xyw);

	float winding = sign1( -dot(edgePlanes[0], v[2].xyw) - dot(edgePlanes[1], v[0].xyw) );

	[unroll]
	for (int i = 0; i < 3; ++i)
	{
		edgePlanes[i] *= winding;
		edgePlanes[i].z -= dot(offset, abs(edgePlanes[i].xy));
	}

	float3 vertexDirs[3];

	// Construct vertex projection lines
	[unroll]
	for (int i = 0; i < 3; ++i)
		vertexDirs[mod3(i + 1)] = cross(edgePlanes[i], edgePlanes[mod3(i + 1)]);

	float3 triNormalZ = cross(v[1].xyz - v[0].xyz, v[2].xyz - v[0].xyz);
	triNormalZ /= triNormalZ.z;
	float triOffsetZ = dot(triNormalZ, v[0].xyz);

	// Construct triangle
	[unroll]
	for (int i = 0; i < 3; ++i)
	{
		cv[i].xyw = vertexDirs[i].xyz / vertexDirs[i].z;
		cv[i].z = triOffsetZ - dot(cv[i].xy, triNormalZ.xy); // (...) / triNormalZ.z; ASSERT: triNormalZ.z == 1
	}

	return winding;
}

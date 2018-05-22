cbuffer cbRaycasting : register(b0) {
	row_major float4x4 g_matQuadToVoxel;
	row_major float4x4 g_matVoxelToScreen;
	float3 g_rayOrigin;
	float3 g_voxLightPos;
	uint2 g_stride;
	uint3 g_gridSize;
	bool g_showLines;
};

//==============================================================================================================================================================

Buffer<uint> g_bufVoxels : register(t0);

//==============================================================================================================================================================

struct VSInput_Quad {
	float4 pos			: POSITION;
	float2 texcoord		: TEXCOORD0;
};

struct PSInput_Quad {
    float4 pos			: SV_Position;
	float4 start		: RAY_START_POS;
};

struct PSOutput_Color {
	float4 color		: SV_Target;
};

//==============================================================================================================================================================

bool IsVoxelSet(int3 pos) {
	int p = pos.x * g_stride.x + pos.y * g_stride.y + (pos.z >> 5);
	int bit = pos.z & 31;
	uint voxels = g_bufVoxels[p];
	return (voxels & (1u << uint(bit))) != 0u;
}

float copysign(float x, float y) {
	return y < 0.0 ? -x : x;
}

float3 ShadeVoxel(int3 cell, float3 p, float3 n) {
	float3 l = g_voxLightPos;
	float dotNV = dot(n, normalize(l-p));

	float a = float(cell.x) / float(g_gridSize.x);

	const float3 color0 = float3(1.0, 0.0, 0.0);
	const float3 color1 = float3(1.0, 1.0, 0.0);
	const float3 color2 = float3(0.0, 1.0, 0.0);
	float3 color;
	if(a < 0.5)
		color = lerp(color0, color1, 2.0 * a);
	else
		color = lerp(color1, color2, 2.0 * a - 1.0);

	return dotNV * color;
}

bool CastRay(float3 o, float3 d, out float3 color, bool lines) {
	const float fltMax = 3.402823466e+38;
	const float eps = exp2(-50.0);

	color = 0.0;

	if(abs(d.x) < eps) d.x = copysign(eps, d.x);
	if(abs(d.y) < eps) d.y = copysign(eps, d.y);
	if(abs(d.z) < eps) d.z = copysign(eps, d.z);

	float3 deltaT = 1.0 / d;

	// determine intersection points with voxel grid box
	float3 tBox0 = (0.0 - o) * deltaT;
	float3 tBox1 = (float3(g_gridSize) - o) * deltaT;

	float3 tBoxMax = max(tBox0, tBox1);
	float3 tBoxMin = min(tBox0, tBox1);

	float tEnter = max(tBoxMin.x, max(tBoxMin.y, tBoxMin.z));
	float tExit  = min(tBoxMax.x, min(tBoxMax.y, tBoxMax.z));

	if(tEnter > tExit || tExit < 0.0)
		return false;

	deltaT = abs(deltaT);
	float t0 = max(tEnter - 0.5 * min(deltaT.x, min(deltaT.y, deltaT.z)), 0.0);		// start outside grid unless origin is inside

	float3 p = o + t0 * d;

	int3 cellStep = 1;
	if(d.x < 0.0) cellStep.x = -1;
	if(d.y < 0.0) cellStep.y = -1;
	if(d.z < 0.0) cellStep.z = -1;

	int3 cell;
	cell.x = int(floor(p.x));
	cell.y = int(floor(p.y));
	cell.z = int(floor(p.z));

	if(d.x < 0.0 && frac(p.x) == 0.0) cell.x--;
	if(d.y < 0.0 && frac(p.y) == 0.0) cell.y--;
	if(d.z < 0.0 && frac(p.z) == 0.0) cell.z--;

	float3 tMax = fltMax;
	if(d.x > 0.0) tMax.x = (float(cell.x + 1) - p.x) * deltaT.x;
	if(d.x < 0.0) tMax.x = (p.x - float(cell.x)) * deltaT.x;
	if(d.y > 0.0) tMax.y = (float(cell.y + 1) - p.y) * deltaT.y;
	if(d.y < 0.0) tMax.y = (p.y - float(cell.y)) * deltaT.y;
	if(d.z > 0.0) tMax.z = (float(cell.z + 1) - p.z) * deltaT.z;
	if(d.z < 0.0) tMax.z = (p.z - float(cell.z)) * deltaT.z;

	// traverse voxel grid until ray hits a voxel or grid is left
	int maxSteps = g_gridSize.x + g_gridSize.y + g_gridSize.z + 1;
	float t;
	float3 tMaxPrev;
	for(int i = 0; i < maxSteps; i++) {
		t = min(tMax.x, min(tMax.y, tMax.z));
		if(t0 + t >= tExit)
			return false;

		tMaxPrev = tMax;
		if(tMax.x <= t) { tMax.x += deltaT.x; cell.x += cellStep.x; }
		if(tMax.y <= t) { tMax.y += deltaT.y; cell.y += cellStep.y; }
		if(tMax.z <= t) { tMax.z += deltaT.z; cell.z += cellStep.z; }

		if(any(cell.xyz <= 0))
			continue;
		if(any(cell.xyz >= int3(g_gridSize)))
			continue;

		if(IsVoxelSet(cell))
			break;
	}

	// process hit point
	float3 n = 0.0;
	if(tMaxPrev.x <= t) n.x = d.x > 0.0 ? -1.0 : 1.0;
	if(tMaxPrev.y <= t) n.y = d.y > 0.0 ? -1.0 : 1.0;
	if(tMaxPrev.z <= t) n.z = d.z > 0.0 ? -1.0 : 1.0;
	n = normalize(n);

	p = o + (t0 + t) * d;

	color = ShadeVoxel(cell, p, n);

	if(lines) {
		float3 sP = mul(g_matVoxelToScreen, float4(p, 1.0)).xyw;
		sP.xy /= sP.z;

		// determine closest voxel edge
		float minDist = 10.0;

		if(tMaxPrev.x > t) {
			float dist = 0.5 - abs(0.5 - frac(abs(p.x)));
			float3 sPn = mul(g_matVoxelToScreen, float4(p.x + dist, p.y, p.z, 1.0)).xyw;
			minDist = min(minDist, distance(sP.xy, sPn.xy / sPn.z));
		}

		if(tMaxPrev.y > t) {
			float dist = 0.5 - abs(0.5 - frac(abs(p.y)));
			float3 sPn = mul(g_matVoxelToScreen, float4(p.x, p.y + dist, p.z, 1.0)).xyw;
			minDist = min(minDist, distance(sP.xy, sPn.xy / sPn.z));
		}

		if(tMaxPrev.z > t) {
			float dist = 0.5 - abs(0.5 - frac(abs(p.z)));
			float3 sPn = mul(g_matVoxelToScreen, float4(p.x, p.y, p.z + dist, 1.0)).xyw;
			minDist = min(minDist, distance(sP.xy, sPn.xy / sPn.z));
		}

		// blend in line if closest edge overlaps pixel
		if(minDist < 1.0) {
			float a = exp2(-2.0 * pow(minDist * 1.8, 4.0));
			const float3 lineColor = 0.2;
			color = lerp(color, lineColor, a);
		}
	}

	return true;
}

//--------------------------------------------------------------------------------------------------------------------------------------------------------------

PSInput_Quad VS_RenderVoxelizationRaycasting(VSInput_Quad input) {
    PSInput_Quad output;

    output.pos = input.pos;
    output.start = mul(g_matQuadToVoxel, float4(input.texcoord, 0.0, 1.0));

    return output;
}

PSOutput_Color PS_RenderVoxelizationRaycasting(PSInput_Quad input) {
	PSOutput_Color output;

	float3 o = g_rayOrigin;
	float3 d = normalize((input.start.xyz / input.start.w) - o);

	if(!CastRay(o, d, output.color.xyz, g_showLines))
		discard;

	output.color.w = 1.0;
	return output;
}

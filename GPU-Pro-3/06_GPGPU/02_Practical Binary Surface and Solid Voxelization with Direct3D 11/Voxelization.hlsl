cbuffer cbVoxelGrid : register(b0) {
	row_major float4x4 g_matModelToProj;
	row_major float4x4 g_matModelToVoxel;
	uint2 g_stride;
	uint3 g_gridSize;
};

cbuffer cbModelInput : register(b1) {
	uint g_numModelTriangles;
	uint g_vertexFloatStride;
};

//==============================================================================================================================================================

RWByteAddressBuffer g_rwbufVoxels : register(u1);

Buffer<float> g_bufVertices : register(t0);
Buffer<uint> g_bufIndices : register(t1);

//==============================================================================================================================================================

struct VSInput_Model {
	float4 pos			: POSITION;
	float3 normal		: NORMAL;
};

//--------------------------------------------------------------------------------------------------------------------------------------------------------------

struct PSOutput_Color {
	float4 color		: SV_Target;
};

//==============================================================================================================================================================

struct PSInput_Voxelize {
    float4 pos			: SV_Position;
    float4 gridPos		: POSITION_GRID;
};

PSInput_Voxelize VS_Voxelize(VSInput_Model input) {
    PSInput_Voxelize output;

	// transform to clip space
    output.pos = mul(g_matModelToProj, input.pos);

	// transform to voxel space
    output.gridPos = mul(g_matModelToVoxel, input.pos);

    return output;
}

PSOutput_Color PS_VoxelizeSurface(PSInput_Voxelize input) {
	PSOutput_Color output;

	// determine voxel
	float3 gridPos = input.gridPos.xyz / input.gridPos.w;
	int3 p = int3(gridPos);

	// set voxel
	g_rwbufVoxels.InterlockedOr(p.x * g_stride.x + p.y * g_stride.y + (p.z >> 5) * 4 , 1 << (p.z & 31));

	// kill fragment
	discard;

	output.color = 0.0;
	return output;
}

PSOutput_Color PS_VoxelizeSolid(PSInput_Voxelize input) {
	PSOutput_Color output;

	// determine first voxel
	float3 gridPos = input.gridPos.xyz / input.gridPos.w;
	int3 p = int3(gridPos.x, gridPos.y, gridPos.z + 0.5);

	// flip all voxels below
	if(p.z < int(g_gridSize.z)) {
		uint address = p.x * g_stride.x + p.y * g_stride.y + (p.z >> 5) * 4;

		g_rwbufVoxels.InterlockedXor(address, 0xffffffffu << (p.z & 31));
		for(p.z = (p.z | 31) + 1; p.z < int(g_gridSize.z); p.z += 32) {
			address += 4;
			g_rwbufVoxels.InterlockedXor(address, 0xffffffffu);
		}
	}

	// kill fragment
	discard;

	output.color = 0.0;
	return output;
}

//==============================================================================================================================================================

[numthreads(256, 1, 1)]
void CS_VoxelizeSolid(uint gtidx : SV_GroupIndex, uint3 gid : SV_GroupID) {
	const uint c_numthreads = 256;
	const uint c_groupCountX = 256;

	// determine triangle
	const uint tri = gtidx + gid.x * c_numthreads + gid.y * c_numthreads * c_groupCountX;

	if(tri >= g_numModelTriangles)
		return;

	// load triangle's vertices and order them ascending by index
	uint3 indices;
	indices.x = g_bufIndices[tri * 3];
	indices.y = g_bufIndices[tri * 3 + 1];
	indices.z = g_bufIndices[tri * 3 + 2];

	uint i0 = min(indices.x, indices.y);
	uint i1 = max(indices.x, indices.y);

	indices.x = min(i0, indices.z);
	i0        = max(i0, indices.z);
	indices.y = min(i1, i0);
	indices.z = max(i1, i0);

	float3 v0, v1, v2;
	v0.x = g_bufVertices[indices.x * g_vertexFloatStride];
	v0.y = g_bufVertices[indices.x * g_vertexFloatStride + 1];
	v0.z = g_bufVertices[indices.x * g_vertexFloatStride + 2];
	v1.x = g_bufVertices[indices.y * g_vertexFloatStride];
	v1.y = g_bufVertices[indices.y * g_vertexFloatStride + 1];
	v1.z = g_bufVertices[indices.y * g_vertexFloatStride + 2];
	v2.x = g_bufVertices[indices.z * g_vertexFloatStride];
	v2.y = g_bufVertices[indices.z * g_vertexFloatStride + 1];
	v2.z = g_bufVertices[indices.z * g_vertexFloatStride + 2];

	// transform vertices to voxel space
	float4 v;
	v = mul(g_matModelToVoxel, float4(v0, 1.0)); v0 = v.xyz / v.w;
	v = mul(g_matModelToVoxel, float4(v1, 1.0)); v1 = v.xyz / v.w;
	v = mul(g_matModelToVoxel, float4(v2, 1.0)); v2 = v.xyz / v.w;

	// determine bounding box in xz
	const float2 vMin = float2(min(v0.x, min(v1.x, v2.x)), min(v0.z, min(v1.z, v2.z)));
	const float2 vMax = float2(max(v0.x, max(v1.x, v2.x)), max(v0.z, max(v1.z, v2.z)));

	// derive bounding box of covered voxel columns
	const int2 voxMin = int2(max(0, int(floor(vMin.x + 0.4999))),
	                         max(0, int(floor(vMin.y + 0.4999))));
	const int2 voxMax = int2(min(int(g_gridSize.x), int(floor(vMax.x + 0.5))),
	                         min(int(g_gridSize.z), int(floor(vMax.y + 0.5))));

	// check if any voxel columns are covered at all
	if(voxMin.x >= voxMax.x || voxMin.y >= voxMax.y)
		return;

	// triangle setup
	const float3 e0 = v1-v0;
	const float3 e1 = v2-v1;
	const float3 e2 = v2-v0;
	const float3 n = cross(e0, e2);

	if(n.y == 0.0)
		return;

	// triangle's plane
	const float dTri = -dot(n, v0);

	// edge equations
	float2 ne0 = float2(-e0.z,  e0.x);
	float2 ne1 = float2(-e1.z,  e1.x);
	float2 ne2 = float2( e2.z, -e2.x);
	if(n.y > 0.0) {
		ne0 = -ne0;
		ne1 = -ne1;
		ne2 = -ne2;
	}

	const float de0 = -(ne0.x * v0.x + ne0.y * v0.z);
	const float de1 = -(ne1.x * v1.x + ne1.y * v1.z);
	const float de2 = -(ne2.x * v0.x + ne2.y * v0.z);

	// determine whether edge is left edge or top edge
	const float eps = 1.175494351e-38f;		// smallest normalized positive number

	float ce0 = 0.0;
	float ce1 = 0.0;
	float ce2 = 0.0;

	if(ne0.x > 0.0 || (ne0.x == 0.0 && ne0.y < 0.0))
		ce0 = eps;
	if(ne1.x > 0.0 || (ne1.x == 0.0 && ne1.y < 0.0))
		ce1 = eps;
	if(ne2.x > 0.0 || (ne2.x == 0.0 && ne2.y < 0.0))
		ce2 = eps;

	const float nyInv = 1.0 / n.y;

	// determine covered pixels/voxel columns
	for(int z = voxMin.y; z < voxMax.y; z++) {
		for(int x = voxMin.x; x < voxMax.x; x++) {
			// pixel center
			float2 p = float2(float(x) + 0.5, float(z) + 0.5);

			// test whether pixel is inside triangle
			// if it is exactly on an edge, the ce* term makes the expression positive if the edge is a left or top edge
			if((dot(ne0, p) + de0) + ce0 <= 0.0) continue;
			if((dot(ne1, p) + de1) + ce1 <= 0.0) continue;
			if((dot(ne2, p) + de2) + ce2 <= 0.0) continue;

			// project p onto plane along y axis (ray/plane intersection)
			const float py = -(p.x * n.x + p.y * n.z + dTri) * nyInv;

			int y = max(0, int(py + 0.5));
			if(int(g_gridSize.y) <= y)
				continue;

			// flip voxel's state
			uint address = uint(x) * g_stride.x + uint(y) * g_stride.y + (uint(z) >> 5) * 4;
			g_rwbufVoxels.InterlockedXor(address, 1u << (z & 31));
		}
	}
}

//--------------------------------------------------------------------------------------------------------------------------------------------------------------

[numthreads(256, 1, 1)]
void CS_VoxelizeSolid_Propagate(uint gtidx : SV_GroupIndex, uint3 gid : SV_GroupID) {
	const uint c_numthreads = 256;
	const uint c_groupCountX = 256;

	const uint section = gtidx + gid.x * c_numthreads + gid.y * c_numthreads * c_groupCountX;

	if(section >= (g_stride.y >> 2))
		return;

	uint address = section * 4;
	uint lastBlock = g_rwbufVoxels.Load(address);
	for(uint y = 1; y < g_gridSize.y; y++) {
		address += g_stride.y;

		uint currBlock = g_rwbufVoxels.Load(address);
		if(lastBlock != 0) {
			currBlock = currBlock ^ lastBlock;
			g_rwbufVoxels.Store(address, currBlock);
		}
		lastBlock = currBlock;
	}
}

//==============================================================================================================================================================

void Determine2dEdge(out float2 ne, out float de, float orientation, float edge_x, float edge_y, float vertex_x, float vertex_y) {
	ne = float2(-orientation * edge_y, orientation * edge_x);
	de = -(ne.x * vertex_x + ne.y * vertex_y);
	de += max(0.0, ne.x);
	de += max(0.0, ne.y);
}

[numthreads(256, 1, 1)]
void CS_VoxelizeSurfaceConservative(uint gtidx : SV_GroupIndex, uint3 gid : SV_GroupID) {
	const uint c_numthreads = 256;
	const uint c_groupCountX = 256;

	// determine triangle
	const uint tri = gtidx + gid.x * c_numthreads + gid.y * c_numthreads * c_groupCountX;

	if(tri >= g_numModelTriangles)
		return;

	// load triangle's vertices
	uint3 indices;
	indices.x = g_bufIndices[tri * 3];
	indices.y = g_bufIndices[tri * 3 + 1];
	indices.z = g_bufIndices[tri * 3 + 2];

	float3 v0, v1, v2;
	v0.x = g_bufVertices[indices.x * g_vertexFloatStride];
	v0.y = g_bufVertices[indices.x * g_vertexFloatStride + 1];
	v0.z = g_bufVertices[indices.x * g_vertexFloatStride + 2];
	v1.x = g_bufVertices[indices.y * g_vertexFloatStride];
	v1.y = g_bufVertices[indices.y * g_vertexFloatStride + 1];
	v1.z = g_bufVertices[indices.y * g_vertexFloatStride + 2];
	v2.x = g_bufVertices[indices.z * g_vertexFloatStride];
	v2.y = g_bufVertices[indices.z * g_vertexFloatStride + 1];
	v2.z = g_bufVertices[indices.z * g_vertexFloatStride + 2];

	// transform vertices to voxel space
	float4 v;
	v = mul(g_matModelToVoxel, float4(v0, 1.0)); v0 = v.xyz / v.w;
	v = mul(g_matModelToVoxel, float4(v1, 1.0)); v1 = v.xyz / v.w;
	v = mul(g_matModelToVoxel, float4(v2, 1.0)); v2 = v.xyz / v.w;

	// determine bounding box
	const float3 vMin = float3(min(v0.x, min(v1.x, v2.x)),
	                           min(v0.y, min(v1.y, v2.y)),
	                           min(v0.z, min(v1.z, v2.z)));
	const float3 vMax = float3(max(v0.x, max(v1.x, v2.x)),
	                           max(v0.y, max(v1.y, v2.y)),
	                           max(v0.z, max(v1.z, v2.z)));

	const float3 voxOrigMin = float3(floor(vMin.x),
	                                 floor(vMin.y),
	                                 floor(vMin.z));
	const float3 voxOrigMax = float3(floor(vMax.x + 1.0),
	                                 floor(vMax.y + 1.0),
	                                 floor(vMax.z + 1.0));

	const float3 voxOrigExtent = voxOrigMax - voxOrigMin;

	// determine bounding box clipped to voxel grid
	const float3 voxMin = float3(max(0.0, voxOrigMin.x),
	                             max(0.0, voxOrigMin.y),
	                             max(0.0, voxOrigMin.z));
	const float3 voxMax = float3(min(float(g_gridSize.x), voxOrigMax.x),
	                             min(float(g_gridSize.y), voxOrigMax.y),
	                             min(float(g_gridSize.z), voxOrigMax.z));

	const float3 voxExtent = voxMax - voxMin;

	// check if any voxels are covered at all
	if(any(voxExtent <= 0.0))
		return;

	// determine dimensions of unclipped extent
	const uint FLATDIM_X = 4;
	const uint FLATDIM_Y = 8;
	const uint FLATDIM_Z = 16;

	uint flatDimensions = 0;
	if(voxOrigExtent.x == 1.0) flatDimensions += 1 | FLATDIM_X;
	if(voxOrigExtent.y == 1.0) flatDimensions += 1 | FLATDIM_Y;
	if(voxOrigExtent.z == 1.0) flatDimensions += 1 | FLATDIM_Z;

	//---- 1D: set all voxels in bounding box ----
	if((flatDimensions & 3) >= 22) {
		uint address = uint(voxMin.x) * g_stride.x + uint(voxMin.y) * g_stride.y + (uint(voxMin.z) >> 5) * 4;

		// 1x1xN: set all voxels, up to 32 consecutive ones at a time
		if((flatDimensions & FLATDIM_Z) == 0) {
			const uint voxMax_z = uint(voxMin.z);

			uint voxels = (~0u) << (uint(voxMin.z) & 31);
			uint lastZ = (uint(voxMax_z) & (~31));

			for(uint z = uint(voxMin.z); z < lastZ; z += 32) {
				g_rwbufVoxels.InterlockedOr(address, voxels);
				address += 4;
				voxels = ~0;
			}

			uint restCount = uint(voxMax_z) & 31;
			if(restCount > 0) {
				voxels &= ~(0xffffffff << restCount);
				g_rwbufVoxels.InterlockedOr(address, voxels);
			}
		}

		// Nx1x1 or 1xNx1: set all voxels, one at a time
		else {
			const uint stride = (flatDimensions & FLATDIM_X) == 0 ? g_stride.x : g_stride.y;
			const uint count = uint(max(voxExtent.x, voxExtent.y));
			const uint voxels = 1u << (uint(voxMin.z) & 31);

			for(uint i = 0; i < count; i++) {
				g_rwbufVoxels.InterlockedOr(address, voxels);
				address += stride;
			}
		}
	}

	//---- 2D or 3D ----
	else {
		// triangle setup
		const float3 e0 = v1-v0;
		const float3 e1 = v2-v1;
		const float3 e2 = v0-v2;
		float3 n = cross(e2, e0);

		//---- 2D: test only for 2D triangle/voxel overlap ----
		if((flatDimensions & 3) == 11) {
			uint address0 = uint(voxMin.x) * g_stride.x + uint(voxMin.y) * g_stride.y + (uint(voxMin.z) >> 5) * 4;

			// NxMx1
			if(flatDimensions & FLATDIM_Z) {
				float2 ne0, ne1, ne2;
				float  de0, de1, de2;

				const float orientation = n.z < 0.0 ? -1.0 : 1.0;
				Determine2dEdge(ne0, de0, orientation, e0.x, e0.y, v0.x, v0.y);
				Determine2dEdge(ne1, de1, orientation, e1.x, e1.y, v1.x, v1.y);
				Determine2dEdge(ne2, de2, orientation, e2.x, e2.y, v2.x, v2.y);

				const uint voxels = 1 << (int(voxMin.z) & 31);

				float2 p;
				for(p.y = voxMin.y; p.y < voxMax.y; p.y++) {
					uint address = address0;
					for(p.x = voxMin.x; p.x < voxMax.x; p.x++) {
						if((dot(ne0, p) + de0 > 0.0) &&
						   (dot(ne1, p) + de1 > 0.0) &&
						   (dot(ne2, p) + de2 > 0.0))
						{
							g_rwbufVoxels.InterlockedOr(address, voxels);
						}
						address += g_stride.x;
					}
					address0 += g_stride.y;
				}
			}

			// 1xNxM or Nx1xM: inner loop along z such that updates to voxels stored in the same 32-bit buffer value result in only one buffer update
			else {
				float2 ne0, ne1, ne2;
				float  de0, de1, de2;

				uint stride;
				float2 p;
				float pxMax;

				if(flatDimensions & FLATDIM_X) {
					const float orientation = n.x < 0.0 ? -1.0 : 1.0;
					Determine2dEdge(ne0, de0, orientation, e0.y, e0.z, v0.y, v0.z);
					Determine2dEdge(ne1, de1, orientation, e1.y, e1.z, v1.y, v1.z);
					Determine2dEdge(ne2, de2, orientation, e2.y, e2.z, v2.y, v2.z);
					stride = g_stride.y;
					p.x = voxMin.y;
					pxMax = voxMax.y;
				} else {
					const float orientation = n.y > 0.0 ? -1.0 : 1.0;
					Determine2dEdge(ne0, de0, orientation, e0.x, e0.z, v0.x, v0.z);
					Determine2dEdge(ne1, de1, orientation, e1.x, e1.z, v1.x, v1.z);
					Determine2dEdge(ne2, de2, orientation, e2.x, e2.z, v2.x, v2.z);
					stride = g_stride.x;
					p.x = voxMin.x;
					pxMax = voxMax.x;
				}

				for(; p.x < pxMax; p.x++) {
					uint address = address0;
					uint voxels = 0;
					for(p.y = voxMin.z; p.y < voxMax.z; p.y++) {
						const uint z31 = uint(p.y) & 31;

						if((dot(ne0, p) + de0 > 0.0) &&
						   (dot(ne1, p) + de1 > 0.0) &&
						   (dot(ne2, p) + de2 > 0.0))
						{
							voxels |= 1 << z31;
						}

						if(z31 == 31) {
							if(voxels) {
								g_rwbufVoxels.InterlockedOr(address, voxels);
								voxels = 0;
							}
							address += 4;
						}
					}

					if((uint(voxMax.z) & 31) && voxels != 0)
						g_rwbufVoxels.InterlockedOr(address, voxels);

					address0 += stride;
				}
			}
		}

		//---- 3D ----
		else {
			// determine edge equations and offsets
			float2 ne0_xy, ne1_xy, ne2_xy;
			float de0_xy, de1_xy, de2_xy;
			const float orientation_xy = n.z < 0.0 ? -1.0 : 1.0;
			Determine2dEdge(ne0_xy, de0_xy, orientation_xy, e0.x, e0.y, v0.x, v0.y);
			Determine2dEdge(ne1_xy, de1_xy, orientation_xy, e1.x, e1.y, v1.x, v1.y);
			Determine2dEdge(ne2_xy, de2_xy, orientation_xy, e2.x, e2.y, v2.x, v2.y);

			float2 ne0_xz, ne1_xz, ne2_xz;
			float de0_xz, de1_xz, de2_xz;
			const float orientation_xz = n.y > 0.0 ? -1.0 : 1.0;
			Determine2dEdge(ne0_xz, de0_xz, orientation_xz, e0.x, e0.z, v0.x, v0.z);
			Determine2dEdge(ne1_xz, de1_xz, orientation_xz, e1.x, e1.z, v1.x, v1.z);
			Determine2dEdge(ne2_xz, de2_xz, orientation_xz, e2.x, e2.z, v2.x, v2.z);

			float2 ne0_yz, ne1_yz, ne2_yz;
			float de0_yz, de1_yz, de2_yz;
			const float orientation_yz = n.x < 0.0 ? -1.0 : 1.0;
			Determine2dEdge(ne0_yz, de0_yz, orientation_yz, e0.y, e0.z, v0.y, v0.z);
			Determine2dEdge(ne1_yz, de1_yz, orientation_yz, e1.y, e1.z, v1.y, v1.z);
			Determine2dEdge(ne2_yz, de2_yz, orientation_yz, e2.y, e2.z, v2.y, v2.z);

			const float maxComponentValue = max(abs(n.x), max(abs(n.y), abs(n.z)));

			// triangle aligns best to yz
			if(maxComponentValue == abs(n.x)) {
				// make normal point in +x direction
				if(n.x < 0.0) {
					n.x = -n.x;
					n.y = -n.y;
					n.z = -n.z;
				}

				// determine triangle plane equation and offset
				const float dTri = -dot(n, v0);

				float dTriProjMin = dTri;
				dTriProjMin += max(0.0, n.y);
				dTriProjMin += max(0.0, n.z);

				float dTriProjMax = dTri;
				dTriProjMax += min(0.0, n.y);
				dTriProjMax += min(0.0, n.z);

				const float nxInv = 1.0 / n.x;

				uint address0 = uint(voxMin.y) * g_stride.y;

				float3 p;
				for(p.y = voxMin.y; p.y < voxMax.y; p.y++) {
					for(p.z = voxMin.z; p.z < voxMax.z; p.z++) {
						if((ne0_yz.x * p.y + ne0_yz.y * p.z + de0_yz >= 0.0) &&
						   (ne1_yz.x * p.y + ne1_yz.y * p.z + de1_yz >= 0.0) &&
						   (ne2_yz.x * p.y + ne2_yz.y * p.z + de2_yz >= 0.0))
						{
							// determine x range: project adjusted p onto plane along x axis (ray/plane intersection)
							float x = -(p.y * n.y + p.z * n.z + dTriProjMin) * nxInv;
							float minX = floor(x);
							if(x == minX) minX--;
							minX = max(voxMin.x, minX);

							x = -(p.y * n.y + p.z * n.z + dTriProjMax) * nxInv + 1.0;
							float maxX = floor(x);
							if(x == maxX) maxX++;
							maxX = min(voxMax.x, maxX);

							// test voxels in x range
							uint address = address0 + uint(minX) * g_stride.x + (uint(p.z) >> 5) * 4;
							const uint voxels = 1 << (uint(p.z) & 31);

							for(p.x = minX; p.x < maxX; p.x++) {
								if((ne0_xy.x * p.x + ne0_xy.y * p.y + de0_xy >= 0.0) &&
								   (ne1_xy.x * p.x + ne1_xy.y * p.y + de1_xy >= 0.0) &&
								   (ne2_xy.x * p.x + ne2_xy.y * p.y + de2_xy >= 0.0) &&
								   (ne0_xz.x * p.x + ne0_xz.y * p.z + de0_xz >= 0.0) &&
								   (ne1_xz.x * p.x + ne1_xz.y * p.z + de1_xz >= 0.0) &&
								   (ne2_xz.x * p.x + ne2_xz.y * p.z + de2_xz >= 0.0))
								{
									g_rwbufVoxels.InterlockedOr(address, voxels);
								}
								address += g_stride.x;
							}
						}
					}
					address0 += g_stride.y;
				}
			}

			// triangle aligns best to xz
			else if(maxComponentValue == abs(n.y)) {
				// make normal point in +y direction
				if(n.y < 0.0) {
					n.x = -n.x;
					n.y = -n.y;
					n.z = -n.z;
				}

				// determine triangle plane equation and offset
				const float dTri = -dot(n, v0);

				float dTriProjMin = dTri;
				dTriProjMin += max(0.0, n.x);
				dTriProjMin += max(0.0, n.z);

				float dTriProjMax = dTri;
				dTriProjMax += min(0.0, n.x);
				dTriProjMax += min(0.0, n.z);

				const float nyInv = 1.0 / n.y;

				uint address0 = uint(voxMin.x) * g_stride.x;

				float3 p;
				for(p.x = voxMin.x; p.x < voxMax.x; p.x++) {
					for(p.z = voxMin.z; p.z < voxMax.z; p.z++) {
						if((ne0_xz.x * p.x + ne0_xz.y * p.z + de0_xz >= 0.0) &&
						   (ne1_xz.x * p.x + ne1_xz.y * p.z + de1_xz >= 0.0) &&
						   (ne2_xz.x * p.x + ne2_xz.y * p.z + de2_xz >= 0.0))
						{
							// determine y range: project adjusted p onto plane along y axis (ray/plane intersection)
							float y = -(p.x * n.x + p.z * n.z + dTriProjMin) * nyInv;
							float minY = floor(y);
							if(y == minY) minY--;
							minY = max(voxMin.y, minY);

							y = -(p.x * n.x + p.z * n.z + dTriProjMax) * nyInv + 1.0;
							float maxY = floor(y);
							if(y == maxY) maxY++;
							maxY = min(voxMax.y, maxY);

							// test voxels in y range
							uint address = address0 + uint(minY) * g_stride.y + (uint(p.z) >> 5) * 4;
							const uint voxels = 1 << (uint(p.z) & 31);

							for(p.y = minY; p.y < maxY; p.y++) {
								if((ne0_xy.x * p.x + ne0_xy.y * p.y + de0_xy >= 0.0) &&
								   (ne1_xy.x * p.x + ne1_xy.y * p.y + de1_xy >= 0.0) &&
								   (ne2_xy.x * p.x + ne2_xy.y * p.y + de2_xy >= 0.0) &&
								   (ne0_yz.x * p.y + ne0_yz.y * p.z + de0_yz >= 0.0) &&
								   (ne1_yz.x * p.y + ne1_yz.y * p.z + de1_yz >= 0.0) &&
								   (ne2_yz.x * p.y + ne2_yz.y * p.z + de2_yz >= 0.0))
								{
									g_rwbufVoxels.InterlockedOr(address, voxels);
								}
								address += g_stride.y;
							}
						}
					}
					address0 += g_stride.x;
				}
			}

			// triangle aligns best to xy
			else {
				// make normal point in +z direction
				if(n.z < 0.0) {
					n.x = -n.x;
					n.y = -n.y;
					n.z = -n.z;
				}

				// determine triangle plane equation and offset
				const float dTri = -dot(n, v0);

				float dTriProjMin = dTri;
				dTriProjMin += max(0.0, n.x);
				dTriProjMin += max(0.0, n.y);

				float dTriProjMax = dTri;
				dTriProjMax += min(0.0, n.x);
				dTriProjMax += min(0.0, n.y);

				const float nzInv = 1.0 / n.z;

				uint address0 = uint(voxMin.x) * g_stride.x + uint(voxMin.y) * g_stride.y;

				float3 p;
				for(p.y = voxMin.y; p.y < voxMax.y; p.y++) {
					uint address1 = address0;
					for(p.x = voxMin.x; p.x < voxMax.x; p.x++) {
						if((ne0_xy.x * p.x + ne0_xy.y * p.y + de0_xy >= 0.0) &&
						   (ne1_xy.x * p.x + ne1_xy.y * p.y + de1_xy >= 0.0) &&
						   (ne2_xy.x * p.x + ne2_xy.y * p.y + de2_xy >= 0.0))
						{
							// determine z range: project adjusted p onto plane along z axis (ray/plane intersection)
							float z = -(p.x * n.x + p.y * n.y + dTriProjMin) * nzInv;
							float minZ = floor(z);
							if(z == minZ) minZ--;
							minZ = max(voxMin.z, minZ);

							z = -(p.x * n.x + p.y * n.y + dTriProjMax) * nzInv + 1.0;
							float maxZ = floor(z);
							if(z == maxZ) maxZ++;
							maxZ = min(voxMax.z, maxZ);

							// test voxels in z range
							uint address = address1 + (uint(minZ) >> 5) * 4;
							uint voxels = 0;

							for(p.z = minZ; p.z < maxZ; p.z++) {
								const uint z31 = uint(p.z) & 31;

								if((ne0_xz.x * p.x + ne0_xz.y * p.z + de0_xz >= 0.0) &&
								   (ne1_xz.x * p.x + ne1_xz.y * p.z + de1_xz >= 0.0) &&
								   (ne2_xz.x * p.x + ne2_xz.y * p.z + de2_xz >= 0.0) &&
								   (ne0_yz.x * p.y + ne0_yz.y * p.z + de0_yz >= 0.0) &&
								   (ne1_yz.x * p.y + ne1_yz.y * p.z + de1_yz >= 0.0) &&
								   (ne2_yz.x * p.y + ne2_yz.y * p.z + de2_yz >= 0.0))
								{
									voxels |= 1 << z31;
								}

								if(z31 == 31) {
									if(voxels) {
										g_rwbufVoxels.InterlockedOr(address, voxels);
										voxels = 0;
									}
									address += 4;
								}
							}

							if(voxels != 0)
								g_rwbufVoxels.InterlockedOr(address, voxels);
						}
						address1 += g_stride.x;
					}
					address0 += g_stride.y;
				}
			}
		}
	}
}


shared Texture1DArray <float4> occluderSphereSetArray;

shared Texture1DArray <float4> nodeTableArray;
shared Texture2DArray <float3> triangleTableArray;

void intersectTriangle(
	float3 rayOrigin,
	float3 rayDir,
	float3 ivm0,
	float3 ivm1,
	float3 ivm2,
	inout float bestDepth,
	out bool hit,
	out float3 barycentricWeights
	)
	{
		float3 planePos = ivm0 + ivm1 + ivm2;
		planePos /= dot(planePos, planePos);
		
		rayOrigin -= planePos<0;
		
		hit = false;
		barycentricWeights = 0;
	
		float hitDepth = (dot(planePos, planePos) - dot(rayOrigin, planePos)) / dot(rayDir, planePos);

		if(hitDepth > 0.1 && hitDepth < bestDepth)
		{
			float3 hitPoint = rayOrigin + (rayDir * hitDepth);

			barycentricWeights.x = dot(ivm0, hitPoint);
			barycentricWeights.y = dot(ivm1, hitPoint);
			barycentricWeights.z = dot(ivm2, hitPoint);
			if(all(barycentricWeights > -0.001))
			{
				bestDepth = hitDepth;
				hit = true;
			}
		}
	}

void processTriangle(float3 rayOrigin, float3 rayDir, uint meshIndex, uint entityIndex, uint i, inout float bestDepth, inout float3 bestBarycentric, inout uint3 bestTriangleId)
{
	//uint2 si = uint2(i >> 4, (i & 15) << 2);
	//float3 ivm0 = triangleTableArray.Load(uint4(si.x, si.y + 0, meshIndex, 0));
	//float3 ivm1 = triangleTableArray.Load(uint4(si.x, si.y + 1, meshIndex, 0));
	//float3 ivm2 = triangleTableArray.Load(uint4(si.x, si.y + 2, meshIndex, 0));
	float3 ivm0 = triangleTableArray.Load(uint4(i , 0, meshIndex, 0));
	float3 ivm1 = triangleTableArray.Load(uint4(i , 1, meshIndex, 0));
	float3 ivm2 = triangleTableArray.Load(uint4(i , 2, meshIndex, 0));

	bool hit;
	float3 barycentricWeights;

	intersectTriangle( rayOrigin, rayDir, ivm0, ivm1, ivm2, bestDepth, hit,	barycentricWeights);

	if(hit)
	{
		bestBarycentric = barycentricWeights;
		bestTriangleId = uint3(i, meshIndex, entityIndex);
	}
}

struct Entity
{
	float4x4	modelMatrix;
	float4x4	modelMatrixInverse;
	float4		diffuse;
	float4		specular;
	uint		meshIndex;
};

shared cbuffer entityBuffer
{
	uint4 nEntities;
	Entity entities[2];
}

float trace(float3 origin, float3 direction)
{
	float bestDepth = 1000000.0;	// max distance
	float3 bestBarycentric = 1;
	uint3 bestTriangleId = 0;

	for(int entityIndex=0; entityIndex<2; entityIndex++)
	{
		//compute eye rays from world coords of full screen quad
		float3 rayOrigin = origin;
		float3 rayDir = normalize(direction);
		rayOrigin = mul(float4(rayOrigin, 1), entities[entityIndex].modelMatrixInverse).xyz;
		rayDir = mul(float4(rayDir,0), entities[entityIndex].modelMatrixInverse).xyz;
		int meshIndex = entities[entityIndex].meshIndex;
		// dir is not normalized!
		
		float3 invRayDir = float3(1, 1, 1) / rayDir;

		float2 raySegment = float2(0, bestDepth); // tMin, tMax		

		float3 rootCellMinDepths = (float3(0, 0, 0) - rayOrigin) * invRayDir;
		float3 rootCellMaxDepths = (float3(1, 1, 1) - rayOrigin) * invRayDir;
		float3 rootCellEntryDepths = (rayDir > 0)?rootCellMinDepths:rootCellMaxDepths;
		float3 rootCellExitDepths = (rayDir > 0)?rootCellMaxDepths:rootCellMinDepths;
		raySegment.x = max(raySegment.x, max(rootCellEntryDepths.x, max(rootCellEntryDepths.y, rootCellEntryDepths.z)));
		raySegment.y = min(raySegment.y, min(rootCellExitDepths.x, min(rootCellExitDepths.y, rootCellExitDepths.z)));

		if(raySegment.x >= raySegment.y)
			continue;

		float3 traversalStack[24];
		
		uint tNode = 0;					// node 0
		uint stackPointer = 1;			// empty stack
		traversalStack[0] = float3(raySegment, asfloat(tNode));

		while(stackPointer > 0)
		{
			bool jumper = false;
			stackPointer--;
			float3 nextNode = traversalStack[stackPointer];
			raySegment = nextNode.xy;
			raySegment.y = min(bestDepth, raySegment.y);
			if(raySegment.x > raySegment.y)
				continue;
			tNode = asuint(nextNode.z);

			float4 rawNode = nodeTableArray.Load(uint3(tNode, meshIndex, 0));
			if(rawNode.w > 0)
			{
				// get address of children
				uint nearChild = (tNode << 1) + 1;		// index of left child
				uint farChild = nearChild;
				float3 axis = step(rawNode.xyz, 0);
				float4 bounds = abs(rawNode);

				//compute intersection with cutting planes
				float4 rooxx, radxx;
				rooxx = dot(rayOrigin, axis).xxxx;
				radxx = dot(invRayDir, axis).xxxx;
				float4 cellBoundDepths;
				cellBoundDepths = (bounds - rooxx) * radxx;

				//compute ray segments within child cells
		
				if(radxx.x > 0.0)
				{
					farChild += 1;	//the right child is the far child
				}
				else
				{
					cellBoundDepths.xyzw = cellBoundDepths.wzyx;
					nearChild += 1;	//the right child is the near child
				}

				float4 childSegments;
				childSegments.xz = max(raySegment.xx, cellBoundDepths.xz);
				childSegments.yw = min(raySegment.yy, cellBoundDepths.yw);

				//find next to traverse
//				if(childSegments.x < childSegments.y)
					traversalStack[stackPointer++] = float3(childSegments.xy, asfloat(nearChild));
//				if(childSegments.z < childSegments.w)
					traversalStack[stackPointer++] = float3(childSegments.zw, asfloat(farChild));					
			}
			else
			{
				//compute texture address where triangle list starts and ends
				//uint triangleStart = rawNode.x * 8192;
				//triangleStart += (uint)(rawNode.z * 8192) * 8192;
				//uint cellFaceCount = rawNode.y * 8192;
				//uint triangleEnd = triangleStart + cellFaceCount;
				uint triangleStart = rawNode.x * 8192;
				uint triangleEnd = rawNode.y * 8192;
				uint i = triangleStart;
				while(i < triangleEnd)
				{
					processTriangle(rayOrigin, rayDir, meshIndex, entityIndex, i,
						bestDepth, bestBarycentric, bestTriangleId);
					i++;
				}
			}
		}
	}
	return bestDepth;
}

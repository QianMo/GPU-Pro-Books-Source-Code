#include "enginePool.fx"

float4 psPTT(QuadOutput input) : SV_TARGET
{
	int meshIndex = 0;
	
	float3 rayOrigin = eyePosition;
	float3 rayDir = normalize(input.viewDir);
	rayOrigin = mul(float4(rayOrigin, 1), entities[meshIndex].modelMatrixInverse).xyz;
	rayDir = mul(float4(rayDir,0), entities[meshIndex].modelMatrixInverse).xyz;

//	return abs(rayDir.xyzz);

	float bestDepth = 1000000.0;
	float3 bestBarycentric = 1;
	uint3 bestTriangleId = 0;

	for(int i=0; i<512; i++)
		processTriangle( rayOrigin, rayDir, meshIndex, 0, i, bestDepth, bestBarycentric, bestTriangleId);
	
	return float4(bestBarycentric, 1.0);
}

Texture2D rayScramblerMap;

float4 psRaycasting(QuadOutput input) : SV_TARGET
{
	float bestDepth = 1000000.0;	// max distance
	float3 bestBarycentric = 1;
	uint3 bestTriangleId = 0;

	float3 scramblerNormal = rayScramblerMap.Sample(linearSampler, input.tex).xyz * 2 - 1;

	[loop]for(int entityIndex=0; entityIndex<2; entityIndex++)
	{
		//compute eye rays from world coords of full screen quad
		float3 rayOrigin = eyePosition;
		float3 rayDir = normalize(input.viewDir);
//		rayDir = refract(rayDir, scramblerNormal, 0.99);

		rayOrigin = mul(float4(rayOrigin, 1), entities[entityIndex].modelMatrixInverse).xyz;
		rayDir = mul(float4(rayDir,0), entities[entityIndex].modelMatrixInverse).xyz;
		int meshIndex = entities[entityIndex].meshIndex;
		// dir is not normalized!
		
		float3 invRayDir = float3(1, 1, 1) / rayDir;

		float2 raySegment = float2(0.0, bestDepth); // tMin, tMax		

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

		[loop]while(stackPointer > 0)
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
//	float3 packedNormals = triangleTableArray.Load(int4(bestTriangleId.x / 16, 
//		(bestTriangleId.x % 16) * 4 + 3, bestTriangleId.y, 0));
	float3 packedNormals = triangleTableArray.Load(int4(bestTriangleId.x, 
		3, bestTriangleId.y, 0));
	uint3 normalFields = asuint(packedNormals);
	int3 nx = int3(normalFields & 255);
	normalFields >>= 8;
	int3 ny = int3(normalFields & 255);
	normalFields >>= 8;
	int3 nz = int3(normalFields & 255);
	float3x3 nn = float3x3(nx, ny, nz);
	float3 smoothNormal = mul(nn, bestBarycentric) - float3(128, 128, 128);
	smoothNormal /= 256;
	smoothNormal = mul(float4(smoothNormal, 0), entities[bestTriangleId.z].modelMatrix);
	return abs(smoothNormal.xyzz);
//	return float4(bestBarycentric.xyz, 0);
}

technique10 raycasting
{
    pass P0
    {
        SetVertexShader ( CompileShader( vs_4_0, vsQuad() ) );
        SetGeometryShader( NULL );
		SetRasterizerState( defaultRasterizer );
        SetPixelShader( CompileShader( ps_4_0, psRaycasting() ) );
//	      SetPixelShader( CompileShader( ps_4_0, psPTT() ) );
		SetDepthStencilState( noDepthTestCompositor, 0);
		SetBlendState(defaultBlender, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xffffffff);
    }
}
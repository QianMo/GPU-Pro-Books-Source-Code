// ================================================================================ //
// Copyright (c) 2011 Arturo Garcia, Francisco Avila, Sergio Murguia and Leo Reyes	//
//																					//
// Permission is hereby granted, free of charge, to any person obtaining a copy of	//
// this software and associated documentation files (the "Software"), to deal in	//
// the Software without restriction, including without limitation the rights to		//
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies	//
// of the Software, and to permit persons to whom the Software is furnished to do	//
// so, subject to the following conditions:											//
//																					//
// The above copyright notice and this permission notice shall be included in all	//
// copies or substantial portions of the Software.									//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR		//
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,			//
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE		//
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER			//
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,	//
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE	//
// SOFTWARE.																		//
// ================================================================================ //

//--------------------------------------------------------------------------------------------------------------------
// COLOR STAGE
//--------------------------------------------------------------------------------------------------------------------
[numthreads(BLOCK_SIZE_X, BLOCK_SIZE_Y, 1)]
void CSComputeColor(uint3 DTid : SV_DispatchThreadID, uint GIndex : SV_GroupIndex)
{
	
	//return;
	// Calculate 1D index of the current thread
	const unsigned int index = DTid.y * N + DTid.x;

	// Get the triangle ID stored in the current pixel
	const int iTriangleId = g_uIntersections[index].iTriangleId;	

	// If Environment Mapping has already been applied to this ray, return.
	if(iTriangleId < g_iEnvMappingFlag) 
	//if(iTriangleId < -1) 
	{
		if(iTriangleId > (-2)) g_uResultTexture[DTid.xy] = 1.f;
		return;
	}

	// Init variables
	const int currentMaterial = g_sMaterials[iTriangleId].iMaterialId;
	float4 vfFinalColor = float4(0.f,0.f,0.f,1.f);
	float3 vfDiffuse = float3(0.f,0.f,0.f);
	float3 vfSpecular = float3(0.f,0.f,0.f);
	float3 vfFactor = g_uRays[index].vfReflectiveFactor;

	float3 vfNormal = 1.f;
	float3 vfNormalNM = 1.f;

	// If ray hits something: Apply iMaterial.
	// Otherwise: Apply environment texture and kill the ray (because the
	// intersection is null).
	if (iTriangleId >= 0)
	{
		float2 vfUV = computeUVs(iTriangleId, g_uIntersections[index]);
		// Shading (Phong Shading or Flat Shading)
		// Generate normal
		const unsigned int offset = iTriangleId * 3;
		Vertex A = g_sVertices[g_sIndices[offset]];
		Vertex B = g_sVertices[g_sIndices[offset+1]];
		Vertex C = g_sVertices[g_sIndices[offset+2]];
		float3 Edge1 = B.vfPosition - A.vfPosition;
		float3 Edge2 = C.vfPosition - A.vfPosition;

		if(g_bIsPhongShadingOn)
		{
			// vfNormal using phong interpolation (per pixel)
			vfNormal = normalize(Vec3BaryCentric(
				A.vfNormal, B.vfNormal, C.vfNormal, 
				float2(g_uIntersections[index].fU,g_uIntersections[index].fV)));
		}
		else
		{
			// vfNormal using flat interpolation (per triangle)
			vfNormal = normalize(cross(Edge1, Edge2));
		}

		// Calculate a normal vector using vfNormal Mapping
		if(g_bIsNormalMapspingOn)
		{
			// Read values from the normal-map texture 
			// normal, tangent and bitangent
			float3 tx_g_sNormalMaps = g_sNormalMaps.SampleLevel(
				g_ssSampler, float3(vfUV,currentMaterial),0).xyz;
			if(length(tx_g_sNormalMaps) < 0.1)
			{
				tx_g_sNormalMaps = float3(0.5f, 0.5f, 1.0f);
			}

			// Center the texture
			tx_g_sNormalMaps = 2 * tx_g_sNormalMaps-1;

			float2 Edge1UV = g_sVertices[g_sIndices[offset+1]].vfUvs - g_sVertices[g_sIndices[offset]].vfUvs;
			float2 Edge2UV = g_sVertices[g_sIndices[offset+2]].vfUvs - g_sVertices[g_sIndices[offset]].vfUvs;
			const float cp = Edge1UV.y*Edge2UV.x - Edge1UV.x*Edge2UV.y;

			if(cp != 0.f)
			{
				float mult = 1.f/cp;
				float3 tangent, bitangent;
				tangent = (Edge2*Edge1UV.y - Edge1*Edge2UV.y)*mult;
				bitangent = (Edge2*Edge1UV.x - Edge1*Edge2UV.x)*mult;
				tangent -= vfNormal*dot(tangent, vfNormal);
				tangent = normalize(tangent);
				bitangent -= vfNormal*dot(bitangent, vfNormal);
				bitangent -= tangent*dot(bitangent, tangent);
				bitangent = normalize(bitangent);
				vfNormalNM = tx_g_sNormalMaps.z*vfNormal +
							tx_g_sNormalMaps.x*tangent -
							tx_g_sNormalMaps.y*bitangent; 
			}
			else
			{
				vfNormalNM = vfNormal;
			}
		}
		else
		{
			vfNormalNM = vfNormal;
		}
		// END IF vfNormal Mapping
		
		// Point normals to face ray source
		if (dot(vfNormalNM,g_uRays[index].vfDirection) > 0.f)
		{
			vfNormalNM = -vfNormalNM;
		}
		
		// Generate hit point and light dir
		float3 vfHitPoint = g_uRays[index].vfOrigin + 
			g_uIntersections[index].fT * g_uRays[index].vfDirection;
		float3 vfLightDir = vfHitPoint - g_vfLightPosition;
		const float fLightDistance = length(vfLightDir);
		vfLightDir /= fLightDistance;
		
		// Ray reflection
		const float3 vfNewDirection = normalize(
			reflect(g_uRays[index].vfDirection,vfNormalNM));

		// Check if light source and ray origin are in the same siIde
		const float lightDirSign = dot(vfNormal,vfLightDir);
		const float rayDirSign = dot(vfNormal,g_uRays[index].vfDirection);
		if ((lightDirSign*rayDirSign)>0.f)
		{
			Ray r = g_uRays[index];
			r.vfOrigin = vfHitPoint;
			r.vfDirection = -vfLightDir.xyz;
			r.iTriangleId = iTriangleId;
			r.fMaxT = fLightDistance;

			// Cast shadows
			if(g_bIsShadowOn)
			{
				int tr = 0;
				if(g_iAccelerationStructure == 0) tr = BVH_IntersectP(r).iTriangleId;
				//else if(g_iAccelerationStructure == 9) tr = LBVH_IntersectP(r, g_uIntersections[index].iRoot).iTriangleId;
				if(tr < 0)
				{
					// Specular light term
					const float shininess = -dot(vfLightDir, vfNewDirection);
					vfSpecular = pow(max (0.f, shininess), 49);
					// Diffuse light term
					vfDiffuse = max(0,-dot(vfNormalNM, vfLightDir.xyz));
				}
			}
			else
			{
				// Specular light term
				vfSpecular = pow(max (0.f, -dot(vfLightDir.xyz, vfNewDirection)), 49);
				// Diffuse light term
				vfDiffuse = max(0,-dot(vfNormalNM, vfLightDir.xyz));
			}
		}
		// END IF	
		
		// Apply Lambert iMaterial
		const float fAmbientLight = 0.2f;
		const float3 tx_TextureColor = g_sTextures.SampleLevel(g_ssSampler, float3(vfUV,currentMaterial),0).xyz;
		//const float3 tx_TextureColor = 1.f;
		const float3 tx_SpecularColor = g_bIsGlossMappingOn*g_sSpecularMaps.SampleLevel(g_ssSampler, float3(vfUV,currentMaterial),0).xyz;

		// Calculate color for the current pixel
		const float3 vfDiffuseColor = (fAmbientLight + vfDiffuse) * tx_TextureColor;
		const float3 vfSpecularColor = vfSpecular * tx_SpecularColor;
		vfFinalColor = float4(vfDiffuseColor + vfSpecularColor,1.f);		
		vfFinalColor *= float4(vfFactor,1.f);
		
		// Ray bounce
		g_uRays[index].vfReflectiveFactor *= tx_SpecularColor;
		g_uRays[index].vfOrigin = vfHitPoint;
		g_uRays[index].vfDirection = vfNewDirection;				// Ray Tracing
		g_uRays[index].iTriangleId = iTriangleId;		
	}
	else
	{
		const float4 tx_Environment = g_sEnvironmentTx.SampleLevel(
				g_ssSampler, g_uRays[index].vfDirection.xyz,0);

		//Environment mapping
		vfFinalColor.xyz = vfFactor.xyz * tx_Environment.xyz;	
		// This indicates that the Environment Mapping has
		// been applied to the current pixel.
		g_uRays[index].iTriangleId = -2;
		//g_uIntersections[index].iTriangleId = -2;
		
	}
	// END IF
	
	// Apply color to texture
	//vfFinalColor.x += (float)(g_uIntersections[index].iVisitedNodes&65535)*0.0001;
	//vfFinalColor.y += (float)(g_uIntersections[index].iVisitedNodes>>16)*0.01;
	//vfFinalColor.z = 0;//(float)(g_uIntersections[index].iVisitedNodes>>16)*0.00001;
	g_uAccumulation[index] += vfFinalColor;
	g_uResultTexture[DTid.xy] = g_uAccumulation[index];
	//float grayScale = (vfNormalNM.x+vfNormalNM.y+vfNormalNM.z)/3;
	//g_uResultTexture[DTid.xy] = float4(grayScale,grayScale,grayScale,1.f);
	//g_uResultTexture[DTid.xy] = float4(g_uRays[index].vfDirection,1.f);
	//g_uResultTexture[DTid.xy] = float4(iTriangleId*0.000008,iTriangleId*0.000008,iTriangleId*0.000008,1);
}
RWStructuredBuffer<uint3> fragmentLinkRwBuffer;
RWByteAddressBuffer startOffsetRwBuffer;
uint materialId = 1;


[earlydepthstencil]
void psStoreFragments( VsosTrafo input, bool frontFace : SV_IsFrontFace, out float4 gtau : SV_Target0, out float4 irradiance : SV_Target1 )
{
	float dist = length(eyePos.xyz - input.worldPos.xyz);

	float opaqueDist = opaqueTexture.Load(uint3(input.pos.xy, 0)).w; 
	if(dist < opaqueDist)
	{
		// Increment and get current pixel count.
		uint uPixelCount = fragmentLinkRwBuffer.IncrementCounter();

		// Read and update Start Offset Buffer.
		uint uIndex = (uint)input.pos.y * frameDimensions.x + (uint)input.pos.x;
		uint uStartOffsetAddress = 4 * uIndex;
		uint uOldStartOffset;
		startOffsetRwBuffer.InterlockedExchange( 
			uStartOffsetAddress, uPixelCount, uOldStartOffset );

		// Create fragment data.    
		uint3 element;
		// encode material id and linked list pointer
		element.x = (materialId << 25) | (frontFace?0:0x80000000) | uOldStartOffset;
		// encode fragment distance
		element.y = asuint(dist);

		float3 viewDir = normalize(input.worldPos - eyePos);
		float3 reflectionDir = reflect(viewDir, input.normal);
		float3 shade = saturate(0.15 * envTexture.Sample(linearSampler, reflectionDir));
		shade *= surfaceReflectionWeight;
		/* for style transfer shading
		float3 cNormal = mul(float4(input.normal, 0), viewMatrix);
		float3 shade = kdTexture.Sample(linearSampler, normalize(cNormal.xyz).xy * 0.5 + 0.5) * 0.2;*/
		shade *= 255.9;
		if(!frontFace)
			shade = 0;

		// encode surface reflection
		element.z = (uint(shade.x) << 16) + (uint(shade.y) << 8) + uint(shade.z);

		// Store fragment link.
		fragmentLinkRwBuffer[uPixelCount] = element;
	}
	// blend-add optical properties to render target to get values at near plane
	gtau = - transparentMaterials[materialId].gtau * (frontFace?1:-1);
	irradiance = - transparentMaterials[materialId].lighting * (frontFace?1:-1);
}
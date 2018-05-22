StructuredBuffer<uint3> fragmentLinkBuffer;
Buffer<uint> startOffsetBuffer;
Texture2D nearPlaneTexture;
Texture2D nearPlaneIrradianceTexture;

float4 psSortAndRender( VsosQuad input ) : SV_Target0
{
	uint pixelIndex = (uint)input.pos.y * frameDimensions.x + (uint)input.pos.x;
    uint nextFragmentIndex = startOffsetBuffer[pixelIndex];

	// Read and insert linked list data to the temporary buffer.
    uint3 s[64];
    uint n = 0;
    while ( nextFragmentIndex != 0x01FFFFFF && n<64)
	{
		uint3 fragment = fragmentLinkBuffer[nextFragmentIndex];
		nextFragmentIndex = fragment & 0x01ffffff;
		fragment.x >>= 25;
  		int j = n;
		if(j > 0)
			[loop]while(asfloat(fragment.y) < asfloat(s[ j-1 ].y) && j>0)
			{
				s[j]=s[j-1];
				j--;
			}
		s[j] = fragment;
		n++;
    }

	// Too many fragments to fit in array
	if(nextFragmentIndex != 0x01FFFFFF)
		return float4(1, 0, 0, 1);

	// read starting values and opaque surface data
	int3 pixelPos = int3(input.pos.x, input.pos.y, 0);
	float4 opaque_L_s = opaqueTexture.Load(pixelPos);
    float4 g_tau = nearPlaneTexture.Load(pixelPos);
	float4 lighting = nearPlaneIrradianceTexture.Load(pixelPos);

	// initialize running varibales
	float3 L = float3(0.0,0.0,0.0);
	float segLength;
	float T = 1.0;

	float segNear = 0;
	// process segment
	for (int x = 0; x < n && T > 0.0001 && segNear < opaque_L_s.w; x++)
	{
		float segFar = asfloat(s[ x ].y);
		// clip to opaque distance
		segFar = min(segFar, opaque_L_s.w);
		segLength = segFar - segNear;

		// evaluate contribution	
		if (g_tau.w > 0.0001)
		{
			float transparency = exp( -g_tau.w * segLength);
			L += lighting * T * saturate(g_tau.xyz ) *  (1 - transparency) /  (g_tau.w);
			T *= transparency;
		}

		// add surface reflection
		float3 surfaceShade;
		uint sxz = s[x].z;
		surfaceShade.z = sxz & 0xff;
		sxz >>= 8;
		surfaceShade.y = sxz & 0xff;
		sxz >>= 8;
		surfaceShade.x = sxz & 0xff;
		surfaceShade /= 255.9;
		L += T * surfaceShade;

		// set up next
		segNear = segFar;

		// update running optical properties (incl. lighting)
		uint materialId = s[ x ].x;
		float4 materialGtau = transparentMaterials[materialId & 0x3f].gtau;
		if( materialId & 0x40 )
			materialGtau *= -1;
		g_tau += materialGtau;
		if( materialId & 0x40 )
			lighting -= transparentMaterials[materialId & 0x3f].lighting;
		else
			lighting += transparentMaterials[materialId & 0x3f].lighting;
	}

	// handle final segment
	segLength = opaque_L_s.w - segNear;
	if (segLength > 0 && g_tau.w > 0.0001)
	{
		float transparency = exp( -g_tau.w * segLength);
		L += lighting * T * saturate(g_tau.xyz ) *  (1 - transparency) /  (g_tau.w);
		T *= transparency;
	}

	// add attenuated opaque surface color
	L += T * opaque_L_s.xyz * (0.5 + lighting*0.5);

    return float4(L,1.0);
	
}

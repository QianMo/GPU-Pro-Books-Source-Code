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

//-------------------------------------------------------------------------------------------
// PRIMARY RAYS STAGE
//-------------------------------------------------------------------------------------------
// Primary rays.

[numthreads(BLOCK_SIZE_X, BLOCK_SIZE_Y, 1)]
void CSGeneratePrimaryRays(uint3 DTid : SV_DispatchThreadID, uint GIndex : SV_GroupIndex )
{
#ifdef GLOBAL_ILLUM
	float2 indexasfloat = float2(DTid.x,DTid.y)/float(N);
	float2 jitterOffset = indexasfloat+float2(g_fRandX,g_fRandY);
	float2 tx_Random = g_sRandomTx.SampleLevel(g_ssSampler, jitterOffset, 0).xy;
	float2 jitter = 2.0f*tx_Random;
	float inverse = 1.0f/(float(N));
	float y = -float(2.f * DTid.y + jitter.x + 1.f - N) * inverse;
	float x = float(2.f * DTid.x + jitter.y + 1.f - N) * inverse;	
	float z = 2.0f;
#else
	float inverse = 1.0f/(float(N));
	float y = -float(2.f * DTid.y + 1.f - N) * inverse;
	float x = float(2.f * DTid.x + 1.f - N) * inverse;	
	float z = 2.0f;
#endif

	// Create new ray from the camera position to the pixel position
	Ray ray;
	float4 aux = (mul(float4(0,0,0,1.f),g_mfWorld));
	ray.vfOrigin = aux.xyz/aux.w;
	float3 vfPixelPosition = mul(float4(x,y,z,1.f),
		g_mfWorld).xyz;
	ray.vfDirection = normalize(vfPixelPosition-ray.vfOrigin);
	ray.fMaxT = 10000000000000000.f;
	ray.fMinT = 0;
	ray.vfReflectiveFactor = float3(1.f,1.f,1.f);
	ray.iTriangleId = -1;

	unsigned int index = DTid.y * N + DTid.x;
	// Copy ray to global UAV
	g_uRays[index] = ray;
	// Initialize accumulation buffer and result buffer
#ifdef GLOBAL_ILLUM
	g_uAccumulation[index] *= g_fFactorAnt;
#else
	g_uAccumulation[index] = 0.0f;
#endif

	g_uResultTexture[DTid.xy] = 0.f;
}
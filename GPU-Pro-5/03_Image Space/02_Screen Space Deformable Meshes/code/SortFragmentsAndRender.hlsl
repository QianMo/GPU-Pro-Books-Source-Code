//Copyright(c) 2009 - 2011, yakiimo02
//	All rights reserved.
//
//Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met :
//
//*Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
//
//	* Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and / or other materials provided with the distribution.
//
//	* Neither the name of Yakiimo3D nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
//
//	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// Original per-pixel linked list implementation code from Yakiimo02 was altered by Joao Raza and Gustavo Nunes for GPU Pro 5 'Screen Space Deformable Meshes via CSG with Per-pixel Linked Lists'

struct FragmentData
{
	unsigned int  nColor;
	float         fDepth;
	unsigned int  ShaderSubtractOperation;
};

struct FragmentLink
{
	FragmentData fragmentData;    // Fragment data
	unsigned int nNext;            // Link to next fragment
};

#define SSO_MESH_PLUS_BACK_FACING_PIXEL  0 
#define SSO_MESH_PLUS_FRONT_FACING_PIXEL 1 
#define SSO_MESH_MINUS_BACK_FACING_PIXEL       2 
#define SSO_MESH_MINUS_FRONT_FACING_PIXEL      3

/**
Constant buffer for StoreFragmentsAndRender and StoreFragments
*/
cbuffer CB : register(b0)
{
	uint g_nFrameWidth      : packoffset(c0.x);
	uint g_nFrameHeight     : packoffset(c0.y);
	uint g_nReserved0       : packoffset(c0.z);
	uint g_nReserved1       : packoffset(c0.w);
}

cbuffer CB2
{
	unsigned int ShaderSubtractOperation;
}

StructuredBuffer<FragmentLink> FragmentLinkSRV            : register(t0);
Buffer<uint> StartOffsetSRV                                : register(t1);

struct QuadVSinput
{
	float4 pos : POSITION;
};

struct QuadVS_Output
{
	float4 pos : SV_POSITION;
};

/**
Draw full screen quad.
*/
QuadVS_Output QuadVS(QuadVSinput Input)
{
	QuadVS_Output Output;
	Output.pos = Input.pos;
	return Output;
}

// Max hardcoded.
// We're in trouble if the fragment linked list is larger than 32...
#define TEMPORARY_BUFFER_MAX        32

struct PSOutput
{
	float4 color : SV_Target0;
};

/**
Sort and render fragments.
*/

float4 get_valid_back_pixel(FragmentData aData[TEMPORARY_BUFFER_MAX], int anIndex[TEMPORARY_BUFFER_MAX], int nNumFragment)
{
	//Obtain the first back pixel of M- when the camera is inside M+	
	int camera_is_inside_m_plus = 0;
	for (int i = 0; i < nNumFragment; i++)
	{
		if (aData[anIndex[i]].ShaderSubtractOperation == SSO_MESH_PLUS_FRONT_FACING_PIXEL)
			break;

		if (aData[anIndex[i]].ShaderSubtractOperation == SSO_MESH_PLUS_BACK_FACING_PIXEL)
		{
			camera_is_inside_m_plus = 1;
			break;
		}
	}

	float4 result = 0.0f;
		for (int i = 0; i < nNumFragment; i++)
		{
			if (aData[anIndex[i]].ShaderSubtractOperation == SSO_MESH_PLUS_BACK_FACING_PIXEL ||
				aData[anIndex[i]].ShaderSubtractOperation == SSO_MESH_PLUS_FRONT_FACING_PIXEL)
			{
				if (aData[anIndex[i]].ShaderSubtractOperation == SSO_MESH_PLUS_BACK_FACING_PIXEL)
					camera_is_inside_m_plus = 0;
				if (aData[anIndex[i]].ShaderSubtractOperation == SSO_MESH_PLUS_FRONT_FACING_PIXEL)
					camera_is_inside_m_plus = 1;
			}
			else
				if (camera_is_inside_m_plus == 1)
				{
					if (aData[anIndex[i]].ShaderSubtractOperation == SSO_MESH_MINUS_BACK_FACING_PIXEL)
					{
						uint nColor = aData[anIndex[i]].nColor;
						float4 color;
						color.r = ((nColor >> 0) & 0xFF) / 255.0f;
						color.g = ((nColor >> 8) & 0xFF) / 255.0f;
						color.b = ((nColor >> 16) & 0xFF) / 255.0f;
						color.a = ((nColor >> 24) & 0xFF) / 255.0f;

						float pixelDepth = aData[anIndex[i]].fDepth;

						result.rgba = float4(color.rgb, 1);
						return float4(pixelDepth, pixelDepth, pixelDepth, pixelDepth);
					}
				}
		}

		return float4(0, 0, 0, 0);
}

float4 get_valid_front_pixel(FragmentData aData[TEMPORARY_BUFFER_MAX], int anIndex[TEMPORARY_BUFFER_MAX], int nNumFragment)
{
	//Removes redundant M+ pixels from the algorithm
	int clean_list_index[TEMPORARY_BUFFER_MAX];
	int clean_list_size = 0;
	for (int i = 0; i < nNumFragment; i++)
	{
		if (aData[anIndex[i]].ShaderSubtractOperation == SSO_MESH_MINUS_FRONT_FACING_PIXEL)
		{
			if (i < nNumFragment - 1)
			{
				if (aData[anIndex[i + 1]].ShaderSubtractOperation == SSO_MESH_MINUS_BACK_FACING_PIXEL)
				{
					i++;
					continue;
				}
			}
		}

		clean_list_index[clean_list_size] = i;
		clean_list_size++;
	}

	//Removes all M+ pixels inside of M-	
	int plus_remove_state = 0;
	for (int i = 0; i < clean_list_size; i++)
	{
		if (aData[anIndex[clean_list_index[i]]].ShaderSubtractOperation == SSO_MESH_MINUS_FRONT_FACING_PIXEL)
			break;

		if (aData[anIndex[clean_list_index[i]]].ShaderSubtractOperation == SSO_MESH_MINUS_BACK_FACING_PIXEL)
		{
			plus_remove_state = 1;
			break;
		}
	}

	int excluded_plus_list_index[TEMPORARY_BUFFER_MAX];
	int excluded_plus_list_size = 0;
	int found_plus_front = 0;
	for (int i = 0; i < clean_list_size; i++)
	{
		if (aData[anIndex[clean_list_index[i]]].ShaderSubtractOperation == SSO_MESH_MINUS_BACK_FACING_PIXEL ||
			aData[anIndex[clean_list_index[i]]].ShaderSubtractOperation == SSO_MESH_MINUS_FRONT_FACING_PIXEL)
		{
			excluded_plus_list_index[excluded_plus_list_size] = clean_list_index[i];
			excluded_plus_list_size++;

			if (aData[anIndex[clean_list_index[i]]].ShaderSubtractOperation == SSO_MESH_MINUS_BACK_FACING_PIXEL)
				plus_remove_state = 0;
			if (aData[anIndex[clean_list_index[i]]].ShaderSubtractOperation == SSO_MESH_MINUS_FRONT_FACING_PIXEL)
				plus_remove_state = 1;
		}
		else
			if (plus_remove_state == 0)
			{
				excluded_plus_list_index[excluded_plus_list_size] = clean_list_index[i];
				excluded_plus_list_size++;

				if (aData[anIndex[clean_list_index[i]]].ShaderSubtractOperation == SSO_MESH_PLUS_FRONT_FACING_PIXEL)
					found_plus_front = 1;
			}
	}

	// Verifies in the excluded list if a front facing pixel of M+ is valid	
	PSOutput output;
	float4 result = 0.0f;
		for (int x = 0; x < excluded_plus_list_size; x++)
		{
			uint nColor = aData[anIndex[excluded_plus_list_index[x]]].nColor;
			float4 color;
			color.r = ((nColor >> 0) & 0xFF) / 255.0f;
			color.g = ((nColor >> 8) & 0xFF) / 255.0f;
			color.b = ((nColor >> 16) & 0xFF) / 255.0f;
			color.a = ((nColor >> 24) & 0xFF) / 255.0f;

			float pixelDepth = aData[anIndex[excluded_plus_list_index[x]]].fDepth;

			if (aData[anIndex[excluded_plus_list_index[x]]].ShaderSubtractOperation == SSO_MESH_PLUS_FRONT_FACING_PIXEL)
			{
				result.rgba = float4(color.rgb, 1);
				return float4(pixelDepth, pixelDepth, pixelDepth, pixelDepth);
			}
		}

		return float4(0, 0, 0, 0);
}

float4 SortFragmentsPS(QuadVS_Output input) : SV_Target0
{
	// index to current pixel.
	uint nIndex = (uint)input.pos.y * g_nFrameWidth + (uint)input.pos.x;

	FragmentData aData[TEMPORARY_BUFFER_MAX];            // temporary buffer
	int anIndex[TEMPORARY_BUFFER_MAX];                // index array for the tempory buffer
	uint nNumFragment = 0;                                // number of fragments in current pixel's linked list.
	uint nNext = StartOffsetSRV[nIndex];                // get first fragment from the start offset buffer.

	// early exit if no fragments in the linked list.
	if (nNext == 0xFFFFFFFF) {
		return float4(0.0, 0.0, 0.0, 0.0);
	}

	// Read and store linked list data to the temporary buffer.
	while (nNext != 0xFFFFFFFF)
	{
		FragmentLink element = FragmentLinkSRV[nNext];
		aData[nNumFragment] = element.fragmentData;
		anIndex[nNumFragment] = nNumFragment;
		++nNumFragment;
		nNext = element.nNext;
	}

	uint N2 = 1 << (int) (ceil(log2(nNumFragment)));

	// bitonic sort implementation needs on pow2 data.
	for (int i = nNumFragment; i < N2; i++)
	{
		anIndex[i] = i;
		aData[i].fDepth = 1.1f;
	}

	// Unoptimized sorting. (Bitonic Sort)

	// loop from Merge( 2 ) to Merge( nCount )
	for (int nMergeSize = 2; nMergeSize <= N2; nMergeSize = nMergeSize * 2)
	{
		// Merge( nCount ) requires log2( nCount ) merges. Merge( nCount/2 ) -> Merge( 2 )
		for (int nMergeSubSize = nMergeSize >> 1; nMergeSubSize > 0; nMergeSubSize = nMergeSubSize >> 1)
		{
			// compare and swap elements
			for (int nElem = 0; nElem < N2; ++nElem)
			{
				int nSwapElem = nElem^nMergeSubSize;
				// check to make sure to only swap once
				if (nSwapElem > nElem)
				{
					// sort in increasing order
					if ((nElem & nMergeSize) == 0 && aData[anIndex[nElem]].fDepth > aData[anIndex[nSwapElem]].fDepth)
					{
						int temp = anIndex[nElem];
						anIndex[nElem] = anIndex[nSwapElem];
						anIndex[nSwapElem] = temp;
					}

					// sort in descending order
					if ((nElem & nMergeSize) != 0 && aData[anIndex[nElem]].fDepth < aData[anIndex[nSwapElem]].fDepth)
					{
						int temp = anIndex[nElem];
						anIndex[nElem] = anIndex[nSwapElem];
						anIndex[nSwapElem] = temp;
					}
				}
			}
		}
	}

	float4 valid_front_pixel = get_valid_front_pixel(aData, anIndex, nNumFragment);
		float4 valid_back_pixel = get_valid_back_pixel(aData, anIndex, nNumFragment);

		if (valid_front_pixel.w != 0)
			return valid_front_pixel;
		else
			return valid_back_pixel;
}

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

struct SceneVS_Output
{
	float4 pos       : SV_POSITION;
	float4 color     : COLOR0;
	float2 texcoord  : TEXCOORD0;
	float4 screenPos : TEXCOORD1;
	float3 normal	 : NORMAL0;
	float3 worldPos : TEXCOORD2;
};

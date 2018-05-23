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

// Original OIT source code from Yakiimo02 altered by Joao Raza and Gustavo Nunes for GPU Pro 5 'Screen Space Deformable Meshes via CSG with Per-pixel Linked Lists'

struct SceneVS_Output
{
	float4 pos       : SV_POSITION;
	float4 color     : COLOR0;
	float2 texcoord  : TEXCOORD0;
	float4 screenPos : TEXCOORD1;
	float3 normal	 : NORMAL0;
	float3 worldPos : TEXCOORD2;
};

cbuffer cbPerObject : register(b0)
{
	row_major matrix    world;
	row_major matrix    viewProj;
}

struct SceneVS_Input
{
	float4 pos      : POSITION0;
	float4 color    : COLOR0;
	float3 normal   : NORMAL0;
	float2 texcoord : TEXCOORD0;
};

/**
Both OIT and Non-OIT pixel shaders use this vertex shader.
*/
SceneVS_Output SceneVS(SceneVS_Input input)
{
	SceneVS_Output output;

	output.color = input.color;
	output.worldPos = mul(float4(input.pos.xyz, 1), world).xyz;
	output.pos = mul(float4(output.worldPos, 1), viewProj);
	output.screenPos.xyzw = output.pos.xyzw;
	output.normal = input.normal;
	output.texcoord = input.texcoord;

	return output;
}

/**
Non-OIT pass pixel shader.
*/
[earlydepthstencil]
float4 ScenePS(SceneVS_Output input) : SV_Target0
{
	//return float4( input.pos.x/320.0 , input.pos.y/240.0, 0, 1 );
	return float4(input.color.xyz, 0.5f);
}

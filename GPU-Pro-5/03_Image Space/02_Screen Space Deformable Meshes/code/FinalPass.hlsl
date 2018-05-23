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

Texture2D shaderTexture : register(t);
SamplerState SampleType : register(s);

cbuffer cbPerFrame : register(b2)
{
	float3 eyePos;
	bool   invert_normal;
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

/**
Non-OIT pass pixel shader.
*/

float4 Shade(float3 lightDir, float3 normal, float3 color, float3 eyeDir)
{
	float3 inv_normal = (invert_normal) ? -normal : normal;

	float3 halfVec = normalize(lightDir + eyeDir);
	float cosTh = saturate(dot(halfVec, inv_normal));
	float specular = pow(cosTh, 500);

	float3 finalColor = (max(dot(-inv_normal, -lightDir), 0) * float3(0, 0, 1)) + 0.1f*color + 0.5f * specular;

	return float4(finalColor, 1);
}

float4 PS(SceneVS_Output input) : SV_Target0
{
	float2 texCoords = float2((1 + (input.screenPos.x / input.screenPos.w)) / 2.0f, (1 - (input.screenPos.y / input.screenPos.w)) / 2.0f);

	float csgDepth = shaderTexture.Sample(SampleType, texCoords).r;

	float3 normal = normalize(float3(-5, -0.5f, 1));

	float3 eyeDir = normalize(eyePos - input.worldPos);

	if (abs((input.screenPos.z / input.screenPos.w) - csgDepth) <= 0.0000009f)
		return Shade(normal, normalize(input.normal), input.color, eyeDir);	
	else
		clip(-1);

	return Shade(normal, normalize(input.normal), input.color, eyeDir);

}


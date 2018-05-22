cbuffer cbPerFrame : register(b0) {
	float3 g_wLightPos;
};

cbuffer cbPerObject : register(b1) {
	row_major float4x4 g_matModelToWorld;
	row_major float4x4 g_matModelToProj;
	float3 g_colDiffuse;
};

//==============================================================================================================================================================

struct VSInput_Model {
	float4 pos			: POSITION;
	float3 normal		: NORMAL;
};

struct PSInput_RenderModel {
    float4 pos			: SV_Position;
    float4 wPos			: POSITION_WORLD;
    float3 wNormal		: NORMAL_WORLD;
};

struct PSOutput_Color {
	float4 color		: SV_Target;
};

//==============================================================================================================================================================

PSInput_RenderModel VS_RenderModel(VSInput_Model input) {
    PSInput_RenderModel output;

    output.pos = mul(g_matModelToProj, input.pos);
    output.wPos = mul(g_matModelToWorld, input.pos);
    output.wNormal = mul((float3x3)g_matModelToWorld, input.normal);

    return output;
}

PSOutput_Color PS_RenderModel(PSInput_RenderModel input) {
	PSOutput_Color output;

	float3 P = input.wPos.xyz / input.wPos.w;
	float3 L = normalize(g_wLightPos - P);
	float3 N = normalize(input.wNormal);
	float cosNL = saturate(dot(N, L));

	output.color.xyz = g_colDiffuse * cosNL;
	output.color.w = 1.0;
	return output;
}

float4x4	MVPMatrix				: WORLDVIEWPROJECTION;
float4x4	ViewProjMatrix			: VIEWPROJECTION;
float4		LightDirModel			: LIGHTDIRMODEL0;
float4		LightDirWorld			: LIGHTDIRWORLD0;
int4		BoneCount				: BONECOUNT;
float4x4	BoneMatrixArray[8]		: BONEMATRIXARRAY;
float4x4	BoneMatrixArrayIT[8]	: BONEMATRIXARRAYINVERSETRANSPOSE;
float3 MatColAmbient				: MATERIALCOLORAMBIENT;
float3 MatColDiffuse				: MATERIALCOLORDIFFUSE;
float3 MatColSpecular				: MATERIALCOLORSPECULAR;

Texture2D tex0 : 	TEXTURE0 < string name = "Legs.pvr"; >;

SamplerState samplerState
{
    Filter = MIN_MAG_MIP_LINEAR;
    AddressU = Wrap;
    AddressV = Wrap;
};

struct VS_INPUT
{	
	float3 Vertex		: POSITION;
	float3 Normal		: NORMAL;
	float2 UV			: TEXCOORD;
	uint4 BoneIndex		: BONEINDICES;
	float4 BoneWeights	: BONEWEIGHT;
};

struct VS_OUTPUT
{
	float4 Position			: SV_POSITION;
	float2 TexCoord			: TEXCOORD;
	float  LightIntensity	: VARDOT;
};

VS_OUTPUT VS(VS_INPUT In)
{
	const float3 cLightDir = float3(-1.0,1.0,-1.0);
	
	VS_OUTPUT Out;
	
	float4 position = float4(0,0,0,0.0);
	float3 normal = float3(0,0,0);
	
	if(BoneCount.x > 0)
	{	
		for(int i = 0; i < 3; ++i)
		{
			if(i<BoneCount.x)
			{
				if(In.BoneWeights[i] > 0.0)
				{
					float4x4 boneMatrix = BoneMatrixArray[In.BoneIndex[i]];
					float4x4 normalMatrix = BoneMatrixArrayIT[In.BoneIndex[i]];
	
					position += mul(float4(In.Vertex, 1.0), boneMatrix) * In.BoneWeights[i];
					normal += mul(float4(In.Normal, 1.0), normalMatrix) * In.BoneWeights[i];
				}
			}
		}		
	
		Out.Position = mul(position, ViewProjMatrix);
		Out.LightIntensity	= dot(normalize(normal.xyz), LightDirWorld);
	}
	else
	{
		Out.Position = mul(float4(In.Vertex, 1.0), MVPMatrix);
		Out.LightIntensity	= dot(normalize(normal.xyz), LightDirModel);
	}

	Out.TexCoord = In.UV;
	return Out;
}

float4 PS(VS_OUTPUT In) : SV_Target
{
	float3 texColor = tex0.Sample(samplerState, float2(In.TexCoord.x, -In.TexCoord.y));
	float3 color = texColor * In.LightIntensity;
	return float4(color, 1.0);
}

technique10 Skinning
{
    pass P0
    {
        SetVertexShader( CompileShader( vs_4_0, VS() ) );
        SetGeometryShader( NULL );
        SetPixelShader( CompileShader( ps_4_0, PS() ) );
    }
}

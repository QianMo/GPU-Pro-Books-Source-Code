/**
 *	Pedro Hermosilla
 *
 *	Moving Group - UPC
 *	Pencil.fx
 */
 
#define PI 3.1416f

Texture2D texPencil;
SamplerState samLinear
{
    Filter = MIN_MAG_MIP_POINT;
    AddressU = Wrap;
    AddressV = Wrap;
};

DepthStencilState EnableDepth
{
    DepthEnable = TRUE;
    DepthWriteMask = ALL;
    DepthFunc = LESS_EQUAL;
};

BlendState NoBlend
{
    BlendEnable[0] = FALSE;
};

cbuffer constantBuffer
{      
    float4x4 WorldViewProj;
	float4x4 world;
	float3 lightDir;
	float width;
	float height;
	float texRepeat;
	float OParam;
	float SParam;
};

struct VERTEXin
{
	float4 Pos : POSITION;
	float3 Normal : NORMAL; 
	float3 curv : TANGENT;
};

struct VERTEXout
{
	float4 Pos : SV_POSITION;
	float3 norm : TEXCOORD0;
	float2 curv : TEXCOORD1;
};

struct GEOMin
{
	float4 Pos : SV_POSITION;
	float3 norm : TEXCOORD0;
	float2 curv : TEXCOORD1;
};

struct GEOMout
{
	float4 Pos : SV_POSITION;
	float3 norm : TEXCOORD0;
	float2 curv1 : TEXCOORD1;
	float2 curv2 : TEXCOORD2;
	float2 curv3 : TEXCOORD3;
};

struct PIXELin
{
	float4 Pos : SV_POSITION;
	float3 norm : TEXCOORD0;
	float2 curv1 : TEXCOORD1;
	float2 curv2 : TEXCOORD2;
	float2 curv3 : TEXCOORD3;
};

////////////////////VERTEX SHADER/////////////////////////

VERTEXout pencilVert(VERTEXin inPut)
{
	VERTEXout outPut;
    outPut.Pos = mul(inPut.Pos,WorldViewProj);
    float4 vert = outPut.Pos;
    vert = vert/ vert.w;
    float4 newPos = inPut.Pos + float4(inPut.curv,0.0f);
    newPos = mul(newPos,WorldViewProj);
    newPos = newPos / newPos.w;
    float2 curvDir = newPos.xy - vert.xy;
    outPut.curv = normalize(curvDir);
    outPut.norm = mul(inPut.Normal,world);
	return outPut;
}

////////////////////GEOMETRY SHADER/////////////////////////

[maxvertexcount(3)]
void pencilGeom( triangle GEOMin input[3], inout TriangleStream<GEOMout> TriStream )
{
	GEOMout outVert;
	outVert.Pos = input[0].Pos;
	outVert.norm = input[0].norm;
	outVert.curv1 = input[0].curv;
	outVert.curv2 = input[1].curv;
	outVert.curv3 = input[2].curv;
	TriStream.Append(outVert);
	
	outVert.Pos = input[1].Pos;
	outVert.norm = input[1].norm;
	outVert.curv1 = input[0].curv;
	outVert.curv2 = input[1].curv;
	outVert.curv3 = input[2].curv;
	TriStream.Append(outVert);
	
	outVert.Pos = input[2].Pos;
	outVert.norm = input[2].norm;
	outVert.curv1 = input[0].curv;
	outVert.curv2 = input[1].curv;
	outVert.curv3 = input[2].curv;
	TriStream.Append(outVert);
	
	TriStream.RestartStrip();
}

////////////////////PIXEL SHADER/////////////////////////

float4 pencilPixel(PIXELin inPut):SV_Target
{
	float2 xdir = float2(1.0f,0.0f);
	float2x2 rotMat;
	
	float2 uv = float2(inPut.Pos.x/width,inPut.Pos.y/height);
	
	float2 uvDir = normalize(inPut.curv1);
	float angle = atan(uvDir.y/uvDir.x);
	angle = (uvDir.x < 0.0f)?angle+PI:(uvDir.y < 0.0f)?angle+(2*PI):angle;
	
	float cosVal = cos(angle);
	float sinVal = sin(angle);
	rotMat[0][0] = cosVal;
	rotMat[1][0] = -sinVal;
	rotMat[0][1] = sinVal;
	rotMat[1][1] = cosVal;
	
	float2 uv1 = mul(uv,rotMat) * texRepeat;
	
	uvDir = normalize(inPut.curv2);
	angle = atan(uvDir.y/uvDir.x);
	angle = (uvDir.x < 0.0f)?angle+PI:(uvDir.y < 0.0f)?angle+(2*PI):angle;
	
	cosVal = cos(angle);
	sinVal = sin(angle);
	rotMat[0][0] = cosVal;
	rotMat[1][0] = -sinVal;
	rotMat[0][1] = sinVal;
	rotMat[1][1] = cosVal;
	
	float2 uv2 = mul(uv,rotMat) * texRepeat;
	
	uvDir = normalize(inPut.curv3);
	angle = atan(uvDir.y/uvDir.x);
	angle = (uvDir.x < 0.0f)?angle+PI:(uvDir.y < 0.0f)?angle+(2*PI):angle;
	
	cosVal = cos(angle);
	sinVal = sin(angle);
	rotMat[0][0] = cosVal;
	rotMat[1][0] = -sinVal;
	rotMat[0][1] = sinVal;
	rotMat[1][1] = cosVal;
	
	float2 uv3 = mul(uv,rotMat) * texRepeat;
	
	float percen = 1.0f - max(dot(normalize(inPut.norm),lightDir),0.0);
	
	float4 color = (texPencil.Sample(samLinear,uv1)*0.333f)
			+(texPencil.Sample(samLinear,uv2)*0.333f)
			+(texPencil.Sample(samLinear,uv3)*0.333f);
			
	color.w = 1.0f;
	
	percen = (percen*OParam) + SParam;
	color.xyz = pow(color.xyz,float3(percen,percen,percen));	
    return color;
}

////////////////////TECHNIQUE/////////////////////////

technique10 Pencil
{
    pass p0
    {
        SetVertexShader( CompileShader( vs_4_0, pencilVert() ) );
        SetGeometryShader( CompileShader( gs_4_0, pencilGeom() ) );
        SetPixelShader( CompileShader( ps_4_0, pencilPixel() ) );
        
        SetDepthStencilState( EnableDepth, 0 );
        SetBlendState( NoBlend, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xFFFFFFFF );
    }
};


Texture2D txScene : register( t0 );
SamplerState samAniso : register( s0 );


#define XAA_EDGE_THRESHOLD 0.25
#define FXAA_EDGE_THRESHOLD_MIN 1.0/16.0
#define FXAA_SEARCH_THRESHOLD 1.0/4.0
#define FXAA_SEARCH_STEPS 16

cbuffer cbParams : register( b0 )
{
	///< Screen Resolution for post process!
    float4 FrameRes;  
    
    ///< 0=No AA
	///< 1=Cheap Edge Detect AA
	///< 2=Line Search! 
    float4 AA_LEVEL;
};

struct VS_INPUT
{
    float4 Pos : POSITION;
};

struct PS_INPUT
{
    float4 Pos : SV_POSITION;
};

///< Vertex Shader
PS_INPUT FXAA_VS(VS_INPUT _x)
{
	PS_INPUT output = (PS_INPUT)0;
	output.Pos = _x.Pos;
	
    return output;    
}

float3 Luma(float3 _rgb)
{
	return _rgb.y * (0.587/0.299) + _rgb.x;
}

///<
float4 FxaaTextureOffset(Texture2D _t, float2 _UV, float2 _offset)
{
	return txScene.Sample(samAniso, (_UV+_offset)*FrameRes);
}

float4 NonLinearColor(float3 _rgb)
{
	///< sRGB conversion when using textures !
	return float4(_rgb,1);
}

///< Pixel Shader
float4 FXAA_PS(PS_INPUT _x) : SV_Target
{

	float2 Res=FrameRes;
	float2 UVPos = _x.Pos.xy*Res;
	
	float3 rgbM = FxaaTextureOffset(txScene, _x.Pos.xy, float2(0,0)).xyz;
	
	///< No AA
	if (AA_LEVEL.x==0)
		return float4(rgbM,1);
	
	float3 rgbN = FxaaTextureOffset(txScene, _x.Pos.xy, float2(0,-1)).xyz;
	float3 rgbS = FxaaTextureOffset(txScene, _x.Pos.xy, float2(0,1)).xyz;
	
	float3 rgbE = FxaaTextureOffset(txScene, _x.Pos.xy, float2(1,0)).xyz;
	float3 rgbW = FxaaTextureOffset(txScene, _x.Pos.xy, float2(-1,0)).xyz;
	
	float lumaN = Luma(rgbN);
	float lumaW = Luma(rgbW);
	float lumaM = Luma(rgbM);
	float lumaE = Luma(rgbE);
	float lumaS = Luma(rgbS);

	float rangeMin = min(lumaM, min(min(lumaN, lumaW), min(lumaS, lumaE)));	
	float rangeMax = max(lumaM, max(max(lumaN, lumaW), max(lumaS, lumaE)));
	
	float range = rangeMax - rangeMin;
		
	if (range < max(FXAA_EDGE_THRESHOLD_MIN, rangeMax * XAA_EDGE_THRESHOLD)) 
	{
		return float4(rgbM,1); 
	}
	
	float lumaL = (lumaN + lumaW + lumaE + lumaS) * 0.25;
    float rangeL = abs(lumaL - lumaM);
    
    float blendL = rangeL / range; 
    
    float3 rgbNW = FxaaTextureOffset(txScene, _x.Pos.xy, float2(-1,-1)).xyz;
    float3 rgbNE = FxaaTextureOffset(txScene, _x.Pos.xy, float2( 1,-1)).xyz;
    float3 rgbSW = FxaaTextureOffset(txScene, _x.Pos.xy, float2(-1, 1)).xyz;
    float3 rgbSE = FxaaTextureOffset(txScene, _x.Pos.xy, float2( 1, 1)).xyz;
    
    float3 rgbL = (rgbN + rgbW + rgbE + rgbS + rgbM + rgbNW + rgbNE + rgbSW + rgbSE)*1.0/9.0;
    		
    float lumaNW = Luma(rgbNW);
    float lumaNE = Luma(rgbNE);
    float lumaSW = Luma(rgbSW);
    float lumaSE = Luma(rgbSE);
    
    float edgeVert = 
        abs((0.25 * lumaNW) + (-0.5 * lumaN) + (0.25 * lumaNE)) +
        abs((0.50 * lumaW ) + (-1.0 * lumaM) + (0.50 * lumaE )) +
        abs((0.25 * lumaSW) + (-0.5 * lumaS) + (0.25 * lumaSE));
    float edgeHorz = 
        abs((0.25 * lumaNW) + (-0.5 * lumaW) + (0.25 * lumaSW)) +
        abs((0.50 * lumaN ) + (-1.0 * lumaM) + (0.50 * lumaS )) +
        abs((0.25 * lumaNE) + (-0.5 * lumaE) + (0.25 * lumaSE));
    
   
	bool horzSpan = edgeHorz >= edgeVert;
    float lengthSign = horzSpan ? -Res.y : -Res.x;
   
   	if(!horzSpan) 
		lumaN = lumaW;
	if(!horzSpan) 
		lumaS = lumaE;
	   
    float gradientN = abs(lumaN - lumaM);
    float gradientS = abs(lumaS - lumaM);    
    
    lumaN = (lumaN + lumaM) * 0.5;
    lumaS = (lumaS + lumaM) * 0.5;		
   
   ///< Pair Test. 
    bool pairN = gradientN >= gradientS;

    if(!pairN) 
    {
		lumaN = lumaS;
		gradientN = gradientS;
		lengthSign *= -1.0;
	}
	gradientN *= FXAA_SEARCH_THRESHOLD;
		
    float2 posN;
    posN.x = UVPos.x + (horzSpan ? 0.0 : lengthSign * 0.5);
    posN.y = UVPos.y + (horzSpan ? lengthSign * 0.5 : 0.0);
	float2 posP = posN;
	
    float2 offNP = horzSpan ? float2(Res.x, 0.0) : float2(0.0f, Res.y); 
    float lumaEndN = lumaN;
    float lumaEndP = lumaN;
    bool doneN = false;
    bool doneP = false;

    posN += offNP * float2(-1.0, -1.0);
    posP += offNP * float2( 1.0,  1.0);
    offNP *= float2(1.0, 1.0);
    
    
    for (int i = 0; i < FXAA_SEARCH_STEPS; i++) 
    {    
  /*
            if(!doneN) 
				lumaEndN = Luma(
				txScene.SampleGrad(samAniso, posN.xy, offNP, offNP).xyz);				
            if(!doneP) 
				lumaEndP = Luma(
				txScene.SampleGrad(samAniso, posP.xy, offNP, offNP).xyz);
				*/

     
        if(!doneN) 
			lumaEndN = Luma( txScene.SampleLevel(samAniso, posN.xy, 0.0).xyz );
        if(!doneP) 
			lumaEndP = Luma( txScene.SampleLevel(samAniso, posP.xy, 0.0).xyz );


        doneN = doneN || (abs(lumaEndN - lumaN) >= gradientN);
        doneP = doneP || (abs(lumaEndP - lumaN) >= gradientN);
        
        if(doneN && doneP)
			break;
        if(!doneN)
			posN -= offNP;
        if(!doneP)
			posP += offNP;

    }        
        
    float dstN = horzSpan ? UVPos.x - posN.x : UVPos.y - posN.y;
    float dstP = horzSpan ? posP.x - UVPos.x : posP.y - UVPos.y;
    
    bool directionN = dstN < dstP;
      
    lumaEndN = directionN ? lumaEndN : lumaEndP;
    
    ///< 
    if(((lumaM - lumaN) < 0.0) == ((lumaEndN - lumaN) < 0.0)) 
        lengthSign = 0.0;
    
    float spanLength = (dstP + dstN);
    dstN = directionN ? dstN : dstP;
    float subPixelOffset = (0.5 + (dstN * (-1.0/spanLength))) * lengthSign;
    
    float3 rgbF = txScene.SampleLevel(samAniso, float2(UVPos.x + (horzSpan ? 0.0 : subPixelOffset), UVPos.y + (horzSpan ? subPixelOffset : 0.0)), 0.0).xyz;
       
    return float4(blendL*rgbL + (1.0-blendL)*rgbF,1);
    
}

 
float Script : STANDARDSGLOBAL <
    string UIWidget = "none";
    string ScriptClass = "object";
    string ScriptOrder = "standard";
    string ScriptOutput = "color";
    string Script = "Technique=Technique?SimplePS:TexturedPS:SimpleQuadraticPS:TexturedQuadraticPS;";
> = 0.8;

/************* UN-TWEAKABLES **************/

//// UN-TWEAKABLES - AUTOMATICALLY-TRACKED TRANSFORMS ////////////////

float4x4 WvpXf : WorldViewProjection < string UIWidget="None"; >;

float Life < string UIWidget = "None"; > = 1.0f;
float3 Color < string UIWidget = "None"; > = float3(1,1,1);

/************* TWEAKABLES **************/

texture ColorTexture : DIFFUSE <
    string ResourceName = "Particle.dds";
    string UIName =  "Diffuse Texture";
    string ResourceType = "2D";
>;

sampler2D ColorSampler = sampler_state {
    Texture = <ColorTexture>;
    FILTER = MIN_MAG_MIP_LINEAR;
    AddressU = Wrap;
    AddressV = Wrap;
};  

/************* DATA STRUCTS **************/

/* data from application vertex buffer */
struct appdata {
    float3 Position	: POSITION;
    float4 UV		: TEXCOORD0;
};

/* data passed from vertex shader to pixel shader */
struct vertexOutput {
    float4 HPosition	: POSITION;
    float2 UV		: TEXCOORD0;
};

/*********** vertex shader for pixel-shaded versions ******/

vertexOutput vShader(appdata IN)
{
    vertexOutput OUT = (vertexOutput)0;
    
    OUT.HPosition = mul(float4(IN.Position.xyz,1),WvpXf);
    OUT.UV = IN.UV;
    
    return OUT;
}

/********* pixel shaders ********/

float4 pShader(vertexOutput IN) : COLOR 
{
    float4 map = tex2D(ColorSampler,IN.UV);
    
    return float4(map.rgb * Color.rgb, map.a * Life);
}

//////////////////////////////////////////////////////////////////////
// TECHNIQUES ////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

BlendState NormalBlend
{
	BlendEnable[0] = TRUE;
    SrcBlend = SRC_ALPHA;
    DestBlend = INV_SRC_ALPHA;
};

BlendState ScreenBlend
{
	BlendEnable[0] = TRUE;
    SrcBlend = INV_DEST_COLOR;
    DestBlend = ONE;
};

BlendState AdditionBlend
{
	BlendEnable[0] = TRUE;
    SrcBlend = SRC_ALPHA;
    DestBlend = ONE;
};

BlendState SubtractBlend
{
	BlendEnable[0] = TRUE;
    SrcBlend = SRC_ALPHA;
    DestBlend = ONE;
    BlendOp = REV_SUBTRACT;
};

BlendState CompareBlend
{
	BlendEnable[0] = TRUE;
    SrcBlend = ONE;
    DestBlend = ONE;
	BlendOp = MAX;
};

BlendState ContrastBlend
{
	BlendEnable[0] = TRUE;
    SrcBlend = DEST_COLOR;
    DestBlend = ONE;
};

RasterizerState NormalState
{
    CullMode = Back;
};

DepthStencilState ZEnabled
{
	DepthEnable = TRUE;
};

DepthStencilState ZDisabled
{
	DepthEnable = FALSE;
};

technique Normal <
	string Script = "Pass=p0;";
> {
	pass p0 <
	string Script = "Draw=geometry;";
    > {		
        VertexShader = compile vs_2_0 vShader();
        ZEnable = true;
        ZWriteEnable = true;
        ZFunc = LessEqual;
        AlphaBlendEnable = true;
        CullMode = CCW;
        SrcBlend = SrcAlpha;
        DestBlend = InvSrcAlpha;
        BlendOp = Add;
        PixelShader = compile ps_2_0 pShader();
    }
}

technique10 Normal10 <
	string Script = "Pass=p0;";
> {
    pass p0 <
	string Script = "Draw=geometry;";
    > {
        SetVertexShader( CompileShader( vs_4_0, vShader() ) );
        SetGeometryShader( NULL );
        SetPixelShader( CompileShader( ps_4_0, pShader() ) );
                
        SetRasterizerState(NormalState);       
		SetDepthStencilState(ZEnabled, 0);
		SetBlendState(NormalBlend, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xFFFFFFFF);
    }
}

technique Screen <
	string Script = "Pass=p0;";
> {
	pass p0 <
	string Script = "Draw=geometry;";
    > {		
        VertexShader = compile vs_2_0 vShader();
        ZEnable = true;
        ZWriteEnable = true;
        ZFunc = LessEqual;
        AlphaBlendEnable = true;
        CullMode = CCW;
        SrcBlend = InvDestColor;
        DestBlend = One;
        BlendOp = Add;
        PixelShader = compile ps_2_0 pShader();
    }
}

technique10 Screen10 <
	string Script = "Pass=p0;";
> {
    pass p0 <
	string Script = "Draw=geometry;";
    > {
        SetVertexShader( CompileShader( vs_4_0, vShader() ) );
        SetGeometryShader( NULL );
        SetPixelShader( CompileShader( ps_4_0, pShader() ) );
                
        SetRasterizerState(NormalState);       
		SetDepthStencilState(ZEnabled, 0);
		SetBlendState(ScreenBlend, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xFFFFFFFF);
    }
}

technique Addition <
	string Script = "Pass=p0;";
> {
	pass p0 <
	string Script = "Draw=geometry;";
    > {		
        VertexShader = compile vs_2_0 vShader();
        ZEnable = true;
        ZWriteEnable = true;
        ZFunc = LessEqual;
        AlphaBlendEnable = true;
        CullMode = CCW;
        SrcBlend = SrcAlpha;
        DestBlend = One;
        BlendOp = Add;
        PixelShader = compile ps_2_0 pShader();
    }
}

technique10 Addition10 <
	string Script = "Pass=p0;";
> {
    pass p0 <
	string Script = "Draw=geometry;";
    > {
        SetVertexShader( CompileShader( vs_4_0, vShader() ) );
        SetGeometryShader( NULL );
        SetPixelShader( CompileShader( ps_4_0, pShader() ) );
                
        SetRasterizerState(NormalState);       
		SetDepthStencilState(ZEnabled, 0);
		SetBlendState(AdditionBlend, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xFFFFFFFF);
    }
}

technique Subtraction <
	string Script = "Pass=p0;";
> {
	pass p0 <
	string Script = "Draw=geometry;";
    > {		
        VertexShader = compile vs_2_0 vShader();
        ZEnable = true;
        ZWriteEnable = true;
        ZFunc = LessEqual;
        AlphaBlendEnable = true;
        CullMode = CCW;
        SrcBlend = SrcAlpha;
        DestBlend = One;
        BlendOp = RevSubtract;
        PixelShader = compile ps_2_0 pShader();
    }
}

technique10 Subtraction10 <
	string Script = "Pass=p0;";
> {
    pass p0 <
	string Script = "Draw=geometry;";
    > {
        SetVertexShader( CompileShader( vs_4_0, vShader() ) );
        SetGeometryShader( NULL );
        SetPixelShader( CompileShader( ps_4_0, pShader() ) );
                
        SetRasterizerState(NormalState);       
		SetDepthStencilState(ZEnabled, 0);
		SetBlendState(SubtractBlend, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xFFFFFFFF);
    }
}

technique CompareLight <
	string Script = "Pass=p0;";
> {
	pass p0 <
	string Script = "Draw=geometry;";
    > {		
        VertexShader = compile vs_2_0 vShader();
        ZEnable = true;
        ZWriteEnable = true;
        ZFunc = LessEqual;
        AlphaBlendEnable = true;
        CullMode = CCW;
        SrcBlend = One;
        DestBlend = One;
        BlendOp = Max;
        PixelShader = compile ps_2_0 pShader();
    }
}

technique10 CompareLight10 <
	string Script = "Pass=p0;";
> {
    pass p0 <
	string Script = "Draw=geometry;";
    > {
        SetVertexShader( CompileShader( vs_4_0, vShader() ) );
        SetGeometryShader( NULL );
        SetPixelShader( CompileShader( ps_4_0, pShader() ) );
                
        SetRasterizerState(NormalState);       
		SetDepthStencilState(ZEnabled, 0);
		SetBlendState(CompareBlend, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xFFFFFFFF);
    }
}

technique Contrast <
	string Script = "Pass=p0;";
> {
	pass p0 <
	string Script = "Draw=geometry;";
    > {		
        VertexShader = compile vs_2_0 vShader();
        ZEnable = true;
        ZWriteEnable = true;
        ZFunc = LessEqual;
        AlphaBlendEnable = true;
        CullMode = CCW;
        SrcBlend = DestColor;
        DestBlend = One;
        BlendOp = Add;
        PixelShader = compile ps_2_0 pShader();
    }
}

technique10 Contrast10 <
	string Script = "Pass=p0;";
> {
    pass p0 <
	string Script = "Draw=geometry;";
    > {
        SetVertexShader( CompileShader( vs_4_0, vShader() ) );
        SetGeometryShader( NULL );
        SetPixelShader( CompileShader( ps_4_0, pShader() ) );
                
        SetRasterizerState(NormalState);       
		SetDepthStencilState(ZEnabled, 0);
		SetBlendState(ContrastBlend, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xFFFFFFFF);
    }
}


technique NormalNoZ <
	string Script = "Pass=p0;";
> {
	pass p0 <
	string Script = "Draw=geometry;";
    > {		
        VertexShader = compile vs_2_0 vShader();
        ZEnable = false;
        ZWriteEnable = false;
        AlphaBlendEnable = true;
        CullMode = CCW;
        SrcBlend = SrcAlpha;
        DestBlend = InvSrcAlpha;
        BlendOp = Add;
        PixelShader = compile ps_2_0 pShader();
    }
}

technique10 NormalNoZ10 <
	string Script = "Pass=p0;";
> {
    pass p0 <
	string Script = "Draw=geometry;";
    > {
        SetVertexShader( CompileShader( vs_4_0, vShader() ) );
        SetGeometryShader( NULL );
        SetPixelShader( CompileShader( ps_4_0, pShader() ) );
                
        SetRasterizerState(NormalState);       
		SetDepthStencilState(ZDisabled, 0);
		SetBlendState(NormalBlend, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xFFFFFFFF);
    }
}

technique ScreenNoZ <
	string Script = "Pass=p0;";
> {
	pass p0 <
	string Script = "Draw=geometry;";
    > {		
        VertexShader = compile vs_2_0 vShader();
        ZEnable = false;
        ZWriteEnable = false;
        AlphaBlendEnable = true;
        CullMode = CCW;
        SrcBlend = InvDestColor;
        DestBlend = One;
        BlendOp = Add;
        PixelShader = compile ps_2_0 pShader();
    }
}

technique10 ScreenNoZ10 <
	string Script = "Pass=p0;";
> {
    pass p0 <
	string Script = "Draw=geometry;";
    > {
        SetVertexShader( CompileShader( vs_4_0, vShader() ) );
        SetGeometryShader( NULL );
        SetPixelShader( CompileShader( ps_4_0, pShader() ) );
                
        SetRasterizerState(NormalState);       
		SetDepthStencilState(ZDisabled, 0);
		SetBlendState(ScreenBlend, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xFFFFFFFF);
    }
}

technique AdditionNoZ <
	string Script = "Pass=p0;";
> {
	pass p0 <
	string Script = "Draw=geometry;";
    > {		
        VertexShader = compile vs_2_0 vShader();
        ZEnable = false;
        ZWriteEnable = false;
        AlphaBlendEnable = true;
        CullMode = CCW;
        SrcBlend = SrcAlpha;
        DestBlend = One;
        BlendOp = Add;
        PixelShader = compile ps_2_0 pShader();
    }
}

technique10 AdditionNoZ10 <
	string Script = "Pass=p0;";
> {
    pass p0 <
	string Script = "Draw=geometry;";
    > {
        SetVertexShader( CompileShader( vs_4_0, vShader() ) );
        SetGeometryShader( NULL );
        SetPixelShader( CompileShader( ps_4_0, pShader() ) );
                
        SetRasterizerState(NormalState);       
		SetDepthStencilState(ZDisabled, 0);
		SetBlendState(AdditionBlend, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xFFFFFFFF);
    }
}

technique SubtractionNoZ <
	string Script = "Pass=p0;";
> {
	pass p0 <
	string Script = "Draw=geometry;";
    > {		
        VertexShader = compile vs_2_0 vShader();
        ZEnable = false;
        ZWriteEnable = false;
        AlphaBlendEnable = true;
        CullMode = CCW;
        SrcBlend = SrcAlpha;
        DestBlend = One;
        BlendOp = RevSubtract;
        PixelShader = compile ps_2_0 pShader();
    }
}

technique10 SubtractionNoZ10 <
	string Script = "Pass=p0;";
> {
    pass p0 <
	string Script = "Draw=geometry;";
    > {
        SetVertexShader( CompileShader( vs_4_0, vShader() ) );
        SetGeometryShader( NULL );
        SetPixelShader( CompileShader( ps_4_0, pShader() ) );
                
        SetRasterizerState(NormalState);       
		SetDepthStencilState(ZDisabled, 0);
		SetBlendState(SubtractBlend, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xFFFFFFFF);
    }
}

technique CompareLightNoZ <
	string Script = "Pass=p0;";
> {
	pass p0 <
	string Script = "Draw=geometry;";
    > {		
        VertexShader = compile vs_2_0 vShader();
        ZEnable = false;
        ZWriteEnable = false;
        AlphaBlendEnable = true;
        CullMode = CCW;
        SrcBlend = One;
        DestBlend = One;
        BlendOp = Max;
        PixelShader = compile ps_2_0 pShader();
    }
}

technique10 CompareLightNoZ10 <
	string Script = "Pass=p0;";
> {
    pass p0 <
	string Script = "Draw=geometry;";
    > {
        SetVertexShader( CompileShader( vs_4_0, vShader() ) );
        SetGeometryShader( NULL );
        SetPixelShader( CompileShader( ps_4_0, pShader() ) );
                
        SetRasterizerState(NormalState);       
		SetDepthStencilState(ZDisabled, 0);
		SetBlendState(CompareBlend, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xFFFFFFFF);
    }
}

technique ContrastNoZ <
	string Script = "Pass=p0;";
> {
	pass p0 <
	string Script = "Draw=geometry;";
    > {		
        VertexShader = compile vs_2_0 vShader();
        ZEnable = false;
        ZWriteEnable = false;
        AlphaBlendEnable = true;
        CullMode = CCW;
        SrcBlend = DestColor;
        DestBlend = One;
        BlendOp = Add;
        PixelShader = compile ps_2_0 pShader();
    }
}

technique10 ContrastNoZ10 <
	string Script = "Pass=p0;";
> {
    pass p0 <
	string Script = "Draw=geometry;";
    > {
        SetVertexShader( CompileShader( vs_4_0, vShader() ) );
        SetGeometryShader( NULL );
        SetPixelShader( CompileShader( ps_4_0, pShader() ) );
                
        SetRasterizerState(NormalState);       
		SetDepthStencilState(ZDisabled, 0);
		SetBlendState(ContrastBlend, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xFFFFFFFF);
    }
}

/***************************** eof ***/

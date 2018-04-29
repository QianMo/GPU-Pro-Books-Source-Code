//------------------------------------------------------------------------------

void SkyboxVP(	float4 position	: POSITION,
				float2 normalUV	: TEXCOORD0,
				
				out float4 oPosition : POSITION,
				out float2 onormalUV : TEXCOORD0,

				uniform float4x4 worldViewProj

            )

{	
	oPosition = mul(worldViewProj, position);
	
	onormalUV = normalUV;

}

//---------------------------------------------------------------------------


void SkyboxFP(float2 normalUV      : TEXCOORD0,			
				out float4 color	: COLOR,	 
				uniform sampler2D albedo : register(s0)
)
{
	float4 alb = pow(tex2D(albedo,normalUV),2.2); //srgb to linear, sRGB state does not work in Ogre so we do our own conversion

	color = alb;

}

void SkyboxHDRFP(float2 normalUV      : TEXCOORD0,
	  			out float4 color	: COLOR,
				uniform sampler2D albedo : register(s0)

)
{

	float4 alb = tex2D(albedo,normalUV);
	
	color = alb;

}
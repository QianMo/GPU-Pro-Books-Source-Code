

//Global compress range is set by hand or defined by the maximum value in the textures [texturename]_1..5
//See baking script for compression.
#define COMPRESSRANGE 0.75 //set to 1 for a compress range of -1..1

//Global multipliers
#define GLOBAL_SCALE 2.5 //2.5066282 is sqrt2pi
#define GLOBAL_TEXTURE 2.5

//set LOD parameters here as in IrrNorm.material
#define LOWERINTERPDIST 50
#define UPPERINTERPDIST 90

#define OVERSQ2PI 0.39894228		
#define SQ32PI 0.69098830		
#define	SQ152PI	1.5450968

void IrrNormVP(	float4 position	: POSITION,
				float2 normalUV	: TEXCOORD0,
				float2 lightUV	: TEXCOORD1,
							
				out float4 oPosition : POSITION,
				out float2 onormalUV : TEXCOORD0,
				out float2 olightUV : TEXCOORD1,
				
				uniform float4x4 worldViewProj

            )

{	
	oPosition = mul(worldViewProj, position);
	
	onormalUV = normalUV;
	olightUV = lightUV;

}

void IrrNormInterpVP(	float4 position	: POSITION,
						float2 normalUV	: TEXCOORD0,
					float2 lightUV	: TEXCOORD1,
													
				out float4 oPosition : POSITION,
				out float2 onormalUV : TEXCOORD0,
				out float2 olightUV : TEXCOORD1,
				out float oworldDist : TEXCOORD2,
				
				uniform float4x4 worldViewProj,
				uniform float4x4 toWorld,
				uniform float3 camWorld			
            )

{	
	oPosition = mul(worldViewProj, position);
	
	onormalUV = normalUV;
	olightUV = lightUV;
	
	float4 p = mul(toWorld,position);
	p.xyz = p.xyz-camWorld.xyz;
	
	oworldDist = length(p);
	
}


//------------------------------------------------------------------------------

void IrrNormFP(	float2 normalUV      : TEXCOORD0,
				float2 lightUV       : TEXCOORD1,
				
				out float4 color	: COLOR,
		 
				uniform sampler2D albedo : register(s0),
				uniform sampler2D normal : register(s1),
				uniform sampler2D coeff0 : register(s2),
				uniform sampler2D coeff1 : register(s3),
				uniform sampler2D coeff2 : register(s4),
				uniform sampler2D coeff3 : register(s5),
				uniform sampler2D coeff4 : register(s6),
				uniform sampler2D coeff5 : register(s7)

)
{
		float4 alb = pow(tex2D(albedo,normalUV),2.2); //srgb to linear, sRGB state does not work in Ogre so we do our own conversion
		float3 norm = 2*tex2D(normal,normalUV)-1;
		
		//un-range compress all coefficients except the first one is build 
		//into formulation

		
		float3 Irr = pow(tex2D(coeff0,lightUV),2.2)*OVERSQ2PI //this should be done with sRGB lookup
					+(2*COMPRESSRANGE*tex2D(coeff1,lightUV)-COMPRESSRANGE)*SQ32PI*norm.y
					+(2*COMPRESSRANGE*tex2D(coeff2,lightUV)-COMPRESSRANGE)*SQ32PI*(1-norm.z)
					+(2*COMPRESSRANGE*tex2D(coeff3,lightUV)-COMPRESSRANGE)*SQ32PI*norm.x
					+(2*COMPRESSRANGE*tex2D(coeff4,lightUV)-COMPRESSRANGE)*SQ152PI*norm.y*norm.x
					+(2*COMPRESSRANGE*tex2D(coeff5,lightUV)-COMPRESSRANGE)*SQ152PI*0.5*(norm.x*norm.x -norm.y*norm.y); //0.5 as in equation
										
		color.rgb = GLOBAL_SCALE*Irr*alb.rgb*GLOBAL_TEXTURE;
		color.a = 1.0;				
}



void IrrNormNoAlbedoFP(	float2 normalUV      : TEXCOORD0,
				float2 lightUV       : TEXCOORD1,
				
				out float4 color	: COLOR,
		 
				uniform sampler2D normal : register(s0),
				uniform sampler2D coeff0 : register(s1),
				uniform sampler2D coeff1 : register(s2),
				uniform sampler2D coeff2 : register(s3),
				uniform sampler2D coeff3 : register(s4),
				uniform sampler2D coeff4 : register(s5),
				uniform sampler2D coeff5 : register(s6)

)
{
		float3 norm = 2*tex2D(normal,normalUV)-1;
		
		//un-range compress all coefficients except the first one is build 
		//into formulation
		
		float3 Irr = pow(tex2D(coeff0,lightUV),2.2)*OVERSQ2PI 
					+(2*COMPRESSRANGE*tex2D(coeff1,lightUV)-COMPRESSRANGE)*SQ32PI*norm.y
					+(2*COMPRESSRANGE*tex2D(coeff2,lightUV)-COMPRESSRANGE)*SQ32PI*(1-norm.z)
					+(2*COMPRESSRANGE*tex2D(coeff3,lightUV)-COMPRESSRANGE)*SQ32PI*norm.x
					+(2*COMPRESSRANGE*tex2D(coeff4,lightUV)-COMPRESSRANGE)*SQ152PI*norm.y*norm.x
					+(2*COMPRESSRANGE*tex2D(coeff5,lightUV)-COMPRESSRANGE)*SQ152PI*0.5*(norm.x*norm.x -norm.y*norm.y); //0.5 as in equation
										
		color.rgb = GLOBAL_SCALE*Irr;
		color.a = 1.0;				
}

void IrrNormLinFP(float2 normalUV      : TEXCOORD0,
				float2 lightUV       : TEXCOORD1,
				
				out float4 color	: COLOR,
		 
				uniform sampler2D albedo : register(s0),
				uniform sampler2D normal : register(s1),
				uniform sampler2D coeff0 : register(s2),
				uniform sampler2D coeff1 : register(s3),
				uniform sampler2D coeff2 : register(s4),
				uniform sampler2D coeff3 : register(s5)
)
{
		float4 alb = pow(tex2D(albedo,normalUV),2.2); //we do our own sRGB to linear, sRGB Gamma correction in the renderwindow must be set.
		float3 norm = 2*tex2D(normal,normalUV)-1;
		
		//un-range compress all coefficients except the first one is build 
		//into formulation

		float3 Irr = pow(tex2D(coeff0,lightUV),2.2)*OVERSQ2PI 
					+(2*COMPRESSRANGE*tex2D(coeff1,lightUV)-COMPRESSRANGE)*SQ32PI*norm.y
					+(2*COMPRESSRANGE*tex2D(coeff2,lightUV)-COMPRESSRANGE)*SQ32PI*(1-norm.z)
					+(2*COMPRESSRANGE*tex2D(coeff3,lightUV)-COMPRESSRANGE)*SQ32PI*norm.x;
										
		color.rgb = GLOBAL_SCALE*Irr*alb.rgb*GLOBAL_TEXTURE;	
		color.a = 1.0;			
}

void IrrNormLinNoAlbedoFP(float2 normalUV      : TEXCOORD0,
				float2 lightUV       : TEXCOORD1,
				
				out float4 color	: COLOR,
		 
				uniform sampler2D normal : register(s0),
				uniform sampler2D coeff0 : register(s1),
				uniform sampler2D coeff1 : register(s2),
				uniform sampler2D coeff2 : register(s3),
				uniform sampler2D coeff3 : register(s4)
)
{
		float3 norm = 2*tex2D(normal,normalUV)-1;
		
		//un-range compress all coefficients except the first one is build 
		//into formulation
		
		float3 Irr = pow(tex2D(coeff0,lightUV),2.2)*OVERSQ2PI 
					+(2*COMPRESSRANGE*tex2D(coeff1,lightUV)-COMPRESSRANGE)*SQ32PI*norm.y
					+(2*COMPRESSRANGE*tex2D(coeff2,lightUV)-COMPRESSRANGE)*SQ32PI*(1-norm.z)
					+(2*COMPRESSRANGE*tex2D(coeff3,lightUV)-COMPRESSRANGE)*SQ32PI*norm.x;

										
		color.rgb = GLOBAL_SCALE*Irr;
		color.a = 1.0;		
}

void IrrNormConstFP(float2 normalUV      : TEXCOORD0,
				float2 lightUV       : TEXCOORD1,
				
				out float4 color	: COLOR,
		 
				uniform sampler2D albedo : register(s0),
				uniform sampler2D coeff0 : register(s1)
)
{
		float4 alb = pow(tex2D(albedo,normalUV),2.2); //we do our own sRGB to linear, sRGB Gamma correction in the renderwindow must be set.
		
		//un-range compress all coefficients except the first one is build 
		//into formulation

		float3 Irr = pow(tex2D(coeff0,lightUV),2.2)*OVERSQ2PI;
					
		color.rgb = GLOBAL_SCALE*Irr*alb.rgb*GLOBAL_TEXTURE;
		color.a = 1.0;			
}

void IrrNormConstNoAlbedoFP(float2 normalUV      : TEXCOORD0,
				float2 lightUV       : TEXCOORD1,
				
				out float4 color	: COLOR,
				
				uniform sampler2D coeff0 : register(s0)
)
{
		
		
		//un-range compress all coefficients except the first one is build 
		//into formulation

		float3 Irr = pow(tex2D(coeff0,lightUV),2.2)*OVERSQ2PI;
									
		color.rgb =GLOBAL_SCALE*Irr;
		color.a = 1.0;		
}

//---------------------------------------------------------------------------


void IrrNormInterpFP(float2 normalUV	: TEXCOORD0,
					 float2 lightUV		: TEXCOORD1,
					 float worldDist	: TEXCOORD2,
				
					out float4 color	: COLOR,
		 
				uniform sampler2D albedo : register(s0),
				uniform sampler2D normal : register(s1),
				uniform sampler2D coeff0 : register(s2),
				uniform sampler2D coeff1 : register(s3),
				uniform sampler2D coeff2 : register(s4),
				uniform sampler2D coeff3 : register(s5) 				
)
{
		float4 alb = pow(tex2D(albedo,normalUV),2.2); //we do our own sRGB to linear, sRGB Gamma correction in the renderwindow must be set.
		float3 norm = 2*tex2D(normal,normalUV)-1;
		
		//un-range compress all coefficients except the first one is build 
		//into formulation
		
		float weight = max(0,((worldDist-LOWERINTERPDIST)/(UPPERINTERPDIST-LOWERINTERPDIST)));
		
		float3 Irr =  pow(tex2D(coeff0,lightUV),2.2)*OVERSQ2PI 
		
		+(1-weight)*(
					(2*COMPRESSRANGE*tex2D(coeff1,lightUV)-COMPRESSRANGE)*SQ32PI*norm.y
					+(2*COMPRESSRANGE*tex2D(coeff2,lightUV)-COMPRESSRANGE)*SQ32PI*(1-norm.z)
					+(2*COMPRESSRANGE*tex2D(coeff3,lightUV)-COMPRESSRANGE)*SQ32PI*norm.x

					);
										
		color.rgb = GLOBAL_SCALE*Irr*alb.rgb*GLOBAL_TEXTURE;	
		color.a = 1.0;		
		
}

void IrrNormInterpNoAlbedoFP(float2 normalUV	: TEXCOORD0,
					 float2 lightUV		: TEXCOORD1,
					 float worldDist	: TEXCOORD2,
				
					out float4 color	: COLOR,
		 

				uniform sampler2D normal : register(s0),
				uniform sampler2D coeff0 : register(s1),
				uniform sampler2D coeff1 : register(s2),
				uniform sampler2D coeff2 : register(s3),
				uniform sampler2D coeff3 : register(s4)
				

				
)
{
		float3 norm = 2*tex2D(normal,normalUV)-1;
		
		//un-range compress all coefficients except the first one is build 
		//into formulation
	
		float weight = max(0,((worldDist-LOWERINTERPDIST)/(UPPERINTERPDIST-LOWERINTERPDIST)));
		
		float3 Irr = pow(tex2D(coeff0,lightUV),2.2)*OVERSQ2PI 
		
		+(1-weight)*(
					(2*COMPRESSRANGE*tex2D(coeff1,lightUV)-COMPRESSRANGE)*SQ32PI*norm.y
					+(2*COMPRESSRANGE*tex2D(coeff2,lightUV)-COMPRESSRANGE)*SQ32PI*(1-norm.z)
					+(2*COMPRESSRANGE*tex2D(coeff3,lightUV)-COMPRESSRANGE)*SQ32PI*norm.x
					);
										
		color.rgb = GLOBAL_SCALE*Irr;	
		color.a = 1.0;		
		
		//color.r = (1-weight);
		

	
}


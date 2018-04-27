#define MAX_LIGHTS 6

shared Texture2D 	NormalTex;
shared Texture2D 	DiffuseTex;
shared Texture2D	SpecularTex;
shared TextureCube	ReflectTex;
shared Buffer<float4> FontBuf;

shared Texture2D	WeightTex;

shared Texture2D	NormalTex1;
shared Texture2D	DiffuseTex1;
shared Texture2D	SpecularTex1;

shared Texture2D	RefractTex;

shared cbuffer PerObject
{
	float4x4 matWVP;
	float4x3 matW;
	float4x3 matW_NS;
	float4x3 matWV_NS;

	float4x4 matCubeMapWVP[6];
	uint cubeFaceIndices[6];


	// for correct lighting of scaled objects

	float3	gScale;
	float3	gInvScale;

	// in object space(OS). NOTE: regardless of scale
	float3 gCamPos_OS; 

	float 	gSpecPower;

	float3	gReflection;
	float	gRefBlur;

	float3 	gAmbient;
	float	gSelfIlluminate;

	float4	gDiffColor; // for non textured objects

	// lights
	float3 	gLightPos_OS[MAX_LIGHTS];

	float3 	gLightDir_OS[MAX_LIGHTS];
	float	gLightAttenuation[MAX_LIGHTS];

	float3 	gLightColor[MAX_LIGHTS];
	float3	gLightSpecular[MAX_LIGHTS]; // premultiplied objspec * lightcolor
	// end of lights

	float	gFlowTime; // for water

};

shared cbuffer Bones
{
	float4x3 matArrBones[128];
};

shared cbuffer DieLetterz
{
	float4 letterPoses[512];
};

struct Light
{
	float3 dir;
	float3 diffuse;
	float3 specular;
	float attenuation;
};

struct LightComponents
{
	float3 diffuse;
	float3 specular;
};

void Init( inout LightComponents lc )
{
	lc.diffuse = 0;
	lc.specular = 0;
}

void FillLightAttribs( inout Light o, int i )
{
	o.diffuse 	= gLightColor[i];
	o.specular	= gLightSpecular[i];
}

Light GetDirLight( int i )
{
	Light o;

	o.dir 			= gLightDir_OS[i];
	o.attenuation	= 1;

	FillLightAttribs( o, i );

	return o;
}

Light GetPointLight( int i, float3 objPos )
{
	Light o;

	float3 ldir 	= gLightPos_OS[i] - objPos;
	float len 		= length( ldir );

	o.dir 			= ldir / len;

	o.attenuation   = saturate( gLightAttenuation[i] / len / len );
	

	FillLightAttribs( o, i );

	return o;
}

// if specular is baked into texture then this is called
Light UpdateLight( Light l, float3 specular )
{
	l.specular = specular * l.diffuse;
	return l;
}

void CalculateLight( inout LightComponents o, Light l, float3 norm, float3 view, float specPower )
{
	float d = dot( norm, l.dir );
	float s = pow( saturate( dot( reflect( -l.dir, norm ), view ) ), specPower );

	float2( d, s ) *= l.attenuation;

	if( d <= 0 ) s = 0;

	o.diffuse 		+=	saturate(d) * l.diffuse;
	o.specular 		+=	s * l.specular;
}
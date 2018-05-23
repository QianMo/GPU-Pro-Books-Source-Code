#include "..\\..\\MaterialAging\\MaterialPacking.h"
#include "random.hlsl"
#include "ColorPacking.hlsl"


//-------------------------------------------------------------------------
// Shader Resources
// Not all of them are available in all passes.
// But improve readability each texture slot is named.
//-------------------------------------------------------------------------

Texture2D Tex_Material			: register( t0 );
Texture2D Tex_Material2			: register( t1 );
Texture2D Tex_SurfaceNormal		: register( t2 );
Texture2D Tex_SurfaceHeight		: register( t3 );
Texture2D Tex_GeometryNormal	: register( t4 );
Texture2D Tex_GeometryTangent	: register( t5 );
Texture2D Tex_SplatSize			: register( t6 );
Texture2D Tex_OriginalColor		: register( t7 );
Texture2D Tex_OriginalHeight	: register( t8 );
Texture2D Tex_OriginalMaterial	: register( t9 );
Texture2D Tex_OriginalMaterial2 : register( t10 );
Texture2D Tex_MaterialProbes	: register( t11 );
Texture2D<uint> Tex_RandomSeeds	: register( t12 );
Texture2D Tex_Unspecified		: register( t13 );  // for dilatation and debug purposes


//TODO
SamplerState samplerLinear	: register( s0 );
SamplerState matSampler		: register( s4 );

//-------------------------------------------------------------------------
// Effect Properties
//-------------------------------------------------------------------------
float SplatScale = 1.0f;

float MinSplatSize = 0.005;			// size in texturespace
float MaxSplatSize = 0.01;			// size in texturespace


float Gravity = 9.81f;				// Gravity in y direction.

float minimum_velocity = 0.05f;	// if particle get slower than this they get killed.
int SimulationIterations;
float TangentialFloatCorrectness = 0.5f;	// floating particles usually act only tangentially. To speed things up, we can avoid this. This ratio (in 0,1) says: 0=no deceleration, 1=only tangential component



float BounceSpeedLoss = 0.6f;
float FloatSpeedLoss = 0.7f;

float BounceProbability = 0.1f;
float FloatProbability = 0.9f;

float PhongLobeExponent_Float = 50000;
float PhongLobeExponent_Bounce = 50000;

// define how much a gammaton takes away from the surface while hitting it
// this amount is then carried by the gammaton
float4 g_volatileness = float4(
		0.2,	// Water
		0.1,	// Dirt
		0.0,	// Metal
		0.0);	// Wood
float4 g_volatileness2 = float4(
		0.01,	// Organic
		0.01,	// Rust
		0.0,	// Stone
		0);		// Dummy

float g_pickUpRatio = 0.5f;	//TODO.. i think it doesn't work 0.5

float Material_AddFactor = 1.0;	//TODO remove
float Material_SubFactor = 0.5;	//TODO remove
float Gammaton_AddFactor = 0.5;	//TODO remove
float Gammaton_SubFactor = 1.0;	//TODO remove

/*	Composition Parameters for each material.
The user is able to influence diffuse color, specular value and height, depending on the material amount

CompositionColorParams.x	=	diffuse base strength
CompositionColorParams.y	=	diffuse random variation
CompositionColorParams.z	=	specular base strength
CompositionColorParams.w	=	specular random variation	

CompositionHeightParams.x	=	height base strength
CompositionHeightParams.y	=	height random variation
*/
#define EMaterialType_COUNT 8
float4 CompositionColorParams[EMaterialType_COUNT];
float4 CompositionHeightParams[EMaterialType_COUNT];


//-------------------------------------------------------------------------
// Some helpers
//-------------------------------------------------------------------------

// a MaterialProperty consists of two float4 vetors
// this will properly change for more material types
// for now these helpers will allow to convert between formats
void materialToFloat4(MaterialProperty mat, out float4 a, out float4 b)
{
	a = float4(mat.Water, mat.Dirt, mat.Metal, mat.Wood);
	b = float4(mat.Organic, mat.Rust, mat.Stone, mat.Dummy);
}

MaterialProperty materialFromFloat4(float4 a, float4 b)
{
	MaterialProperty mat;
	mat.Water = a.x; mat.Dirt = a.y; mat.Metal = a.z; mat.Wood = a.w;
	mat.Organic = b.x; mat.Rust = b.y; mat.Stone = b.z;	mat.Dummy = b.w;
	return mat;
}

//-------------------------------------------------------------------------
// Rule Interfaces
//-------------------------------------------------------------------------

#define NUM_RULES 4

struct RuleParams
{
	float Chance;
	float Speed;

	// optional: additional rule parameters 
};

RuleParams g_RuleParamters[NUM_RULES];	

interface IRule
{
	MaterialProperty Apply(MaterialProperty mat , RuleParams p);
};

// metal + water = rust
class IRust : IRule
{
	MaterialProperty Apply(MaterialProperty mat , RuleParams p)
	{
		// amount of new rust
		float dRust = min(mat.Water, mat.Metal) * 0.1;

		// add the new rust to the material
		mat.Rust += dRust;
		return mat;
	}
};


// wood + water = organic + dirt
class IDecay : IRule
{
	MaterialProperty Apply(MaterialProperty mat , RuleParams p)
	{
		// amount
		float delta = min(mat.Water, mat.Wood) * 0.1;

		// add the new amounts to the material
		mat.Organic += delta;
		mat.Dirt += delta; // create as much dirt as organic. This is properly the designers choice.
		return mat;
	}
};

// dirt + water = organic
class IGrow : IRule
{
	MaterialProperty Apply(MaterialProperty mat , RuleParams p)
	{
		// amount
		float dOrganic = min(mat.Water, mat.Dirt) * 0.1;

		// add the new organic material to the material
		mat.Organic += dOrganic;
		return mat;
	}
};

// water + time + (later maybe light) = no water 
class IEvaporation : IRule
{
	MaterialProperty Apply(MaterialProperty mat , RuleParams p)
	{
		// amount
		float delta = -1.0 * mat.Water * 0.1;

		// add the new amounts to the material
		mat.Water += delta;
		return mat;
	}
};

// class instances
IRust pRust;
IDecay pDecay;
IGrow pGrow;
IEvaporation pEvaporation;

// instance interface 
IRule pRules[NUM_RULES] = {pRust, pDecay, pGrow, pEvaporation};

//-------------------------------------------------------------------------
// Stage Interfaces
//-------------------------------------------------------------------------

struct GAMMATON
{
	float3 Velocity : VELOCITY;
	int Flags : FLAGS;
	float2 TexCoord : TEXCOORD0;
	uint2 CarriedMaterial : CARRIED_MATERIAL;
	uint Seed : RANDOM_SEED;
	uint3 Dummy : DUMMY;
};

struct ATLAS_POSITION
{
	float4 Position : SV_POSITION;
};

struct GSPS_INPUT
{
	float4 Position : SV_POSITION;
	float2 TexCoord : TEXCOORD0;
	float2 TexCoordSplat : TEXCOORD1;
	int Flags : TEXCOORD2;
	uint2 CarriedMaterial : CARRIED_MATERIAL;
	uint3 Dummy : DUMMY;
};

struct PS_OUTPUT_SurfaceUpdate
{
	float4 Material : SV_TARGET0;		// material after transfer with gammatons
	float4 Material2 : SV_TARGET1;		// material after transfer with gammatons
};

struct PS_OUTPUT_AgingProcess
{
	float4 Material : SV_TARGET0;		// material after application of the rules
	float4 Material2 : SV_TARGET1;		// material after application of the rules
	uint4 Noise : SV_Target2;			// random seeds (ping pong)
};

struct PS_OUTPUT_CompositionPhase1
{
	float4 Color : SV_TARGET0;			// color of the aged material (baked)
	float4 Height : SV_TARGET1;			// height map of the aged material (baked)
};

struct PS_OUTPUT_CompositionPhase2
{
	float4 Normal : SV_TARGET0;			// normal map of the aged material in tangent space (baked)
};

struct PS_OUTPUT_Remeshing
{
	float4 Normal : SV_TARGET0;			// surface normals in world space
};

//--------------------------------------------------------------------------------------
// Vertex Shader
//--------------------------------------------------------------------------------------
ATLAS_POSITION VS_PassThrough(float3 Position : POSITION0)
{
	ATLAS_POSITION output;
	output.Position = float4(Position, 1);  
	return output;
}

GSPS_INPUT VS_ToAtlasCoords(GAMMATON input)
{
	GSPS_INPUT output;
	
	float2 pos2D = input.TexCoord * 2.0 - 1.0;
	pos2D.y *= -1;
	output.Position = float4(pos2D, 0, 1);  
	output.TexCoord = input.TexCoord;
	output.TexCoordSplat = float2(0,0);
	output.Flags = input.Flags;
	output.CarriedMaterial = input.CarriedMaterial;
	output.Dummy = input.Dummy;

	return output;
}

// Vertex Shader, der die Gammaton Hit daten direkt an den Geometry Shader weiterreicht (TODO: Stream out aus VS)
GAMMATON VS_PassThrough_Gammaton(GAMMATON input) {
	return input;
}

//--------------------------------------------------------------------------------------
// Geometry Shader
//--------------------------------------------------------------------------------------

// Caluclation of the amount of material to tranfer
// this amount is removed from the surface and added to gammaton
// Note that the values can be negative in which case material is dropped by the gammaton
void transferRate(float4 sMat, float4 sMat2, float4 gMat, float4 gMat2, out float4 delta, out float4 delta2) 
{
	delta = sMat * ( 1.0 - g_pickUpRatio ) - gMat * g_pickUpRatio;
	delta *= g_volatileness;

	delta2 = sMat2 * ( 1.0 - g_pickUpRatio ) - gMat2 * g_pickUpRatio;
	delta2 *= g_volatileness2; 
}

[maxvertexcount(4)]
void GS_SurfaceUpdate( point GSPS_INPUT input[1], inout TriangleStream<GSPS_INPUT> TriStream )
{
	// do not splat gammatons that are not on the surface
	bool hit = IS_HIT(input[0].Flags);
	if (!hit) return;

	// scale each texel depending on the world space area they cover
	float2 texelScale = saturate(Tex_SplatSize.SampleLevel( samplerLinear, input[0].TexCoord, 0).xy);

	float SplatSizeH = max(MinSplatSize, (MaxSplatSize * SplatScale / (texelScale.x))); 
	float SplatSizeV = max(MinSplatSize, (MaxSplatSize * SplatScale / (texelScale.y))); 
	
	float4 center = input[0].Position;
	float2 centerTex = input[0].TexCoord;

	float4 offsetH = float4(SplatSizeH, 0, 0, 0);
	float4 offsetV = float4(0, SplatSizeV, 0, 0);

	float2 offsetTexH = float2(SplatSizeH*0.5, 0);
	float2 offsetTexV = float2(0, -SplatSizeV*0.5);

	//	2	4
	//	  \
	//	1	3

	GSPS_INPUT outVert;
	outVert.Flags = input[0].Flags;
	outVert.CarriedMaterial = input[0].CarriedMaterial;
	outVert.Dummy = input[0].Dummy;

	outVert.Position = center - offsetH - offsetV;
	outVert.TexCoord = centerTex - offsetTexH - offsetTexV;	
	outVert.TexCoordSplat = float2(0,0);
	TriStream.Append( outVert );

	outVert.Position = center - offsetH + offsetV;
	outVert.TexCoord = centerTex - offsetTexH + offsetTexV;
	outVert.TexCoordSplat = float2(0,1);
	TriStream.Append( outVert );

	outVert.Position = center + offsetH - offsetV;
	outVert.TexCoord = centerTex + offsetTexH - offsetTexV;
	outVert.TexCoordSplat = float2(1,0);
	TriStream.Append( outVert );

	outVert.Position = center + offsetH + offsetV;
	outVert.TexCoord = centerTex + offsetTexH + offsetTexV;
	outVert.TexCoordSplat = float2(1,1);
	TriStream.Append( outVert );
	
	TriStream.RestartStrip();
}

void PickUpMaterial(inout GAMMATON gammaton)
{
	// get the material on the surface
	float4 surfaceMat = Tex_Material.SampleLevel( samplerLinear, gammaton.TexCoord, 0);
	float4 surfaceMat2 = Tex_Material2.SampleLevel( samplerLinear, gammaton.TexCoord, 0);

	// get the material carried by the gammaton
	float4 gammatonMat;
	float4 gammatonMat2;
	materialToFloat4(unpackMaterial(gammaton.CarriedMaterial), gammatonMat, gammatonMat2);

	// calulate the amount of material that is transfered between surface and gammaton
	float4 transfered, transfered2;
	transferRate(surfaceMat, surfaceMat2, gammatonMat, gammatonMat2, transfered, transfered2);

	// the transfered amount is added to the current amount the gammaton carries
	// not that on the "other side", the surface update, the same amount is removed
	// also we keep the value within the allowed range
	gammatonMat = saturate(gammatonMat + transfered);
	gammatonMat2 = saturate(gammatonMat2 + transfered2);

	// pack and store
	gammaton.CarriedMaterial = packMaterial(materialFromFloat4(gammatonMat, gammatonMat2));
}

// Handles the bouncing and floating collision responses
void PhongLobeBounce(inout GAMMATON output)
{
	// Read surface normal
	float3 normal = normalize(Tex_SurfaceNormal.SampleLevel( samplerLinear, output.TexCoord, 0).xyz * 2 - 1);

	float speed = length(output.Velocity);		// compute speed
	float3 velocity = output.Velocity / speed;	// Unit vector for velocity direction

	// reflect velocity at normal
	float3 outgoing = reflect(velocity, normal);

	if (IS_BOUNCE(output.Flags))
	{
		// Get geometric normal or even better the face normal
		float3 faceNormal = normalize(Tex_GeometryNormal.SampleLevel( samplerLinear, output.TexCoord, 0).xyz * 2 - 1);
		// Randomize the bounce direction by carrying out rejection sampling to avoid penetration of particles into objects
		outgoing = normalize(samplePhongLobeSafe(PhongLobeExponent_Bounce, outgoing, faceNormal, output.Seed));
		// Decelerate
		speed *= BounceSpeedLoss;
	}
	else
	{
		outgoing = normalize(samplePhongLobe(PhongLobeExponent_Float, outgoing, output.Seed));
		// Decelerate
		speed *= FloatSpeedLoss;

		// project outgoing direction to tangent plane
		outgoing -= dot(normal, outgoing) * normal;
		
		// maintain part of speed (artist parameter for faster spreading)
		outgoing = lerp(normalize(outgoing), outgoing, TangentialFloatCorrectness);

		// move slightly in direction of gravity -> this makes the particle grab for the next surface
		outgoing -= normal * 1.0;   // h = 1

		// after subtraction of normal component (used to pull back to the surface) we normalize the vector once more (to maintain the speed)
		outgoing = normalize(outgoing);
	}
			
	
	// Compose new velocity
	output.Velocity = outgoing * speed;

	// If too slow, remove particle (set inactive)
	if (speed < minimum_velocity && normal.y > 0.8 )
		SET_DEAD(output.Flags);
	else 
		SET_ALIVE(output.Flags);
}

// Geometry shader that updates the gammatons
[maxvertexcount(1)]
void GS_GammatonUpdate( point GAMMATON input[1], inout PointStream<GAMMATON> PntStream )
{
	GAMMATON output = input[0];
	if (IS_MIDAIR(input[0].Flags) || IS_DEAD(input[0].Flags))		// gammaton is still traveling or left the domain
	{
		PntStream.Append(output);
		return;
	}

	PickUpMaterial(output);

	// Russian roulette
	float roulette = rnd(output.Seed);	// Random value in [0,1)
	if (roulette < BounceProbability)
	{
		SET_BOUNCE(output.Flags); // set bounce
		PhongLobeBounce(output);  // handles bounce
	}
	else if (roulette < BounceProbability + FloatProbability)
	{
		SET_FLOAT(output.Flags); // set float
		PhongLobeBounce(output); // handles float
	}
	else // Absorbed
		SET_DEAD(output.Flags);
	
	PntStream.Append(output);
}

//--------------------------------------------------------------------------------------
// Pixel Shader
//--------------------------------------------------------------------------------------

// some of the functions that are not directly involved in the material aging
// were removed from the effect to increate readablity
#include "HelperShader.hlsl"

// Material transfer on surface side
// therefore a small quad is rendered
PS_OUTPUT_SurfaceUpdate PS_SurfaceUpdate( GSPS_INPUT input)
{
	PS_OUTPUT_SurfaceUpdate output = (PS_OUTPUT_SurfaceUpdate) 0;

	// get the material on the surface
	float4 surfaceMat = Tex_Material.Load(input.Position.xyz);
	float4 surfaceMat2 = Tex_Material2.Load(input.Position.xyz);

	// get the material carried by the gammaton
	float4 gammatonMat;
	float4 gammatonMat2;
	materialToFloat4(unpackMaterial(input.CarriedMaterial), gammatonMat, gammatonMat2);

	// calulate the amount of material that is transfered between surface and gammaton
	float4 transfered, transfered2;
	transferRate(surfaceMat, surfaceMat2, gammatonMat, gammatonMat2, transfered, transfered2);

	// to get a visually smoother result we use a distance kernel
	float strength = length(input.TexCoordSplat * 2.0 - 1.0);
	strength = saturate(1.0 - strength);
	strength = strength * strength; // quadric fall of

	// the transfered amount is removed to the current amount on the surface
	// not that on the "other side", the gammaton update, the same amount is added
	// also we keep the value within the allowed range
	surfaceMat = saturate(surfaceMat - transfered * strength);
	surfaceMat2 = saturate(surfaceMat2 - transfered2 * strength);

	// since there is no real influence on geometry we don't want "holes" in our surface
	// to safe the geometry from loosing their entire material we make sure
	// that the original material is always present
	output.Material = max(surfaceMat, Tex_OriginalMaterial.Load(input.Position.xyz));
	output.Material2 = max(surfaceMat2, Tex_OriginalMaterial2.Load(input.Position.xyz));

	return output;
}

// the application of the aging rules
PS_OUTPUT_AgingProcess PS_AgingProcess( ATLAS_POSITION input)
{
	PS_OUTPUT_AgingProcess output = (PS_OUTPUT_AgingProcess) 0;

	// read current material on the surface
	MaterialProperty mat = CreateMaterialPropertyFromFloat4( Tex_Material.Load(input.Position.xyz), 
															 Tex_Material2.Load(input.Position.xyz));	
	// draw a random number 
	uint randomSeed = Tex_RandomSeeds.Load(input.Position.xyz);

	[unroll]
	for(int r=0; r<NUM_RULES; r++)
	{
		// do not apply each rule every frame to allow higher temporal resolution
		// this will give the user more control about how strong the effect of a rule is
		// this is need due to the fact that we have only 256 discrete material amount steps
		if (rnd(randomSeed) < g_RuleParamters[r].Chance )
		{
			mat = pRules[r].Apply( mat , g_RuleParamters[r]);
		}
	}

	// write the aged material into the atlas (via ping pong)
	materialToFloat4(mat, output.Material, output.Material2);
	output.Noise = randomSeed.xxxx;
	return output;
}


// defines how transported material influences the final diffuse color of the aged scene
float3 ComposeDiffuseColor(float3 originalDiffuseColor, MaterialProperty deltaMaterial, inout uint randomSeed)
{
	// random offset inside the material probe texture
	float2 tileUV = float2(0.333, 0.5) * rnd2(randomSeed);

	// accumulate the influence of each material on the current diffuse color 
	//																											BaseStrength						Random Variation
	float4 color = float4(0,0,0,0);
	color += deltaMaterial.Water * float4(-1.0f,-1.0f,-1.0f, 1.0f)										*   (CompositionColorParams[0].x + CompositionColorParams[0].y * rnd(randomSeed));
	color += deltaMaterial.Dirt * Tex_MaterialProbes.Sample(matSampler, tileUV)							*   (CompositionColorParams[1].x + CompositionColorParams[1].y * rnd(randomSeed));
	color += deltaMaterial.Metal * Tex_MaterialProbes.Sample(matSampler, tileUV + float2(0.333, 0))		*   (CompositionColorParams[2].x + CompositionColorParams[2].y * rnd(randomSeed));
	color += deltaMaterial.Wood * Tex_MaterialProbes.Sample(matSampler, tileUV + float2(0.666, 0))		*   (CompositionColorParams[3].x + CompositionColorParams[3].y * rnd(randomSeed));
	color += deltaMaterial.Organic * Tex_MaterialProbes.Sample(matSampler, tileUV + float2(0.0, 0.5))	*   (CompositionColorParams[4].x + CompositionColorParams[4].y * rnd(randomSeed));
	color += deltaMaterial.Rust * Tex_MaterialProbes.Sample(matSampler, tileUV + float2(0.333, 0.5))	*   (CompositionColorParams[5].x + CompositionColorParams[5].y * rnd(randomSeed));
	color += deltaMaterial.Stone * Tex_MaterialProbes.Sample(matSampler, tileUV + float2(0.666, 0.5))	*   (CompositionColorParams[6].x + CompositionColorParams[6].y * rnd(randomSeed));
	color += deltaMaterial.Dummy * float4(0,0,0,0)														*   (CompositionColorParams[7].x + CompositionColorParams[7].y * rnd(randomSeed));

	// some kind of normalizing the result color, because in general the weights do not sum up to 1
	// note that a value below 1 indicates that there is not much transported material
	if(color.a > 1)
		color /= color.a;

	// combine both colors
	// if there is a lot of deposed material then we want to see that
	// if not, the original color should be visible 
	return lerp(originalDiffuseColor, color.rgb, color.a); // optional: use smoothstep(0.0, 1.0, color.a ) as interpolant
}


// estimate specular value of the final aged material
float ComposeShininess(float originalShininess, MaterialProperty deltaMaterial, inout uint randomSeed)
{
	float estimation = originalShininess;

	// apply the influence of each material 
	//										BaseStrength						Random Variation
	estimation += deltaMaterial.Water	* (CompositionColorParams[0].z + CompositionColorParams[0].w * rnd(randomSeed));
	estimation += deltaMaterial.Dirt	* (CompositionColorParams[1].z + CompositionColorParams[1].w * rnd(randomSeed));
	estimation += deltaMaterial.Metal	* (CompositionColorParams[2].z + CompositionColorParams[2].w * rnd(randomSeed));
	estimation += deltaMaterial.Wood	* (CompositionColorParams[3].z + CompositionColorParams[3].w * rnd(randomSeed));
	estimation += deltaMaterial.Organic	* (CompositionColorParams[4].z + CompositionColorParams[4].w * rnd(randomSeed));
	estimation += deltaMaterial.Rust	* (CompositionColorParams[5].z + CompositionColorParams[5].w * rnd(randomSeed));
	estimation += deltaMaterial.Stone	* (CompositionColorParams[6].z + CompositionColorParams[6].w * rnd(randomSeed));
	estimation += deltaMaterial.Dummy	* (CompositionColorParams[7].z + CompositionColorParams[7].w * rnd(randomSeed));

	return estimation;
}

// estimate the hight of aged surface
// these heights are later used to estimate the surface normals (e.g., for normal mapping)
float ComposeHeight(int3 atlasCoord, MaterialProperty originalMat, MaterialProperty deltaMat, inout uint randomSeed)
{
	// load original height
	float height = Tex_OriginalHeight.Load(atlasCoord).x;

	// apply the influence of each material 
	//								BaseStrength						Random Variation
	height += deltaMat.Water	* (CompositionHeightParams[0].x + CompositionHeightParams[0].y * rnd(randomSeed));
	height += deltaMat.Dirt		* (CompositionHeightParams[1].x + CompositionHeightParams[1].y * rnd(randomSeed));
	height += deltaMat.Metal	* (CompositionHeightParams[2].x + CompositionHeightParams[2].y * rnd(randomSeed));
	height += deltaMat.Wood		* (CompositionHeightParams[3].x + CompositionHeightParams[3].y * rnd(randomSeed));
	height += deltaMat.Organic	* (CompositionHeightParams[4].x + CompositionHeightParams[4].y * rnd(randomSeed));
	height += deltaMat.Rust		* (CompositionHeightParams[5].x + CompositionHeightParams[5].y * rnd(randomSeed));
	height += deltaMat.Stone	* (CompositionHeightParams[6].x + CompositionHeightParams[6].y * rnd(randomSeed));
	height += deltaMat.Dummy	* (CompositionHeightParams[7].x + CompositionHeightParams[7].y * rnd(randomSeed));

	return height;
}

// Generate final aged Diffuse Normal and Height map
// from original material properties, textures and the current simulated materials
PS_OUTPUT_CompositionPhase1 PS_CompositionPhase1( ATLAS_POSITION input)
{
	int3 atlasCoord = int3(input.Position.xy, 0);

	PS_OUTPUT_CompositionPhase1 output = (PS_OUTPUT_CompositionPhase1) 0;

	// draw random seed for this texel
	uint randomSeed = Tex_RandomSeeds.Load(atlasCoord);

	// load and unpack original diffuse color and shininess
	float3 originalDiffuseColor;
	float originalShininess;
	UnpackColor(Tex_OriginalColor.Load(atlasCoord), originalDiffuseColor, originalShininess);

	// materials
	MaterialProperty surfaceMat = CreateMaterialPropertyFromFloat4(Tex_Material.Load(atlasCoord), Tex_Material2.Load(atlasCoord));
	MaterialProperty originalMat = CreateMaterialPropertyFromFloat4(Tex_OriginalMaterial.Load(atlasCoord), Tex_OriginalMaterial2.Load(atlasCoord));

	// change of the material amounts over the simulation time
	MaterialProperty deltaMat = Substract(surfaceMat, originalMat);

	// calculate the new material color
	float3 estimatedDiffuse = ComposeDiffuseColor(originalDiffuseColor, deltaMat, randomSeed);

	// calculate new specular value
	float estimatedShininess = ComposeShininess(originalShininess, deltaMat, randomSeed);

	// calculate height 
	float estimatedHeight = ComposeHeight(atlasCoord, originalMat, deltaMat, randomSeed);

	// ignore atlas areas with not geometry
	float onSurface = any(originalDiffuseColor);
	output.Color = PackColor(estimatedDiffuse, estimatedShininess) * onSurface;
	output.Height = float4( estimatedHeight.xxx * onSurface, 1);
	return output;
}

PS_OUTPUT_CompositionPhase2 PS_CompositionPhase2( ATLAS_POSITION input)
{
	PS_OUTPUT_CompositionPhase2 output = (PS_OUTPUT_CompositionPhase2)0;

	int3 atlasCoord = int3(input.Position.xy, 0);

	const int offset = 1;

	// estimate gradiant by forwards differentiation
	float center = Tex_SurfaceHeight.Load(atlasCoord).x;
	float right = Tex_SurfaceHeight.Load(atlasCoord + int3(offset , 0, 0)).x;
	float up = Tex_SurfaceHeight.Load(atlasCoord + int3(0 , -offset, 0)).x;
	
	float dzToDx = (right - center) / offset;
	float dzToDy = (up - center) / offset; 

	float3 estimatedNormal = normalize(float3(-dzToDx, -dzToDy, 1));	// simplified for cross(normalize(float3(1,0,dzToDx)), normalize(float3(0,1,dzToDy)))

	output.Normal = float4( estimatedNormal * 0.5 + 0.5, 1);
	return output;
}

// in this third step  we are recalculating surface normals in world space for gammaton reflection calculation
PS_OUTPUT_Remeshing PS_CompositionPhase3( ATLAS_POSITION input)
{
	PS_OUTPUT_Remeshing output = (PS_OUTPUT_Remeshing) 0;

	int3 atlasCoord = int3(input.Position.xy, 0);

	// build the world space tangent frame at the current position
	float4 tangentTexValue = Tex_GeometryTangent.Load(int3(input.Position.xy, 0));
	float3 Nn = Tex_GeometryNormal.Load(int3(input.Position.xy, 0)).xyz  * 2.0 - 1.0; // stored normalized
	float3 Tn = tangentTexValue.xyz * 2.0 - 1.0; // stored normalized
	float3 Bn = cross(Nn, Tn) * tangentTexValue.w;
	Tn = cross(Nn, -Bn);

	// set up TBN matrix to transform from tangent to world space
	float3x3 invTBN = float3x3(Tn, Bn, Nn);

	// load the previously estimated surface normal in tangent space
	float3 surfaceNormal = Tex_SurfaceNormal.Load(int3(input.Position.xy, 0)).xyz * 2.0 - 1.0; // stored normalized
	surfaceNormal = mul(surfaceNormal, invTBN);

	// surface normal to world space
	output.Normal = float4( surfaceNormal * 0.5 + 0.5, 1);
	return output;
}

//--------------------------------------------------------------------------------------
GeometryShader gsGammatonUpdate=ConstructGSWithSO( CompileShader( gs_4_0, GS_GammatonUpdate() ),"VELOCITY.xyz; FLAGS.x; TEXCOORD.xy; CARRIED_MATERIAL.xy; RANDOM_SEED.x; DUMMY.xyz" );

technique11 ShaderModel5
{
	// some of the functions that are not directly involved in the material aging
	// were removed from the effect to increate readablity
	#include "HelperPasses.hlsl"

	pass SurfaceUpdate
	{
		SetVertexShader( CompileShader( vs_5_0, VS_ToAtlasCoords() ) );
		SetHullShader( 0 );
		SetDomainShader( 0 );
		SetGeometryShader( CompileShader( gs_5_0, GS_SurfaceUpdate() ) );
		SetPixelShader( CompileShader( ps_5_0, PS_SurfaceUpdate() ) );
	}

	pass GammatonUpdate
	{		
		SetVertexShader( CompileShader( vs_5_0, VS_PassThrough_Gammaton() ) );
		SetHullShader( 0 );
		SetDomainShader( 0 );
		SetGeometryShader( gsGammatonUpdate );
		SetPixelShader( 0 );
	}

	pass AgingProcess
	{
		SetVertexShader( CompileShader( vs_5_0, VS_PassThrough() ) );
		SetHullShader( 0 );
		SetDomainShader( 0 );
		SetGeometryShader( 0 );
		SetPixelShader( CompileShader( ps_5_0, PS_AgingProcess() ) );
	}

	pass Composition1
	{
		SetVertexShader( CompileShader( vs_5_0, VS_PassThrough() ) );
		SetHullShader( 0 );
		SetDomainShader( 0 );
		SetGeometryShader( 0 );
		SetPixelShader( CompileShader( ps_5_0, PS_CompositionPhase1() ) );
	}

	pass Composition2
	{
		SetVertexShader( CompileShader( vs_5_0, VS_PassThrough() ) );
		SetHullShader( 0 );
		SetDomainShader( 0 );
		SetGeometryShader( 0 );
		SetPixelShader( CompileShader( ps_5_0, PS_CompositionPhase2() ) );
	}

	pass Composition3
	{
		SetVertexShader( CompileShader( vs_5_0, VS_PassThrough() ) );
		SetHullShader( 0 );
		SetDomainShader( 0 );
		SetGeometryShader( 0 );
		SetPixelShader( CompileShader( ps_5_0, PS_CompositionPhase3() ) );
	}
}

//--------------------------------------------------------------------------------------

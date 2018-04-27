/*
SCREEN SPACE SOFT SHADOWS

Jesús Gumbau Portalés

e-mail: jgumbau@uji.es
		jesusgumbau@gmail.com

This technique is able of computing soft shadows with 
perceptually-correct penumbrae by using a separable 
Gaussian blur filter in screen space.

The advantage of this algorithm is that, as the Gaussian 
blur is a sepatable filter, it can be decomposed in two 
different passes, horizontal and vertical. This reduces 
the cost of computing the penumbrae to O(n+n) where n is 
the size the kernel, instead of O(n^2) for the standard 
PCSS algorithm.

In order to correctly deal with occluded portions of 
penumbrae, our technique needs to compute the first k 
layers of geometry in view space. Actually, the real
cost of the algorithm is O(k+n+n) for the worst case 
and O(n+n) for the best case (when sampling the first 
layer of geometry is sufficient for computing the soft
shadow).

This algorithm implements an 11x11 Gaussian filter which
has an average cost of 30 texture accesses per pixel. Using 
PCSS with such kernel size would lead to 121 texture fetches
per pixel.

The algorithm is performed in 6 different steps.

1. Compute the shadow map in light space.
     [Pass=MakeShadow]

2. Compute the hard shadows (standard shadow mapping) for the
     first layer of visible geometry. This pass uses MRT for
	 computing simultaneously the following buffers:
	 	 2.1. The unshadowed illuminated color buffer (gColorMap)
		 2.2. The distances map (gDistMap, see bellow)
		 2.3. The hard shadows buffer in view space (gHardShadowsMap)
		 2.4. The normal-depth texture (gNormalDepthMap)
	  [Pass=MakeHardShadowsL1]

3. 	Compute the hard shadows for the second visible geometry layer. 
	  (gHardShadowsMapL2).
	  [Pass=MakeHardShadowsL2]
	  
4. Perform the horizontal pass of the Gaussian blur. This step is done
     as a post-processing effect by rendering a quad at full screen.
     [Pass=SmoothShadowHoriz]
	 
5. Perform the vertical pass of the Gaussian blur. This step is done
     as a post-processing effect by rendering a quad at full screen.
     [Pass=SmoothShadowVert]
	 
6. This straightforward step just combines the computed soft shadows 
     with the unshadowed color (gColorMap).
	 [Pass=CombineFinal]   				   


% Screen space soft shadows.
% For more information, please see the 
% "GPU Pro".

*******************************************************************************
******************************************************************************/

#ifndef FXCOMPOSER_VERSION	/* for very old versions */
#define FXCOMPOSER_VERSION 180
#endif /* FXCOMPOSER_VERSION */

#ifndef DIRECT3D_VERSION
#define DIRECT3D_VERSION 0x900
#endif /* DIRECT3D_VERSION */

/*****************************************************************/
/*** EFFECT-SPECIFIC CODE BEGINS HERE ****************************/
/*****************************************************************/

/******* Lighting Macros *******/
/** To use "Object-Space" lighting definitions, change these two macros: **/
#define LIGHT_COORDS "World"
// #define OBJECT_SPACE_LIGHTS /* Define if LIGHT_COORDS is "Object" */

#include <include\\Quad.fxh>

float Script : STANDARDSGLOBAL <
    string UIWidget = "none";
    string ScriptClass = "scene";
    string ScriptOrder = "postprocess";
    string ScriptOutput = "color";
    string Script = "Technique=Main;";
> = 0.8;

// color and depth used for full-screen clears

float4 gClearColor <
    string UIWidget = "None";    
> = {1,1,1,1};

float gClearDepth <string UIWidget = "none";> = 1.0;

#define FAR 1.0f

float4 gShadowClearColor <
	string UIName = "Shadow Far BG";
    string UIWidget = "none";
> = {FAR,FAR,FAR,0.0};

texture gFloorTexture <
    string UIName = "Surface Texture";
    string ResourceName = "floor.dds";
    //string ResourceName = "default_color.dds";
>;

sampler2D gFloorSampler = sampler_state
{
    texture = <gFloorTexture>;
    AddressU = Wrap;
    AddressV = Wrap;
#if DIRECT3D_VERSION >= 0xa00
    Filter = MIN_MAG_MIP_LINEAR;
#else /* DIRECT3D_VERSION < 0xa00 */
    MinFilter = Linear;
    MipFilter = Linear;
    MagFilter = Linear;
#endif /* DIRECT3D_VERSION */
};

/**** UNTWEAKABLES: Hidden & Automatically-Tracked Parameters **********/

// transform object vertices to world-space:
float4x4 gWorldXf : World < string UIWidget="None"; >;
// transform object normals, tangents, & binormals to world-space:
float4x4 gWorldITXf : WorldInverseTranspose < string UIWidget="None"; >;
// transform object vertices to view space and project them in perspective:
float4x4 gWvpXf : WorldViewProjection < string UIWidget="None"; >;
// provide tranform from "view" or "eye" coords back to world-space:
float4x4 gViewIXf : ViewInverse < string UIWidget="None"; >;
// transform object normals, tangents, & binormals to view-space:
float4x4 gWorldViewITXf : WorldViewInverseTranspose < string UIWidget="None"; >;

float4x4 gWorldViewXf : WorldView < string UIWidget="None"; >;

// and these transforms are used for the shadow projection:

float4x4 gLampViewXf : View <
   string UIName = "Lamp View Xform";
   //string UIWidget="None";
   string Object = "SpotLight0";
>;

float4x4 gLampProjXf : Projection <
   string UIName = "Lamp Projection Xform";
   //string UIWidget="None";
   string Object = "SpotLight0";
>;

///////////////////////////////////////////////////////////////
/// TWEAKABLES ////////////////////////////////////////////////
///////////////////////////////////////////////////////////////

float3 gSpotLamp0Pos : POSITION <
    string Object = "SpotLight0";
    string UIName =  "Lamp 0 Position";
    string Space = (LIGHT_COORDS);
> = {-0.5f,2.0f,1.25f};

float3 gLamp0Color : COLOR <
    string UIName =  "Lamp 0";
    string Object = "Spotlight0";
    string UIWidget = "Color";
> = {1.0f,1.0f,1.0f};

float3 gSpotLamp0Dir : DIRECTION <
    string Object = "SpotLight0";
    string UIName =  "Lamp 0 Direction";
    string Space = (LIGHT_COORDS);
>;

float gFalloffAngle : FALLOFFANGLE <
	string Object = "SpotLight0";
>;

// Parameters for the algorithm

// LIGHT SIZE:
// The size of the light is directly proportional to the size 
// of the penumbra.
float gLightSize <
   string UIWidget = "slider";
   float UIMin = 0.010;
   float UIMax = 0.100;
   float UIStep = 0.01;
   string UIName = "Light Size";
> = 0.05f;


// SHADOW BIAS:
// Used for solving self-shadowing issues.
float gShadBias <
   string UIWidget = "slider";
   float UIMin = 0.0;
   float UIMax = 10.3;
   float UIStep = 0.0001;
   string UIName = "Shadow Bias";
> = 0.01;

// SCENE SCALE:
// Distance for the near plane.
float gSceneScale <
   string UIWidget = "slider";
   float UIMin = 0.1;
   float UIMax = 100.0;
   float UIStep = 0.1;
   string UIName = "Near Plane Factor";
> = 1.0f;


////////////////////////////////////////////// surface

// SURFACE COLOR:
// Used to modulate the color of the surfaces.
float3 gSurfaceColor : DIFFUSE <
    string UIName =  "Surface";
    string UIWidget = "Color";
> = {1,1,1};

// DIFFUSE CONSTANT
float gKd <
    string UIWidget = "slider";
    float UIMin = 0.0;
    float UIMax = 1.0;
    float UIStep = 0.01;
    string UIName =  "Diffuse";
> = 0.9;



////////////////////////////////////////////////////////
/// TEXTURES ///////////////////////////////////////////
////////////////////////////////////////////////////////

// The size of the shadow map
#define SHADOW_SIZE 1024
#define SHADOW_FMT  "r32f"

texture gShadMap : RENDERCOLORTARGET <
   float2 Dimensions = { SHADOW_SIZE, SHADOW_SIZE };
   string Format = (SHADOW_FMT) ;
   string UIWidget = "None";
>;

sampler2D gShadSampler = sampler_state {
    texture = <gShadMap>;
    AddressU = Clamp;
    AddressV = Clamp;
#if DIRECT3D_VERSION >= 0xa00
    Filter = MIN_MAG_MIP_POINT;
#else /* DIRECT3D_VERSION < 0xa00 */
    MinFilter = Point;
    MagFilter = Point;
    MipFilter = Point;
#endif /* DIRECT3D_VERSION */
};



// gTempMap: temporal buffer used to store results of the horizontal
//           blurring pass.
//  R: shadow intensity
//  B: pixel depth
DECLARE_QUAD_TEX(gTempMap,gTempMapSampler,"a32b32g32r32f")

// gDistMap: distances map. Store some useful distances per pixel
//  R: penumbra size
//  G: pixel depth
//  B: blocker mask (tells whether this pixel receives penumbra or not)
DECLARE_QUAD_TEX(gDistMap,gDistMapSampler,"a32b32g32r32f")

// gHardShadowsMap: stores hard shadows rendered in view space.
//  R: 0 if shadow / 1 if not
//  B: pixel depth
DECLARE_QUAD_TEX(gHardShadowsMap,gHardShadowsMapSampler,"a32b32g32r32f")
DECLARE_QUAD_TEX(gHardShadowsMapL2,gHardShadowsMapL2Sampler,"a32b32g32r32f")

// gSoftShadowsMap: stores the computed soft shadows.
//  R: intensity of the shadow
//  B: pixel depth (used for the vertical blurring pass)
DECLARE_QUAD_TEX(gSoftShadowsMap,gSoftShadowsMapSampler,"a32b32g32r32f")

// gNormalDepthMap: stores the per pixel normal and depth 
//  RGB: normal vector in eye space
//  A: pixel depth
DECLARE_QUAD_TEX(gNormalDepthMap,gNormalDepthMapSampler,"a32b32g32r32f")

// gColorMap: stores the unshadowed and illuminated color buffer
DECLARE_QUAD_TEX(gColorMap,gColorMapSampler,"a32b32g32r32f")

// gDepthTexture: depth buffer
DECLARE_QUAD_DEPTH_BUFFER(gDepthTexture,"D24S8")

// BUG! FX-Composer 2.5 seems to have a bug which forces me to use 
//      then following render targets, even if I don't use them
DECLARE_QUAD_TEX(gDummyMap1,gDummyMap1Sampler,"a32b32g32r32f")
DECLARE_QUAD_TEX(gDummyMap2,gDummyMap2Sampler,"a32b32g32r32f")
DECLARE_QUAD_TEX(gDummyMap3,gDummyMap3Sampler,"a32b32g32r32f")


// gShadDepthTarget: depth buffer used for creating the shadow map
texture gShadDepthTarget : RENDERDEPTHSTENCILTARGET <
   float2 Dimensions = { SHADOW_SIZE, SHADOW_SIZE };
   string format = "D24S8";
   string UIWidget = "None";
>;

/////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////
/// SHADER CODE BEGINS /////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////

/* data from application vertex buffer */
struct ShadowAppData {
   float3 Position     : POSITION;
   float4 UV           : TEXCOORD0;
   float4 Normal       : NORMAL;
};

// Connector from vertex to pixel shader
struct ShadowVertexOutput {
   float4 HPosition    : POSITION;
   float2 UV           : TEXCOORD0;
   float3 LightVec     : TEXCOORD1;
   float3 WNormal      : TEXCOORD2;
   float3 WView        : TEXCOORD3;
   float4 LP           : TEXCOORD4;   // current position in light-projection space
   float4 VertPos      : TEXCOORD5;   // Vertex position in eye space (multiplied by WorldView matrix)
   float3 VNormal	   : TEXCOORD6;   // Normal in view space
   float3 HPos		   : TEXCOORD7;   // the same as HPosition. Used to make HPosition accessible from the pixel shader
};

// Connector from vertex to pixel shader
struct JustShadowVertexOutput {
   float4 HPosition    : POSITION;
   float4 LP           : TEXCOORD0;    // current position in light-projection space
   float3 LightVec	   : TEXCOORD1;    // vector to the light
};

////////////////////////////////////////////////////////////////////////////////
/// Vertex Shaders /////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// This Vertex Shader is used in the MakeShadow pass in order. This is almost a
// pass-through shader
JustShadowVertexOutput shadVS(ShadowAppData IN,
			       	uniform float4x4 WorldITXf, // our four standard "untweakable" xforms
					uniform float4x4 WorldXf,
					uniform float4x4 ViewIXf,
					uniform float4x4 WvpXf,
			       	uniform float4x4 ShadowViewProjXf,
				   	uniform float3 SpotLightPos
) {
   JustShadowVertexOutput OUT = (JustShadowVertexOutput)0;
   float4 Po = float4(IN.Position.xyz,(float)1.0);     // object coordinates
   float4 Pw = mul(Po,WorldXf);                        // "P" in world coordinates
   float4 Pl = mul(Pw,ShadowViewProjXf);  // "P" in light coords
   OUT.LP = Pl;                // view coords (also lightspace projection coords in this case)
   OUT.HPosition = Pl; // screen clipspace coords
   
   OUT.LightVec = SpotLightPos - Pw.xyz;               // world coords   
   
   return OUT;
}


// Vertex Shader used to process the full-screen quad for 
// the post-processing FX
JustShadowVertexOutput quadShadVS(ShadowAppData IN) {
   JustShadowVertexOutput OUT = (JustShadowVertexOutput)0;
   OUT.LP = IN.UV;               
   OUT.HPosition = float4(IN.Position.xyz,(float)1.0); 
   return OUT;
}



// This is the main vertex shader used for passing data in world coordinates
// to the pixel shader for computing the different buffers neede for our algorithm

ShadowVertexOutput mainCamVS(ShadowAppData IN,
			       uniform float4x4 WorldITXf, // our four standard "untweakable" xforms
				   uniform float4x4 WorldViewITXf,
				   uniform float4x4 WorldXf,
				   uniform float4x4 ViewIXf,
				   uniform float4x4 WvpXf,
				   uniform float4x4 gWorldViewXf,
			       uniform float4x4 ShadowViewProjXf,
			       uniform float3 SpotLightPos,
			       uniform float ShadBias
) {
   ShadowVertexOutput OUT = (ShadowVertexOutput)0;
   OUT.WNormal = mul(IN.Normal,WorldITXf).xyz; // world coords
   float4 Po = float4(IN.Position.xyz,(float)1.0);     // "P" in object coordinates
   float4 Pw = mul(Po,WorldXf);                        // "P" in world coordinates
   float4 Pl = mul(Pw,ShadowViewProjXf);  // "P" in light coords
   Pl.z -= ShadBias;	// factor in bias here to save pixel shader work
   OUT.LP = Pl;                                                       
// ...for pixel-shader shadow calcs
   OUT.WView = normalize(ViewIXf[3].xyz - Pw.xyz);     // world coords
   OUT.HPosition = mul(Po,WvpXf);    // screen clipspace coords
   OUT.UV = IN.UV.xy;                                                 
// pass-thru
   OUT.LightVec = SpotLightPos - Pw.xyz;               // world coords
   OUT.VertPos = mul(Po,gWorldViewXf);
   OUT.VNormal = mul(IN.Normal,WorldViewITXf);
   OUT.HPos = OUT.HPosition.xyz;
   return OUT;
}

/*********************************************************/
/*********** pixel shaders********************************/
/*********************************************************/

// Pixel shader used to calculate the shadow map
// the falloff angle is taken into account in order
// to produce a shadow outside the area affected by
// the shadow map. This way, this shadow is also smoothed
// by our shadowing algorithm
float4 shadPS(JustShadowVertexOutput IN,
			  uniform float falloffAngle,
			  uniform float3 spotLightDir) : COLOR
{
	float3 Ln = normalize(IN.LightVec);	
	
  	if (dot(spotLightDir,-Ln)-0.05f > cos(falloffAngle*0.5f*3.14159f/180.0f))
   		return float4(IN.LP.zzz,1);
	else
		return float4(1.3f,1.3f,1.3f,1); // distance which is behind all objects in scene (Near plane in light space)
}


// -------------------------------------
// Search for potential blockers in the shadow map to estimate the penumbra at a given pixel location (see PCSS)
// -------------------------------------
float findBlocker(float2 uv,
		float4 LP,
		uniform sampler2D ShadowMap,
		uniform float bias,
		float searchWidth,
		float numSamples)
{
        // divide filter width by number of samples to use
        float stepSize = 2 * searchWidth / numSamples;

        // compute starting point uv coordinates for search
        uv = uv - float2(searchWidth, searchWidth);

        // reset sum to zero
        float blockerSum = 0;
        float receiver = LP.z;
        float blockerCount = 0;
        float foundBlocker = 0;

        // iterate through search region and add up depth values
        for (int i=0; i<numSamples; i++) {
               for (int j=0; j<numSamples; j++) {
                       float shadMapDepth = tex2D(ShadowMap, uv +
                                                 float2(i*stepSize,j*stepSize)).x;
                       // found a blocker
                       if (shadMapDepth < receiver) {
                               blockerSum += shadMapDepth;
                               blockerCount++;
                               foundBlocker = 1;
                       }
               }
        }

		float result;
		
		if (foundBlocker == 0) {
			// set it to a unique number so we can check
			// later to see if there was no blocker
			result = 999;
		}
		else {
		    // return average depth of the blockers
			result = blockerSum / blockerCount;
		}
		
		return result;
}

// ------------------------------------------------
// Estimate penumbra based on
// blocker estimate, receiver depth, and light size
// ------------------------------------------------
float estimatePenumbra(float4 LP,
			float Blocker,
			uniform float LightSize)
{
       // receiver depth
       float receiver = LP.z;
	   
       // estimate penumbra using parallel planes approximation
       float penumbra = (receiver - Blocker) * LightSize / Blocker;
       return penumbra;
}

// This pixel shader calculates in different buffers:
// - the unshadowed illumination of the scene
// - the distances map (which stores the penumbra size)
// - the hard shadows of the first layer of visible geometry
// - the per-pixel normal and depth

void makeBuffersAndHardShadowsL1PS(ShadowVertexOutput IN,
       uniform float3 SpotLightColor,
       uniform float LightSize,
       uniform float SceneScale,
       uniform float ShadBias,
       uniform float Kd,
       uniform float3 SurfColor,
	   uniform float falloffAngle,
	   uniform float3 spotLightDir,
       uniform sampler2D ShadSampler,
       uniform sampler2D FloorSampler,
	   out float4 ColorOutput : COLOR0,
	   out float4 DistMapOutput : COLOR1,
	   out float4 HardShadowsMapOutput : COLOR2,
	   out float4 NormalMapOutput : COLOR3)
{
   // Generic lighting code 
   float3 Nn = normalize(IN.WNormal);
   float3 Vn = normalize(IN.WView);
   float3 Ln = normalize(IN.LightVec);
   float ldn = dot(Ln,Nn);
   float3 diffContrib = SurfColor*(Kd*ldn * SpotLightColor);
   
   // Compute uv coordinates for the point being shaded
   // Saves some future recomputation.
   float2 uv = float2(.5,-.5)*(IN.LP.xy)/IN.LP.w + float2(.5,.5);

   // ---------------------------------------------------------
   // Find blocker estimate
   float searchSamples = 6;   // how many samples to use for blocker search
   float zReceiver = IN.LP.z;
   float searchWidth = SceneScale * (zReceiver - 1.0) / zReceiver;
   float blocker = findBlocker(uv, IN.LP, ShadSampler, ShadBias,
                              SceneScale * LightSize / IN.LP.z, searchSamples);
   
   // ---------------------------------------------------------
   // Step 2: Estimate penumbra using parallel planes approximation
   float penumbra;  
   penumbra = estimatePenumbra(IN.LP, blocker, LightSize);

   float3 floorColor = tex2D(FloorSampler, IN.UV*2).rgb;      
   float depth = -IN.VertPos.z;
  
   ColorOutput = float4((diffContrib*floorColor),1);
   DistMapOutput = float4(penumbra,depth*0.1,(blocker > 998),1);

   // compute hard shadows
   HardShadowsMapOutput = (IN.LP.z < tex2D(ShadSampler,uv).x && dot(spotLightDir,-Ln) > cos(falloffAngle*0.5f*3.14159f/180.0f) ? float4(1,1,depth,1) : float4(0,0,depth,1));
   NormalMapOutput = float4(normalize(IN.VNormal),depth);   
}


// This pixel shader calculates the hard shadows of the second layer of 
// visible geometry. It uses as input the depth stored in the first level 
// hardShadows in order to discard pixels behind the geometry rendered in 
// the previus layer.

void makeHardShadowsL2PS(ShadowVertexOutput IN,
       uniform float3 SpotLightColor,
       uniform float LightSize,
       uniform float SceneScale,
       uniform float ShadBias,
       uniform float Kd,
       uniform float3 SurfColor,
	   uniform float falloffAngle,
	   uniform float3 spotLightDir,	   
       uniform sampler2D ShadSampler,
       uniform sampler2D FloorSampler,
	   uniform sampler2D HardShadSampler,
	   out float4 hardShadowsL2Output : COLOR0,
	   out float4 DummyOutput1 : COLOR1,
	   out float4 DummyOutput2 : COLOR2,
	   out float4 DummyOutput3 : COLOR3)
{
    float2 pixelPos = IN.HPos.xy/IN.HPos.z*float2(.5,-.5)+float2(.5,.5);  
    float depth = -IN.VertPos.z;
	
	if (depth <= tex2D(HardShadSampler,pixelPos).z+0.01f)
   		discard;  
		
   // Compute uv coordinates for the point being shaded
   float2 uv = float2(.5,-.5)*(IN.LP.xy)/IN.LP.w + float2(.5,.5);
   
   float3 Ln = normalize(IN.LightVec);
		 
   hardShadowsL2Output = (IN.LP.z < tex2D(ShadSampler,uv).x && dot(spotLightDir,-Ln) > cos(falloffAngle*0.5f*3.14159f/180.0f) ? float4(1,1,depth,1) : float4(0,0,depth,1));
   DummyOutput1 = float4(1,1,1,1);   
   DummyOutput2 = float4(1,1,1,1);	
   DummyOutput3 = float4(1,1,1,1);   
   }

// weights used for our Gaussian blur filter
const float weights[11] = {
	0.10369036,
	
	0.10226017,
	0.098086871,
	0.091506422,
	0.083028749,
	0.073272616,
	
	0.10226017,
	0.098086871,
	0.091506422,
	0.083028749,
	0.073272616,
};

// This pixel shader computes a single pass of the Gaussian filter.
// The direction (horizontal/vertical) of the pass is defined by the
// uniform parameter horizVert, which can be either (1,0) for the 
// horizontal pass and (0,1) for the vertical pass.

float4 blurShadowPS(JustShadowVertexOutput IN,
       uniform float LightSize,
       uniform float SceneScale,
	   uniform float2 horizVert,
	   uniform sampler2D hardShadows,
	   uniform sampler2D distMap,
	   uniform sampler2D normalDepthMap,
	   uniform sampler2D hardShadows2) : COLOR
{
	float4 distmap = tex2D(distMap,IN.LP.xy);	
	
	float kernelWidth = 1.0f;
	float2 sampleStep = kernelWidth*float2(1.0f,1.0f);
	
	float4 normalDepth = tex2D(normalDepthMap,IN.LP.xy);
	
	// the following factors affect to the size of the gaussian blur:	
	sampleStep *= horizVert; // select vertical or horizontal pass
	sampleStep *= distmap.r; // apply penumbra size (from the distances map)
	sampleStep *= LightSize; // apply Light size (proportional to the size of the penumbra)
	sampleStep *= sqrt(dot(float3(0,0,1),normalDepth.xyz)); //apply anisotropy depending on the normal of the pixel
	sampleStep *= 1.0f/distmap.y;  // the pixel depth affects the size of the gaussian filter (because is screen space)
	
	if (distmap.z < 0.5f) // only apply blurring for penumbra regions (mask stored on the distances map)
	{		
		float2 offsets[10]={5.0f*sampleStep,4.0f*sampleStep,3.0f*sampleStep,2.0f*sampleStep,sampleStep,
						   	-5.0f*sampleStep,-4.0f*sampleStep,-3.0f*sampleStep,-2.0f*sampleStep,-sampleStep};
								
		float4 sample0 = tex2D(hardShadows,IN.LP.xy);  // sample the pixel of the center
		
		float finalColor = sample0.x * weights[0]; // apply the weight according to the gaussian blur
		float sumWeightsOK = weights[0];  // maintain a count of valid weights
		float errDepth = 0.3f;	// this error is used to discard "invalid" samples 
			
		// fetch samples for the gaussian blur
		// each sample contains the shadow factor (X) and its depth in view space (Z)
		for (int i=0; i<10; i++)
		{
			// first, read the sample from the first level
			float4 sampleL0 = tex2D(hardShadows,IN.LP.xy+offsets[i]);			
			
			// compare depths in view space in order to discard invalid pixels
			if (abs(sampleL0.z-sample0.z)<errDepth)
			{
				// the sample is considered valid
				sumWeightsOK += weights[i+1];     // accumulate valid weights
				finalColor += sampleL0.x * weights[i+1];   // accumulate weighted shadow value
			}
			else 
			{
				// if the sample of the first layer is considered invalid 
				// then try sampling the second layer
				float4 sampleL1 = tex2D(hardShadows2,IN.LP.xy+offsets[i]);  
				if (abs(sampleL1.z-sample0.z)<errDepth)
				{									
					// the sample is considered valid
					sumWeightsOK += weights[i+1];  // accumulate valid weights
					finalColor += sampleL1.x * weights[i+1];  // accumulate weighted shadow value
				}
				// if no valid sample can be found we consider we have not enough information
				// and will not take into account any layer for this sample
			}
		}
		
		// lastly, average by the weights that passed the previous tests
		finalColor /= sumWeightsOK;		
		
		return float4(finalColor,0,normalDepth.w,1); // this depth is used by the vertical blurring pass
	}
	else
	 	// if this pixel is not affected by the penumbra just leave it unshadowed
		return float4(1,0,normalDepth.w,1); // this depth is used by the vertical blurring pass //tex2D(hardShadows,IN.LP.xy);
}


// this shader simply combines the unshadowed illuminated color buffer with the soft shadows
float4 combineShadowPS(JustShadowVertexOutput IN,
	   uniform sampler2D softShadowsMap,
	   uniform sampler2D colorMap) : COLOR
{
	return float4(tex2D(softShadowsMap,IN.LP.xy).xxx*tex2D(colorMap,IN.LP.xy).xyz,1.0f);
}





////////////////////////////////////////////////////////////////////
/// TECHNIQUES /////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////

technique Main <
       string Script = "Pass=MakeShadow;"
		       		   "Pass=MakeHardShadowsL1;"
					   "Pass=MakeHardShadowsL2;"
					   "Pass=SmoothShadowHoriz;"
					   "Pass=SmoothShadowVert;"
					   "Pass=CombineFinal;";
> {
	   // 1. Compute the shadow map in light space.
       pass MakeShadow <
               string Script = "RenderColorTarget0=gShadMap;"
				"RenderDepthStencilTarget=gShadDepthTarget;"
				"RenderPort=SpotLight0;"
				"ClearSetColor=gShadowClearColor;"
				"ClearSetDepth=gClearDepth;"
				"Clear=Color;"
				"Clear=Depth;"
				"Draw=geometry;";
       > {
	   	VertexShader = compile vs_3_0 shadVS(gWorldITXf,gWorldXf,
						gViewIXf,gWvpXf,
					   	mul(gLampViewXf,gLampProjXf),
						gSpotLamp0Pos);
	    ZEnable = true;
		ZWriteEnable = true;
		ZFunc = LessEqual;
		AlphaBlendEnable = false;
		CullMode = None;
	   	PixelShader = compile ps_3_0 shadPS(gFalloffAngle,gSpotLamp0Dir);
       }
	   
	   
	   // 2. Compute the hard shadows (standard shadow mapping) for the
       //  first layer of visible geometry. This pass uses MRT for
	   //  computing simultaneously the following buffers:
	   //	 2.1. The unshadowed illuminated color buffer (gColorMap)
	   //	 2.2. The distances map (gDistMap, see bellow)
	   //	 2.3. The hard shadows buffer in view space (gHardShadowsMap)
	   //	 2.4. The normal-depth texture (gNormalDepthMap)	  		   
       pass MakeHardShadowsL1 <
               string Script = "RenderColorTarget0=gColorMap;"
			   	   "RenderColorTarget1=gDistMap;"
				   "RenderColorTarget2=gHardShadowsMap;"
				   "RenderColorTarget3=gNormalDepthMap;"
			       "RenderDepthStencilTarget=gDepthTexture;"
			       "RenderPort=;"
			       "ClearSetColor=gClearColor;"
			       "ClearSetDepth=gClearDepth;"
			       "Clear=Color;"
			       "Clear=Depth;"
			       "Draw=geometry;";
       > {
	   VertexShader = compile vs_3_0 mainCamVS(gWorldITXf,gWorldViewITXf,gWorldXf,
					   gViewIXf,gWvpXf,gWorldViewXf,
					   mul(gLampViewXf,gLampProjXf),
					   gSpotLamp0Pos,
					   gShadBias);
	    ZEnable = true;
		ZWriteEnable = true;
		ZFunc = LessEqual;
		AlphaBlendEnable = false;
		CullMode = None;
	   PixelShader = compile ps_3_0 makeBuffersAndHardShadowsL1PS(
					       gLamp0Color,
					       gLightSize,
					       gSceneScale,
					       gShadBias,
					       gKd,
					       gSurfaceColor,
						   gFalloffAngle,
						   gSpotLamp0Dir,
					       gShadSampler,
					       gFloorSampler );
       }
	   
	   
	// 3. 	Compute the hard shadows for the second visible geometry layer. 
	//	  (gHardShadowsMapL2).
	   
      pass MakeHardShadowsL2 <
               string Script = "RenderColorTarget0=gHardShadowsMapL2;"
			   	   "RenderColorTarget1=gDummyMap1;"
				   "RenderColorTarget2=gDummyMap2;"
				   "RenderColorTarget3=gDummyMap3;"				   
			       "RenderDepthStencilTarget=gDepthTexture;"
			       "RenderPort=;"
			       "ClearSetColor=gClearColor;"
			       "ClearSetDepth=gClearDepth;"
			       "Clear=Color;"
			       "Clear=Depth;"
			       "Draw=geometry;";
       > {
	    VertexShader = compile vs_3_0 mainCamVS(gWorldITXf,gWorldViewITXf,gWorldXf,
					   gViewIXf,gWvpXf,gWorldViewXf,
					   mul(gLampViewXf,gLampProjXf),
					   gSpotLamp0Pos,
					   gShadBias);
	    ZEnable = true;
		ZWriteEnable = true;
		ZFunc = LessEqual;
		AlphaBlendEnable = false;
		CullMode = cw;
	   PixelShader = compile ps_3_0 makeHardShadowsL2PS(
					       gLamp0Color,
					       gLightSize,
					       gSceneScale,
					       gShadBias,
					       gKd,
					       gSurfaceColor,
						   gFalloffAngle,
						   gSpotLamp0Dir,
					       gShadSampler,
					       gFloorSampler,
						   gHardShadowsMapSampler
					       );
       }
	   
	   
	// 4. Perform the horizontal pass of the Gaussian blur. This step is done
    //  as a post-processing effect by rendering a quad at full screen.
	
       pass SmoothShadowHoriz <
               string Script = "RenderColorTarget0=gTempMap;"
			       "RenderDepthStencilTarget=gDepthTexture;"
			       "RenderPort=;"
			       "ClearSetColor=gClearColor;"
			       "Draw=buffer;";
       > {
	   VertexShader = compile vs_3_0 quadShadVS();
	    ZEnable = true;
		ZWriteEnable = true;
		ZFunc = LessEqual;
		AlphaBlendEnable = false;
		CullMode = None;
	   PixelShader = compile ps_3_0 blurShadowPS(
					       gLightSize,
					       gSceneScale,
						   float2(1,0),
						   gHardShadowsMapSampler,
						   gDistMapSampler,
						   gNormalDepthMapSampler,
						   gHardShadowsMapL2Sampler);
       }	  
	   
	   
	// 5. Perform the vertical pass of the Gaussian blur. This step is done
    //  as a post-processing effect by rendering a quad at full screen.      
	   
       pass SmoothShadowVert <
               string Script = "RenderColorTarget0=gSoftShadowsMap;"
			       "RenderDepthStencilTarget=gDepthTexture;"
			       "RenderPort=;"
			       "ClearSetColor=gClearColor;"
			       "Draw=buffer;";
       > {
	   VertexShader = compile vs_3_0 quadShadVS();
	    ZEnable = true;
		ZWriteEnable = true;
		ZFunc = LessEqual;
		AlphaBlendEnable = false;
		CullMode = None;
	   PixelShader = compile ps_3_0 blurShadowPS(
					       gLightSize,
					       gSceneScale,
						   float2(0,1),
						   gTempMapSampler,
						   gDistMapSampler,
						   gNormalDepthMapSampler,
						   gHardShadowsMapL2Sampler);
       }	     
	   
	 //  6. This straightforward step just combines the computed soft shadows 
     //   with the unshadowed color (gColorMap).

       pass CombineFinal <
               string Script = "RenderColorTarget0=;"
			       "RenderDepthStencilTarget=gDepthTexture;"
			       "RenderPort=;"
			       "ClearSetColor=gClearColor;"
			       "Draw=buffer;";
       > {
	   VertexShader = compile vs_3_0 quadShadVS();
	    ZEnable = true;
		ZWriteEnable = true;
		ZFunc = LessEqual;
		AlphaBlendEnable = false;
		CullMode = None;
	   PixelShader = compile ps_3_0 combineShadowPS(gSoftShadowsMapSampler,gColorMapSampler);
       }	 
	  	   
}

/***************************** eof ***/

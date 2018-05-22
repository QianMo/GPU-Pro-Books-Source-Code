
#include "shaders/DeferredRendererShader_GI_IV_Propagation.h"

void DRShaderGI_IV_Propagation::start()
{
	DRShaderGI::start();

	shader->begin();

	shader->setUniform1i(0,10,uniform_photonmap_composited_red);		
	shader->setUniform1i(0,11,uniform_photonmap_composited_green);
	shader->setUniform1i(0,12,uniform_photonmap_composited_blue);
	shader->setUniform1i(0,13,uniform_photonmap_occupied);

	shader->setUniform1f(0,cfactor,uniform_cfactor);
	shader->setUniform1f(0,(float)iteration,uniform_iteration);
	shader->setUniform3f(0,1.0f/(float)dimx, 1.0f/(float)dimy, 1.0f/(float)dimz, uniform_photonmap_resolution);

//	printf ("PROPAGATION: %f %f - %f %f %f\n", cfactor, (float)iteration,
//		1.0f/(float)dimx, 1.0f/(float)dimy, 1.0f/(float)dimz);
}

bool DRShaderGI_IV_Propagation::init(class DeferredRenderer* _renderer)
{
	if (initialized)
		return true;

	if (!DRShaderGI::init(_renderer))
		return false;
	
	char * shader_text_vert;
    char * shader_text_frag;
	
	shader_text_vert = DRSH_GI_IV_Vert;
	shader_text_frag = DRSH_GI_IV_Frag;

	shader = shader_manager.loadfromMemory ("Global Illumination Prop Vert", shader_text_vert, 
		                                     "Global Illumination Prop Frag", shader_text_frag);
	if (!shader)
	{
		EAZD_TRACE ("DeferredRenderer::buildShaders() : ERROR - Problem compiling GI shader (IV Propagation).\n");
		return false;
	}
	else
	{
		uniform_photonmap_composited_red = shader->GetUniformLocation("photonmap_composited_red");
		uniform_photonmap_composited_green = shader->GetUniformLocation("photonmap_composited_green");
		uniform_photonmap_composited_blue = shader->GetUniformLocation("photonmap_composited_blue");
		uniform_photonmap_occupied = shader->GetUniformLocation("photonmap_occupied");

		uniform_cfactor = shader->GetUniformLocation("cfactor");
		uniform_iteration = shader->GetUniformLocation("iteration");
		uniform_photonmap_resolution = shader->GetUniformLocation("photonmap_resolution");
	}
	
	initialized = true;
	return true;
}

//----------------- Shader text ----------------------------

char DRShaderGI_IV_Propagation::DRSH_GI_IV_Vert[] = "\n\
varying vec3 Necs; \n\
varying vec4 Pecs; \n\
\n\
void main (void) \n\
{ \n\
    gl_Position = ftransform (); \n\
	\n\
    Necs = gl_NormalMatrix * gl_Normal; \n\
    Pecs = gl_ModelViewMatrix * gl_Vertex; \n\
    gl_TexCoord[0] = gl_TextureMatrix[0] * gl_Vertex; \n\
}";

char DRShaderGI_IV_Propagation::DRSH_GI_IV_Frag[] = "\n\
uniform sampler3D   photonmap_composited_red; \n\
uniform sampler3D   photonmap_composited_green; \n\
uniform sampler3D   photonmap_composited_blue; \n\
uniform sampler3D   photonmap_occupied; \n\
uniform float		iteration; \n\
uniform float       cfactor; \n\
uniform vec3        photonmap_resolution; \n\
\n\
const float reflectivity = 0.5; \n\
//float cfactor = 0.27; \n\
\n\
#define DO_BOUNCE 1 \n\
#define LUM_CULLING 1 \n\
#define THREE_GI_BANDS \n\
//#define ACCUM \n\
#define DO_BLOCK 1 \n\
\n\
#define VEC3_POS_X      vec3 ( 1.0,  0.0,  0.0) \n\
#define VEC3_NEG_X      vec3 (-1.0,  0.0,  0.0) \n\
#define VEC3_POS_Y      vec3 ( 0.0,  1.0,  0.0) \n\
#define VEC3_NEG_Y      vec3 ( 0.0, -1.0,  0.0) \n\
#define VEC3_POS_Z      vec3 ( 0.0,  0.0,  1.0) \n\
#define VEC3_NEG_Z      vec3 ( 0.0,  0.0, -1.0) \n\
\n\
// Map [-1.0,1.0] to [0.0,1.0] \n\
//#define MAP_MINUS1TO1_0TO1(_value)  (0.5 * ((_value) + 1.0)) \n\
#define MAP_MINUS1TO1_0TO1(_value)  _value \n\
\n\
// Map [0.0,1.0] to [-1.0,1.0] \n\
//#define MAP_0TO1_MINUS1TO1(_value)  (2.0 * (_value) - 1.0) \n\
#define MAP_0TO1_MINUS1TO1(_value)  _value \n\
\n\
// ---------------- Function Declarations --------------------- \n\
\n\
vec4 SHBasis (const in vec3 dir) \n\
{ \n\
    float   L00  = 0.282094792; \n\
    float   L1_1 = 0.488602512 * dir.y; \n\
    float   L10  = 0.488602512 * dir.z; \n\
    float   L11  = 0.488602512 * dir.x; \n\
    return vec4 (L11, L1_1, L10, L00); \n\
}\n\
\n\
vec4 SHProjectCone (const in vec3 cdir) \n\
{ \n\
	vec3 dir = normalize(cdir); \n\
	return vec4( 1.023326*cdir.x, 1.023326*cdir.y, 1.023326*cdir.z, 0.886226); \n\
} \n\
vec4 Intensity2SH (in vec3 dir, in float L) \n\
{ \n\
	return L*SHBasis(dir); \n\
} \n\
 \n\
float SH2Intensity (in vec4 sh, in vec3 dir) \n\
{ \n\
	return max( dot(sh, SHBasis(dir)),0); \n\
} \n\
// ---------------------------------------------------------- \n\
\n\
vec4 propagateDir (inout vec4 hit, const in sampler3D volume, const in vec3 nOffset, const float mask, const vec3 Necs) \n\
{ \n\
	vec4 sampleCoeffs = texture3D (volume, gl_TexCoord[0].xyz - (photonmap_resolution * nOffset)); \n\
	sampleCoeffs = MAP_0TO1_MINUS1TO1 (sampleCoeffs); \n\
	\n\
    // generate function for incoming direction from adjacent cell \n\
    vec4 shIncomingDirFunction = SHProjectCone (nOffset); \n\
    \n\
    // integrate incoming radiance with this function \n\
    float incidentLuminance  = 0; \n\
    \n\
#if LUM_CULLING \n\
	vec4 neighbor_mask = texture3D (photonmap_occupied, gl_TexCoord[0].xyz - (photonmap_resolution * nOffset)); \n\
	if(neighbor_mask.a == 1) \n\
	{ \n\
		vec3 neighbor_normal = MAP_0TO1_MINUS1TO1( neighbor_mask.xyz); \n\
		if(dot(neighbor_normal, nOffset)>0.0) \n\
		{ \n\
			incidentLuminance  = max (0.0, dot (sampleCoeffs, shIncomingDirFunction)); \n\
		} \n\
	} \n\
    else \n\
        incidentLuminance = max (0.0, dot (sampleCoeffs, shIncomingDirFunction)); \n\
#else \n\
	incidentLuminance = max (0.0, dot (sampleCoeffs, shIncomingDirFunction)); \n\
#endif \n\
#ifdef DO_BLOCK // LPV-style \n\
	if(mask > 0.5) \n\
#if DO_BOUNCE \n\
	{ \n\
		//create a new, reversed VPL \n\
		hit+= incidentLuminance * shIncomingDirFunction; \n\
		return  Intensity2SH (-nOffset, reflectivity * incidentLuminance);  \n\
	} \n\
#else \n\
	{ \n\
		//attenuate incoming illumination due to blocking \n\
		vec4 rad = 0.1* incidentLuminance * shIncomingDirFunction; \n\
		hit+= rad; \n\
		return rad;  \n\
	} \n\
#endif \n\
	else \n\
#endif \n\
	{ \n\
		// add it to the result \n\
		return incidentLuminance * shIncomingDirFunction; \n\
	} \n\
} \n\
\n\
#define PROPAGATE_FUNC propagateDir \n\
\n\
void main (void) \n\
{ \n\
    // check to see if the current cell is occupied or not \n\
    vec4    occp_cell   = texture3D (photonmap_occupied, gl_TexCoord[0].xyz); \n\
    vec3 Necs = MAP_0TO1_MINUS1TO1( occp_cell.xyz);  //[0,1] ---> [-1, 1] \n\
	\n\
    //the average normal can be zero!!!! \n\
	if( all (equal (Necs, vec3(0.0, 0.0, 0.0)))) \n\
		Necs =  vec3(1.0, 0.0, 0.0); \n\
	\n\
    // ------------------------------------------------------------------------- \n\
    // Do the propagation \n\
    // ------------------------------------------------------------------------- \n\
	\n\
    vec4 pixelCoeffs = vec4 (0.0, 0.0, 0.0, 0.0); \n\
    vec4 pixelCoeffsGreen = vec4 (0.0, 0.0, 0.0, 0.0); \n\
	vec4 pixelCoeffsBlue = vec4 (0.0, 0.0, 0.0, 0.0); \n\
	vec4 hitRed = vec4 (0.0, 0.0, 0.0, 0.0); \n\
    vec4 hitGreen = vec4 (0.0, 0.0, 0.0, 0.0); \n\
	vec4 hitBlue = vec4 (0.0, 0.0, 0.0, 0.0); \n\
	\n\
    // 6-point axial gathering stencil cross \n\
    pixelCoeffs = \n\
            PROPAGATE_FUNC (hitRed,photonmap_composited_red, VEC3_NEG_X, occp_cell.a, Necs) + \n\
            PROPAGATE_FUNC (hitRed,photonmap_composited_red, VEC3_POS_X, occp_cell.a, Necs) + \n\
			PROPAGATE_FUNC (hitRed,photonmap_composited_red, VEC3_NEG_Y, occp_cell.a, Necs) + \n\
            PROPAGATE_FUNC (hitRed,photonmap_composited_red, VEC3_POS_Y, occp_cell.a, Necs) + \n\
			PROPAGATE_FUNC (hitRed,photonmap_composited_red, VEC3_NEG_Z, occp_cell.a, Necs) + \n\
            PROPAGATE_FUNC (hitRed,photonmap_composited_red, VEC3_POS_Z, occp_cell.a, Necs) ; \n\
			\n\
		pixelCoeffsGreen = \n\
            PROPAGATE_FUNC (hitGreen,photonmap_composited_green, VEC3_NEG_X, occp_cell.a, Necs) + \n\
            PROPAGATE_FUNC (hitGreen,photonmap_composited_green, VEC3_POS_X, occp_cell.a, Necs) + \n\
            PROPAGATE_FUNC (hitGreen,photonmap_composited_green, VEC3_NEG_Y, occp_cell.a, Necs) + \n\
            PROPAGATE_FUNC (hitGreen,photonmap_composited_green, VEC3_POS_Y, occp_cell.a, Necs) + \n\
            PROPAGATE_FUNC (hitGreen,photonmap_composited_green, VEC3_NEG_Z, occp_cell.a, Necs) + \n\
            PROPAGATE_FUNC (hitGreen,photonmap_composited_green, VEC3_POS_Z, occp_cell.a, Necs) ; \n\
			\n\
		pixelCoeffsBlue = \n\
            PROPAGATE_FUNC (hitBlue,photonmap_composited_blue, VEC3_NEG_X, occp_cell.a, Necs) + \n\
            PROPAGATE_FUNC (hitBlue,photonmap_composited_blue, VEC3_POS_X, occp_cell.a, Necs) + \n\
            PROPAGATE_FUNC (hitBlue,photonmap_composited_blue, VEC3_NEG_Y, occp_cell.a, Necs) + \n\
            PROPAGATE_FUNC (hitBlue,photonmap_composited_blue, VEC3_POS_Y, occp_cell.a, Necs) + \n\
            PROPAGATE_FUNC (hitBlue,photonmap_composited_blue, VEC3_NEG_Z, occp_cell.a, Necs) + \n\
            PROPAGATE_FUNC (hitBlue,photonmap_composited_blue, VEC3_POS_Z, occp_cell.a, Necs) ; \n\
            \n\
//		pixelCoeffs=0.5*cfactor*pixelCoeffs-0.5*texture3D (photonmap_composited_red, gl_TexCoord[0].xyz); \n\
//		pixelCoeffsGreen=0.5*cfactor*pixelCoeffsGreen-0.5*texture3D (photonmap_composited_green, gl_TexCoord[0].xyz); \n\
//		pixelCoeffsBlue=0.5*cfactor*pixelCoeffsBlue-0.5*texture3D (photonmap_composited_blue, gl_TexCoord[0].xyz); \n\
        gl_FragData[0] = MAP_MINUS1TO1_0TO1(cfactor*pixelCoeffs); \n\
        gl_FragData[1] = MAP_MINUS1TO1_0TO1(cfactor*pixelCoeffsGreen); \n\
        gl_FragData[2] = MAP_MINUS1TO1_0TO1(cfactor*pixelCoeffsBlue); \n\
\n\
	//	gl_FragData[0] = texture3D (photonmap_composited_red, gl_TexCoord[0].xyz); \n\
	//	gl_FragData[1] = texture3D (photonmap_composited_green, gl_TexCoord[0].xyz); \n\
	//	gl_FragData[2] = texture3D (photonmap_occupied, gl_TexCoord[0].xyz); \n\
#ifdef ACCUM \n\
		gl_FragData[3] = (occp_cell.a>0 && iteration>0)?cfactor*hitRed:vec4(0.0,0.0,0.0,0.0); \n\
		gl_FragData[4] = (occp_cell.a>0 && iteration>0)?cfactor*hitGreen:vec4(0.0,0.0,0.0,0.0); \n\
		gl_FragData[5] = (occp_cell.a>0 && iteration>0)?cfactor*hitBlue:vec4(0.0,0.0,0.0,0.0); \n\
#endif \n\
}";

/*
vec4 SHRotate (const in vec3 vcDir, const in vec2 vZHCoeffs) \n\
{ \n\
// compute sine and cosine of theta angle \n\
    // beware of singularity when both x and y are 0 (no need to rotate at all) \n\
    vec2 theta12_cs; \n\
    if (vcDir.xz == vec2 (0.0, 0.0))    // save operations by not doing the normalize \n\
        theta12_cs = vec2 (0.0, 0.0); \n\
    else \n\
        theta12_cs = normalize (vcDir.xz); \n\
    // compute sine and cosine of phi angle \n\
    vec2 phi12_cs; \n\
    phi12_cs.x = sqrt (1.0 - vcDir.y * vcDir.y); \n\
    phi12_cs.y = vcDir.y; \n\
    vec4 vResult; \n\
	\n\
    // The first band is rotation-independent \n\
    vResult.w =  vZHCoeffs.x; \n\
	\n\
    // rotating the second band of SH \n\
    vResult.z =  vZHCoeffs.y * phi12_cs.x * theta12_cs.y;   // cos_phi * sin_theta \n\
    vResult.y =  vZHCoeffs.y * phi12_cs.y;                  // sin_phi \n\
    vResult.x =  vZHCoeffs.y * phi12_cs.x * theta12_cs.x;   // cos_phi * cos_theta \n\
	\n\
    return vResult; \n\
} \n\
\n\
// Here is a listing of method that takes a direction and a cone angle as an input \n\
// and returns SH coefficients of this cone of given angle rotated towards given direction \n\
vec4 SHProjectCone (const in vec3 vcDir, float angle) \n\
{ \n\
    vec2 vZHCoeffs = vec2 (0.50 * (1.0 - cos (angle)),          // 1/2 (1 - Cos[\[Alpha]]) \n\
                           0.75 * sin (angle) * sin (angle));   // 3/4 Sin[\[Alpha]]^2 \n\
\n\
    return SHRotate (vcDir, vZHCoeffs); \n\
} \n\
\n\
// Here is a listing of method that takes a direction as an input and returns \n\
// SH coefficients of hemispherical cosine lobe rotated towards given direction: \n\
vec4 SHProjectCone (const in vec3 vcDir) \n\
{ \n\
    vec2 vZHCoeffs = vec2 (0.25,    // 1/4 \n\
                           0.50);   // 1/2 \n\
\n\
    return SHRotate (vcDir, vZHCoeffs); \n\
} \n\
\n\
// 4 spherical harmonics: sh = { L(0,0), L(1,1), L(1,0), L(1,-1) } \n\
 \n\
// antimetwpish eidikhs periptwshs dir=(0,1,0) \n\
// metafora toy sta8eroy oroy (meso xrwma) sto A component gia na blepoume ti \n\
// pairnoume kai na einai krithrio gia occupancy. sh.xyz antistoixoun me tous \n\
// antistoixoys a3ones twra. \n\
 \n\
vec4 Intensity2SH (in vec3 dir, in float L) \n\
{ \n\
	float   phi = (abs (dir.y) == 1.0) ? 0.0 : atan (dir.z, dir.x); \n\
    float   theta = acos (dir.y); \n\
    float   sin_phi   = sin (phi); \n\
    float   cos_phi   = cos (phi); \n\
    float   sin_theta = sin (theta); \n\
    float   cos_theta = cos (theta); \n\
 \n\
    float   L00  = 0.282094792; \n\
    float   L1_1 = -0.488602512 * sin_theta * sin_phi; \n\
    float   L10  = 0.488602512 * cos_theta; \n\
    float   L11  = 0.488602512 * sin_theta * cos_phi; \n\
 \n\
    // sh is in [-1,1] range \n\
    return L * vec4 (L11, L1_1, L10, L00); \n\
} \n\
 \n\
float SH2Intensity (in vec4 sh, in vec3 dir) \n\
{ \n\
    return 0.886227 * sh.a + 1.023328 * dot (sh.rgb, dir); \n\
} \n\
 \n\
vec4 SHBasis (in vec3 dir) \n\
{ \n\
    float   cos_theta = dir.y; \n\
    float   sin_theta = sqrt (1.0 - dir.y * dir.y); \n\
    float   sin_phi; \n\
    float   cos_phi; \n\
    if(dir.xz==vec2(0.0,0.0)){ \n\
    //if(abs (dir.y) == 1.0){ \n\
		sin_phi = 0;  \n\
		cos_phi=0; \n\
    } \n\
    else{ \n\
		sin_phi = dir.z;  \n\
		cos_phi=dir.x; \n\
    } \n\
 \n\
    float   L00  = 0.282094792; \n\
    float   L1_1 = -0.488602512 * sin_theta * sin_phi; \n\
    float   L10  = 0.488602512 * cos_theta; \n\
    float   L11  = 0.488602512 * sin_theta * cos_phi; \n\
 \n\
    // sh is in [-1,1] range \n\
    return vec4 (L11, L1_1, L10, L00); \n\
} \n\
*/
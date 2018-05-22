#include "shaders/DeferredRendererShader_SphericalHarmonics.h"

// 4-band spherical harmonics base creation and transformation functions
char DRShader_SH::DRSH_SH_Basis[]="\n\
// --- NVidia GLSL compiling options \n\
#pragma optionNV(fastmath on) \n\
#pragma optionNV(fastprecision on) \n\
#pragma optionNV(ifcvt none) \n\
#pragma optionNV(inline all) \n\
#pragma optionNV(strict on) \n\
#pragma optionNV(unroll all) \n\
 \n\
// Arbitrary rotation of function with circularly symmetry around Z \n\
// Here is a listing of method that takes a direction and zonal harmonics \n\
// coefficients as an input and returns SH coefficients of this function rotated \n\
// towards given direction \n\
 \n\
vec4 SHRotate (const in vec3 vcDir, const in vec2 vZHCoeffs) \n\
{ \n\
    // compute sine and cosine of theta angle \n\
    // beware of singularity when both x and y are 0 (no need to rotate at all) \n\
    vec2 theta12_cs; \n\
    if (vcDir.xz == vec2 (0.0, 0.0))    // save operations by not doing the normalize \n\
        theta12_cs = vec2 (0.0, 0.0); \n\
    else \n\
        theta12_cs = normalize (vcDir.xz); \n\
 \n\
    // compute sine and cosine of phi angle \n\
    vec2 phi12_cs; \n\
    phi12_cs.x = sqrt (1.0 - vcDir.y * vcDir.y); \n\
    phi12_cs.y = vcDir.y; \n\
 \n\
    vec4 vResult; \n\
 \n\
    // The first band is rotation-independent \n\
    vResult.w =  vZHCoeffs.x; \n\
 \n\
    // rotating the second band of SH \n\
    vResult.z =  vZHCoeffs.y * phi12_cs.x * theta12_cs.y;   // cos_phi * sin_theta \n\
    vResult.y = -vZHCoeffs.y * phi12_cs.y;                  // sin_phi \n\
    vResult.x =  vZHCoeffs.y * phi12_cs.x * theta12_cs.x;   // cos_phi * cos_theta \n\
 \n\
    return vResult; \n\
} \n\
 \n\
// Here is a listing of method that takes a direction and a cone angle as an input \n\
// and returns SH coefficients of this cone of given angle rotated towards given direction \n\
vec4 SHProjectCone (const in vec3 vcDir, float angle) \n\
{ \n\
    vec2 vZHCoeffs = vec2 (0.50 * (1.0 - cos (angle)),          \n\
                           0.75 * sin (angle) * sin (angle));   \n\
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
 vec4 SHBasis (const in vec3 dir) \n\
{ \n\
    float   phi = (abs (dir.y) == 1.0) ? 0.0 : atan (dir.z, dir.x); \n\
    float   theta = acos (dir.y); \n\
 \n\
    float   sin_phi   = sin (phi); \n\
    float   cos_phi   = cos (phi); \n\
    float   sin_theta = sin (theta); \n\
    float   cos_theta = cos (theta); \n\
 \n\
    float   L00  = 0.282094792; \n\
    float   L1_1 = 0.488602512 * sin_theta * sin_phi; \n\
    float   L10  = 0.488602512 * cos_theta; \n\
    float   L11  = 0.488602512 * sin_theta * cos_phi; \n\
 \n\
    // sh is in [-1,1] range \n\
    return vec4 (L11, L10, L1_1, L00); \n\
}";


// Functions for the projection of scalar and vector values to 4-band SH basis
char DRShader_SH::DRSH_SH_Projection[]="\n\
vec4 Intensity2SH (in vec3 dir, in float L) \n\
{ \n\
    return L * SHBasis (dir); \n\
} \n\
 \n\
void RGB2SH (in vec3 dir, in vec3 L, out vec4 sh_r, out vec4 sh_g, out vec4 sh_b) \n\
{ \n\
	vec4 sh = SHBasis (dir); \n\
 \n\
    sh_r = L.r * sh; \n\
    sh_g = L.g * sh; \n\
    sh_b = L.b * sh; \n\
}";

// Functions for the un-projection (reconstruction) of scalar and vector 
// values from a 4-band SH basis
char DRShader_SH::DRSH_SH_Unprojection[]="\n\
float SH2Intensity (in vec4 sh, in vec3 dir) \n\
{ \n\
    return 0.886227 * sh.a + 1.023328 * dot (sh.rgb, dir); \n\
} \n\
 \n\
vec3 SH2RGB (in vec4 sh_r, in vec4 sh_g, in vec4 sh_b, in vec3 dir) \n\
{ \n\
    return vec3 (0.886227 * sh_r.a + 1.023328 * dot (sh_r.rgb, dir), \n\
                 0.886227 * sh_g.a + 1.023328 * dot (sh_g.rgb, dir), \n\
                 0.886227 * sh_b.a + 1.023328 * dot (sh_b.rgb, dir)); \n\
}";

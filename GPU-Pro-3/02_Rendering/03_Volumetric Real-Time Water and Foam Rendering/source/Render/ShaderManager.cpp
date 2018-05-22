#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <iostream>
#include <assert.h>

#include <windows.h>
#include "GL/glew.h"
#include <GL/gl.h>

#include "../Render/ShaderManager.h"
#include "../Level/Light.h"

#include <GL/glut.h>

// -----------------------------------------------------------------------------
// ---------------------- ShaderManager::ShaderManager -------------------------
// -----------------------------------------------------------------------------
ShaderManager::ShaderManager(void)
{
}

// -----------------------------------------------------------------------------
// --------------------- ShaderManager::~ShaderManager -------------------------
// -----------------------------------------------------------------------------
ShaderManager::~ShaderManager(void)
{
	Exit();
}

// -----------------------------------------------------------------------------
// -------------------------- ShaderManager::Init ------------------------------
// -----------------------------------------------------------------------------
void ShaderManager::Init(Light* light)
{
	context = cgCreateContext();
	CheckForCgError("creating context");

	cgGLRegisterStates(context);
	CheckForCgError("registering standard CgFX states");
	cgGLSetManageTextureParameters(context, CG_TRUE);
	CheckForCgError("manage texture parameters");

	printf("loading shaders\n");

	LoadEffect("data/shaders/skyboxEffect.cgfx", SHADER_EFFECT_SKYBOX);
	LoadEffect("data/shaders/summedAreaTableEffect.cgfx", SHADER_EFFECT_SAT);
	LoadEffect("data/shaders/shadowMapEffect.cgfx", SHADER_EFFECT_SHADOW_MAP);
	LoadEffect("data/shaders/surfaceEffect.cgfx", SHADER_EFFECT_SURFACE);
	LoadEffect("data/shaders/fluidEffect.cgfx", SHADER_EFFECT_FLUID);
	LoadEffect("data/shaders/postProcessEffect.cgfx", SHADER_EFFECT_POST_PROCESS);

	printf("\n");

	cgPass = NULL;

	// SkyboxShader
	GetEffectParameter(0, SP_SKYBOX_FACE,			"skyboxFace");

	// SummedAreaTableShader
	GetEffectParameter(1, SP_SAT_MAP,			"satMap");
	GetEffectParameter(1, SP_SAT_ARRAY_MAP,		"satArrayMap");
	GetEffectParameter(1, SP_SAT_PASS_INDEX,	"satPassIndex");
	GetEffectParameter(1, SP_SAT_INDEX,			"satIndex");
	GetEffectParameter(1, SP_SAT_TEXEL_SIZE,	"satTexelSize");

	// ShadowMapShader
	GetEffectParameter(2, SP_SME_SHADOW_MAP,	"smeShadowMap");
	GetEffectParameter(2, SP_SME_CK_INDEX,		"smeCkIndex");
	GetEffectParameter(2, SP_SME_SIN_COS_FLAG,	"smeSinCosFlag");
	GetEffectParameter(2, SP_SME_CLIP_PLANES,	"smeClipPlanes");

	// SurfaceShader
	GetEffectParameter(3, SP_VIEW,					"viewMatrix");

	GetEffectParameter(3, SP_LIGHT_POS_OBJ_SPACE,	"lightPosObjSpace");
	GetEffectParameter(3, SP_EYE_POS_OBJ_SPACE,		"eyePosObjSpace");

	GetEffectParameter(3, SP_TEXTURE,				"textureMap");
	GetEffectParameter(3, SP_NORMAL_MAP,			"normalMap");
	GetEffectParameter(3, SP_SHADOW_MAP,			"shadowMap");

	GetEffectParameter(3, SP_SHADOW_MAP_TEXTURE_MATRIX,			"shadowMapTextureMatrix");
	GetEffectParameter(3, SP_SHADOW_MAP_LINEAR_TEXTURE_MATRIX,	"shadowMapLinearTextureMatrix");

	GetEffectParameter(3, SP_RECONSTRUCTION_ORDER,		"reconstructionOrder");
	GetEffectParameter(3, SP_RECONSTRUCTION_OFFSET,		"reconstructionOffset");
	GetEffectParameter(3, SP_SHADOW_MAP_SIZE,			"shadowMapSize");
	GetEffectParameter(3, SP_SHADOW_MAP_SIZE_SQUARED,	"shadowMapSizeSquared");
	GetEffectParameter(3, SP_SHADOW_MAP_TEXEL_SIZE,		"shadowMapTexelSize");
	GetEffectParameter(3, SP_SHADOW_MAP_FOV,			"shadowMapFOV");

	GetEffectParameter(3, SP_SIN_MAP,	"csmSinMap");
	GetEffectParameter(3, SP_COS_MAP,	"csmCosMap");

	GetEffectParameter(3, SP_CLIP_PLANES,	"clipPlanes");
	GetEffectParameter(3, SP_USE_MIP_MAPS,	"useMipMaps");

	GetEffectParameter(3, SP_KA,			"Ka");
	GetEffectParameter(3, SP_KD,			"Kd");
	GetEffectParameter(3, SP_KS,			"Ks");
	GetEffectParameter(3, SP_SHININESS,		"shininess");

	// FluidShader
	GetEffectParameter(4, SP_FLUID_RENDER_MODE,		"renderMode");

	GetEffectParameter(4, SP_FLUID_THICKNESS_MAP,		"thicknessMap");
	GetEffectParameter(4, SP_FLUID_FOAM_THICKNESS_MAP,	"foamThicknessMap");

	GetEffectParameter(4, SP_FLUID_DEPTH_MAP,			"depthMap");
	GetEffectParameter(4, SP_FLUID_FOAM_DEPTH_MAP,		"foamDepthMap");

	GetEffectParameter(4, SP_FLUID_DEPTH_MASK,			"depthMask");
	GetEffectParameter(4, SP_FLUID_NOISE_MAP,			"noiseMap");

	GetEffectParameter(4, SP_FLUID_PERLIN_MAP,			"perlinMap");
	GetEffectParameter(4, SP_FLUID_SCENE_MAP,			"sceneMap");
	GetEffectParameter(4, SP_FLUID_CUBE_MAP,			"cubeMap");

	GetEffectParameter(4, SP_FLUID_PARTICLE_SIZE,			"particleSize");
	GetEffectParameter(4, SP_FLUID_PARTICLE_SCALE,			"particleScale");
	GetEffectParameter(4, SP_FLUID_BUFFER_SIZE,				"bufferSize");
	GetEffectParameter(4, SP_FLUID_LIGHT_POS_EYE_SPACE,		"lightPosEyeSpace");

	GetEffectParameter(4, SP_FLUID_INV_VIEW,				"invView");
	GetEffectParameter(4, SP_FLUID_INV_PROJ,				"invProj");
	GetEffectParameter(4, SP_FLUID_INV_VIEWPROJ,			"invViewProj");
	
	GetEffectParameter(4, SP_FLUID_LOW_RES_FACTOR,			"lowResFactor");
	GetEffectParameter(4, SP_FLUID_INV_VIEWPORT,			"invViewport");
	GetEffectParameter(4, SP_FLUID_INV_FOCAL_LENGTH,		"invFocalLength");
	GetEffectParameter(4, SP_FLUID_INV_CAMERA,				"invCamera");
	GetEffectParameter(4, SP_FLUID_CAMERA,					"camera");
	GetEffectParameter(4, SP_FLUID_CURRENT_ITERATION,		"currentIteration");

	GetEffectParameter(4, SP_FLUID_BASE_COLOR,				"baseColor");
	GetEffectParameter(4, SP_FLUID_COLOR_FALLOFF,			"colorFalloff");
	GetEffectParameter(4, SP_FLUID_FALLOFF_SCALE,			"falloffScale");
	GetEffectParameter(4, SP_FLUID_SPECULAR_COLOR,			"fluidSpecularColor");
	GetEffectParameter(4, SP_FLUID_SHININESS,				"fluidShininess");
	GetEffectParameter(4, SP_FLUID_DENSITY_THRESHOLD,		"densityThreshold");
	GetEffectParameter(4, SP_FLUID_FRESNEL_BIAS,			"fresnelBias");
	GetEffectParameter(4, SP_FLUID_FRESNEL_SCALE,			"fresnelScale");
	GetEffectParameter(4, SP_FLUID_FRESNEL_POWER,			"fresnelPower");
	GetEffectParameter(4, SP_FLUID_THICKNESS_REFRACTION,	"thicknessRefraction");
	GetEffectParameter(4, SP_FLUID_FACE_SCALE,				"faceScale");
	GetEffectParameter(4, SP_FLUID_EPSILON,					"epsilon");
	GetEffectParameter(4, SP_FLUID_BLUR_DEPTH_FALLOFF,		"blurDepthFalloff");
	GetEffectParameter(4, SP_FLUID_THRESHOLDMIN,			"thresholdMin");
	GetEffectParameter(4, SP_FLUID_EULER_ITERATION_FACTOR,	"eulerIterationFactor");
	GetEffectParameter(4, SP_FLUID_DEPTH_THRESHOLD,			"depthThreshold");
	GetEffectParameter(4, SP_FLUID_USE_NOISE,				"useNoise");
	GetEffectParameter(4, SP_FLUID_NOISE_DEPTH_FALLOFF,		"noiseDepthFalloff");
	GetEffectParameter(4, SP_FLUID_NORMAL_NOISE_WEIGTH,		"normalNoiseWeight");

	GetEffectParameter(4, SP_FLUID_FOAM_BACK_COLOR,				"foamBackColor");
	GetEffectParameter(4, SP_FLUID_FOAM_FRONT_COLOR,			"foamFrontColor");
	GetEffectParameter(4, SP_FLUID_FOAM_FALLOFF_SCALE,			"foamFalloffScale");
	GetEffectParameter(4, SP_FLUID_FOAM_DEPTH_THRESHOLD,		"foamDepthThreshold");
	GetEffectParameter(4, SP_FLUID_FOAM_FRONT_FALLOFF_SCALE,	"foamFrontFalloffScale");

	GetEffectParameter(4, SP_FLUID_SCALED_DOWN_SIZE, "scaledDownSize");

	GetEffectParameter(4, SP_FLUID_SHADOW_MAP_TEXTURE_MATRIX,			"shadowMapTextureMatrix");
	GetEffectParameter(4, SP_FLUID_SHADOW_MAP_LINEAR_TEXTURE_MATRIX,	"shadowMapLinearTextureMatrix");

	GetEffectParameter(4, SP_FLUID_RECONSTRUCTION_ORDER,	"reconstructionOrder");
	GetEffectParameter(4, SP_FLUID_RECONSTRUCTION_OFFSET,	"reconstructionOffset");
	GetEffectParameter(4, SP_FLUID_SHADOW_MAP_SIZE,			"shadowMapSize");
	GetEffectParameter(4, SP_FLUID_SHADOW_MAP_SIZE_SQUARED,	"shadowMapSizeSquared");
	GetEffectParameter(4, SP_FLUID_SHADOW_MAP_TEXEL_SIZE,	"shadowMapTexelSize");
	GetEffectParameter(4, SP_FLUID_SHADOW_MAP_FOV,			"shadowMapFOV");

	GetEffectParameter(4, SP_FLUID_SIN_MAP,	"csmSinMap");
	GetEffectParameter(4, SP_FLUID_COS_MAP,	"csmCosMap");

	GetEffectParameter(4, SP_FLUID_CLIP_PLANES,		"clipPlanes");
	GetEffectParameter(4, SP_FLUID_USE_MIP_MAPS,	"useMipMaps");

	// PostProcessShader
	GetEffectParameter(5, SP_POST_SCENE_MAP,		"postSceneMap");
}

// -----------------------------------------------------------------------------
// ---------------------- ShaderManager::EnableShader --------------------------
// -----------------------------------------------------------------------------
void ShaderManager::EnableShader(SHADER_EFFECT effect, const char* passName)
{
	assert(std::string(passName).compare("") != 0);

	cgPass = cgGetNamedPass(cgTechniques[effect], passName);
	cgSetPassState(cgPass);
}

// -----------------------------------------------------------------------------
// ---------------------- ShaderManager::DisableShader -------------------------
// -----------------------------------------------------------------------------
void ShaderManager::DisableShader()
{
	if (cgPass)
		cgResetPassState(cgPass);

	glActiveTextureARB(GL_TEXTURE0);
	glClientActiveTextureARB(GL_TEXTURE0);
	glEnable(GL_TEXTURE_2D);
}

// -----------------------------------------------------------------------------
// ---------------------- ShaderManager::EnableTextureParameter ----------------
// -----------------------------------------------------------------------------
void ShaderManager::EnableTextureParameter(SHADER_PARAMETER parameter)
{
	cgGLEnableTextureParameter(shaderParameters[parameter]);
	CheckForCgError("EnableTextureParameter");
}

// -----------------------------------------------------------------------------
// ------------------ ShaderManager::SetTextureParameter -----------------------
// -----------------------------------------------------------------------------
void ShaderManager::SetParameterTexture(SHADER_PARAMETER parameter, unsigned int texture)
{
	cgGLSetTextureParameter(shaderParameters[parameter], texture);
	CheckForCgError("SetParameterTexture");
}

// -----------------------------------------------------------------------------
// --------------------- ShaderManager::SetParameter1f -------------------------
// -----------------------------------------------------------------------------
void ShaderManager::SetParameter1f(SHADER_PARAMETER parameter, float x)
{
	cgSetParameter1f(shaderParameters[parameter], x);
	CheckForCgError("SetParameter1f");
}

// -----------------------------------------------------------------------------
// --------------------- ShaderManager::SetParameter2f -------------------------
// -----------------------------------------------------------------------------
void ShaderManager::SetParameter2f(SHADER_PARAMETER parameter, const float* v)
{
	cgSetParameter2fv(shaderParameters[parameter], v);
	CheckForCgError("SetParameter2f");
}

// -----------------------------------------------------------------------------
// -------------------- ShaderManager::SetParameter3fv -------------------------
// -----------------------------------------------------------------------------
void ShaderManager::SetParameter3fv(SHADER_PARAMETER parameter, const float* v)
{
	cgSetParameter3fv(shaderParameters[parameter], v);
	CheckForCgError("SetParameter3fv");
}

// -----------------------------------------------------------------------------
// -------------------- ShaderManager::SetParameter4fv -------------------------
// -----------------------------------------------------------------------------
void ShaderManager::SetParameter4fv(SHADER_PARAMETER parameter, const float* v)
{
	cgSetParameter4fv(shaderParameters[parameter], v);
	CheckForCgError("SetParameter4fv");
}

// -----------------------------------------------------------------------------
// ----------------- ShaderManager::SetMatrixParameterfc -----------------------
// -----------------------------------------------------------------------------
void ShaderManager::SetMatrixParameterfc(SHADER_PARAMETER parameter, float* matrix)
{
	cgSetMatrixParameterfc(shaderParameters[parameter], matrix);
	CheckForCgError("SetMatrixParameterfc");
}

// -----------------------------------------------------------------------------
// -------------------------- ShaderManager::Exit ------------------------------
// -----------------------------------------------------------------------------
void ShaderManager::Exit(void)
{
	unsigned int i;
	for (i=0; i<SHADER_EFFECT_COUNT; i++)
		if (cgEffects[i])
			cgDestroyEffect(cgEffects[i]);

	cgDestroyContext(context);
}

// -----------------------------------------------------------------------------
// ------------------------- ShaderManager::LoadEffect -------------------------
// -----------------------------------------------------------------------------
void ShaderManager::LoadEffect(const char* fileName, int idx)
{
	cgEffects[idx] = cgCreateEffectFromFile(context, fileName, NULL);
	CheckForCgError("creating effectShader.cgfx effect");
	assert(cgEffects[idx]);

	cgTechniques[idx] = cgGetFirstTechnique(cgEffects[idx]);

#if _DEBUG
	/*
	while (cgTechniques[idx] && cgValidateTechnique(cgTechniques[idx]) == CG_FALSE)
	{
		printf("Technique %s did not validate.  Skipping.\n", cgGetTechniqueName(cgTechniques[idx]));
		cgTechniques[idx] = cgGetNextTechnique(cgTechniques[idx]);
	}
	*/
#endif

	if (cgTechniques[idx]) {
		printf("\t%s\n",	cgGetTechniqueName(cgTechniques[idx]));
	} else {
		printf("No valid technique\n");
		assert(false);
	}
}

// -----------------------------------------------------------------------------
// --------------------- ShaderManager::GetEffectParameter ---------------------
// -----------------------------------------------------------------------------
void ShaderManager::GetEffectParameter(SHADER_PARAMETER parameter, const char* parameterName)
{
	unsigned int i;
	for (i=0; i<SHADER_EFFECT_COUNT; i++)
	{
		shaderParameters[parameter] = cgGetNamedEffectParameter(cgEffects[i], parameterName);
		CheckForCgError("Could not get effect parameter");

		// return if parameter was found
		if (shaderParameters[parameter] != 0)
			return;
	}

	printf("!!! Could not get effect parameter (%s) !!!\n", parameterName);
	assert(false);
}

void ShaderManager::GetEffectParameter(unsigned int index, SHADER_PARAMETER parameter, const char* parameterName)
{
	assert(index < SHADER_EFFECT_COUNT);

	shaderParameters[parameter] = cgGetNamedEffectParameter(cgEffects[index], parameterName);
	CheckForCgError("Could not get effect parameter");

	// return if parameter was found
	if (shaderParameters[parameter] != 0)
		return;

	printf("!!! Could not get effect parameter (%s) !!!\n", parameterName);
	assert(false);
}

// -----------------------------------------------------------------------------
// --------------------- ShaderManager::CheckForCgError ------------------------
// -----------------------------------------------------------------------------
void ShaderManager::CheckForCgError(const char *situation)
{
	CGerror error;
	const char *string = cgGetLastErrorString(&error);

	if (error != CG_NO_ERROR) {
		printf("%s: %s: %s\n",
			"vertex_and_fragment_program", situation, string);
		if (error == CG_COMPILER_ERROR) {
			printf("%s\n", cgGetLastListing(context));
		}
	}
}
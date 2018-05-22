#ifndef __SHADERMANAGER__H__
#define __SHADERMANAGER__H__

#include <Cg/cg.h>
#include <Cg/cgGL.h>

#include "../Util/Singleton.h"

class Light;

class ShaderManager : public Singleton<ShaderManager>
{
	friend class Singleton<ShaderManager>;

public:
	enum SHADER_EFFECT
	{
		SHADER_EFFECT_SKYBOX=0,
		SHADER_EFFECT_SAT,
		SHADER_EFFECT_SHADOW_MAP,
		SHADER_EFFECT_SURFACE,
		SHADER_EFFECT_FLUID,
		SHADER_EFFECT_POST_PROCESS,
		SHADER_EFFECT_COUNT
	};

	enum SHADER_PARAMETER
	{
		//////////////////////////////////////////////////////////////////////////
		SP_SKYBOX_FACE=0,
		//////////////////////////////////////////////////////////////////////////
		SP_SAT_MAP,
		SP_SAT_ARRAY_MAP,
		SP_SAT_PASS_INDEX,
		SP_SAT_INDEX,
		SP_SAT_TEXEL_SIZE,
		//////////////////////////////////////////////////////////////////////////
		SP_SME_SHADOW_MAP,
		SP_SME_CK_INDEX,
		SP_SME_SIN_COS_FLAG,
		SP_SME_CLIP_PLANES,
		//////////////////////////////////////////////////////////////////////////
		SP_VIEW,
		///
		SP_LIGHT_POS_OBJ_SPACE,
		SP_EYE_POS_OBJ_SPACE,
		///
		SP_TEXTURE,
		SP_NORMAL_MAP,
		SP_SHADOW_MAP,
		///
		SP_SHADOW_MAP_TEXTURE_MATRIX,
		SP_SHADOW_MAP_LINEAR_TEXTURE_MATRIX,
		///
		SP_RECONSTRUCTION_ORDER,
		SP_RECONSTRUCTION_OFFSET,
		SP_SHADOW_MAP_SIZE,
		SP_SHADOW_MAP_SIZE_SQUARED,
		SP_SHADOW_MAP_TEXEL_SIZE,
		SP_SHADOW_MAP_FOV,
		///
		SP_SIN_MAP,
		SP_COS_MAP,
		///
		SP_CLIP_PLANES,
		SP_USE_MIP_MAPS,
		///
		SP_KA,
		SP_KD,
		SP_KS,
		SP_SHININESS,
		//////////////////////////////////////////////////////////////////////////
		SP_FLUID_RENDER_MODE,
		///
		SP_FLUID_THICKNESS_MAP,
		SP_FLUID_FOAM_THICKNESS_MAP,

		SP_FLUID_DEPTH_MAP,
		SP_FLUID_FOAM_DEPTH_MAP,

		SP_FLUID_DEPTH_MASK,
		SP_FLUID_NOISE_MAP,

		SP_FLUID_PERLIN_MAP,
		SP_FLUID_SCENE_MAP,
		SP_FLUID_CUBE_MAP,
		///
		SP_FLUID_PARTICLE_SIZE,
		SP_FLUID_PARTICLE_SCALE,
		SP_FLUID_BUFFER_SIZE,
		SP_FLUID_LIGHT_POS_EYE_SPACE,

		SP_FLUID_INV_VIEW,
		SP_FLUID_INV_PROJ,
		SP_FLUID_INV_VIEWPROJ,
		
		SP_FLUID_LOW_RES_FACTOR,
		SP_FLUID_INV_VIEWPORT,
		SP_FLUID_INV_FOCAL_LENGTH,
		SP_FLUID_INV_CAMERA,
		SP_FLUID_CAMERA,
		SP_FLUID_CURRENT_ITERATION,
		///
		SP_FLUID_BASE_COLOR,
		SP_FLUID_COLOR_FALLOFF,
		SP_FLUID_FALLOFF_SCALE,
		SP_FLUID_SPECULAR_COLOR,
		SP_FLUID_SHININESS,
		SP_FLUID_DENSITY_THRESHOLD,
		SP_FLUID_FRESNEL_BIAS,
		SP_FLUID_FRESNEL_SCALE,
		SP_FLUID_FRESNEL_POWER,
		SP_FLUID_THICKNESS_REFRACTION,
		SP_FLUID_FACE_SCALE,
		SP_FLUID_EPSILON,
		SP_FLUID_BLUR_DEPTH_FALLOFF,
		SP_FLUID_THRESHOLDMIN,
		SP_FLUID_EULER_ITERATION_FACTOR,
		SP_FLUID_DEPTH_THRESHOLD,
		SP_FLUID_USE_NOISE,
		SP_FLUID_NOISE_DEPTH_FALLOFF,
		SP_FLUID_NORMAL_NOISE_WEIGTH,
		///
		SP_FLUID_FOAM_BACK_COLOR,
		SP_FLUID_FOAM_FRONT_COLOR,
		SP_FLUID_FOAM_FALLOFF_SCALE,
		SP_FLUID_FOAM_DEPTH_THRESHOLD,
		SP_FLUID_FOAM_FRONT_FALLOFF_SCALE,
		///
		SP_FLUID_SCALED_DOWN_SIZE,
		///
		SP_FLUID_SHADOW_MAP_TEXTURE_MATRIX,
		SP_FLUID_SHADOW_MAP_LINEAR_TEXTURE_MATRIX,
		///
		SP_FLUID_RECONSTRUCTION_ORDER,
		SP_FLUID_RECONSTRUCTION_OFFSET,
		SP_FLUID_SHADOW_MAP_SIZE,
		SP_FLUID_SHADOW_MAP_SIZE_SQUARED,
		SP_FLUID_SHADOW_MAP_TEXEL_SIZE,
		SP_FLUID_SHADOW_MAP_FOV,
		///
		SP_FLUID_SIN_MAP,
		SP_FLUID_COS_MAP,
		///
		SP_FLUID_CLIP_PLANES,
		SP_FLUID_USE_MIP_MAPS,
		//////////////////////////////////////////////////////////////////////////
		SP_POST_SCENE_MAP,
		//////////////////////////////////////////////////////////////////////////

		SP_PARAMETER_COUNT
	};

	ShaderManager(void);
	~ShaderManager(void);

	// inits the shader manager
	void Init(Light* light);

	// enables a shader
	void EnableShader(SHADER_EFFECT effect, const char* passName = "");

	// disables a shader
	void DisableShader();

	// enables a texture parameter
	void EnableTextureParameter(SHADER_PARAMETER parameter);

	// sets a texture parameter
	void SetParameterTexture(SHADER_PARAMETER parameter, unsigned int texture);

	/// Sets a float parameter with 1 components
	void SetParameter1f(SHADER_PARAMETER parameter, float x);

	/// Sets a float parameter with 2 components
	void SetParameter2f(SHADER_PARAMETER parameter, const float* v);

	/// Sets a float parameter with 3 components
	void SetParameter3fv(SHADER_PARAMETER parameter, const float* v);

	/// Sets a float parameter with 4 components
	void SetParameter4fv(SHADER_PARAMETER parameter, const float* v);

	// sets a matrix
	void SetMatrixParameterfc(SHADER_PARAMETER parameter, float* matrix);

	// exits the shader manager
	void Exit(void);

private:

	/// Load effect from file
	void LoadEffect(const char* fileName, int idx);

	/// Get named parameter from effect
	void GetEffectParameter(SHADER_PARAMETER parameter, const char* parameterName);
	void GetEffectParameter(unsigned int index, SHADER_PARAMETER parameter, const char* parameterName);

	/// Check errors
	void CheckForCgError(const char *situation);

	CGcontext context;
	//CGprofile vertexProfile;
	//CGprofile fragmentProfile;

	//////////////////////////////////

	CGeffect cgEffects[SHADER_EFFECT_COUNT];
	CGtechnique cgTechniques[SHADER_EFFECT_COUNT];
	CGpass cgPass;

	CGparameter shaderParameters[SP_PARAMETER_COUNT];
};

#endif
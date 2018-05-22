#ifndef DIRECTIONAL_LIGHT_H
#define DIRECTIONAL_LIGHT_H

#include <ILIGHT.h>

class DX11_RASTERIZER_STATE;
class DX11_DEPTH_STENCIL_STATE;
class DX11_BLEND_STATE;
class RENDER_TARGET_CONFIG;
class DX11_SHADER;

// DIRECTIONAL_LIGHT
//   For direct illumination a full-screen quad is rendered deferred into the accumulation render-target 
//   of the GBuffer. In order to cast shadows a shadow map is generated. For indirect illumination the 
//   voxel-grid of the GLOBAL_ILLUM post-processor is illuminated, whereby the shadow map, that was generated
//   for direct illumination, is reused.
class DIRECTIONAL_LIGHT: public ILIGHT
{
public:
  DIRECTIONAL_LIGHT()
	{
		multiplier = 0.0f;
		invShadowMapSize = 0.0f;
		lightShader = NULL;
		shadowMapShader = NULL;
		lightGridShaders[FINE_GRID] = NULL;
    lightGridShaders[COARSE_GRID] = NULL;
		uniformBuffer = NULL;
		noneCullRS = NULL;
		backCullRS = NULL;
		defaultDSS = NULL;
		noDepthTestDSS = NULL;
		noColorBS = NULL;
		blendBS = NULL;
		rtConfig = NULL;
		frustumRadius = 0.0f;
		frustumRatio = 0.0f; 
	}

	bool Create(const VECTOR3D &direction,const COLOR &color,float multiplier);

	lightTypes GetLightType() const;

	virtual void Update();

	virtual void SetupShadowMapSurface(SURFACE *surface);

	virtual void AddLitSurface();

	virtual void AddGridSurfaces();

	virtual DX11_UNIFORM_BUFFER* GetUniformBuffer() const;

	void SetDirection(VECTOR3D &direction);

	VECTOR3D GetDirection() const
	{
		return direction;
	}

	void SetColor(COLOR &color);

	COLOR GetColor() const
	{
		return color;
	}

	void SetMultiplier(float multiplier);

	float GetMultiplier() const
	{
		return multiplier;
	}

private:
	virtual void CalculateMatrices();

	virtual void UpdateUniformBuffer();

	void CalculateFrustum();

	// data for directional light uniform-buffer
	VECTOR3D direction;
	float multiplier;
	COLOR color;
	MATRIX4X4 shadowViewProjMatrix;
	MATRIX4X4 shadowViewProjTexMatrix;
	float invShadowMapSize;

	DX11_SHADER *lightShader;
	DX11_SHADER *shadowMapShader;
	DX11_SHADER *lightGridShaders[2];
	DX11_UNIFORM_BUFFER *uniformBuffer;
	DX11_RASTERIZER_STATE *noneCullRS;
	DX11_RASTERIZER_STATE *backCullRS;
	DX11_DEPTH_STENCIL_STATE *defaultDSS;
	DX11_DEPTH_STENCIL_STATE *noDepthTestDSS;
	DX11_BLEND_STATE *noColorBS;
	DX11_BLEND_STATE *blendBS;
	RENDER_TARGET_CONFIG *rtConfig;

	MATRIX4X4 shadowTexMatrix;
	MATRIX4X4 shadowProjMatrix;	
	float frustumRadius;
	float frustumRatio; 
	
};

#endif

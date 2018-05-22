#ifndef POINT_LIGHT_H
#define POINT_LIGHT_H

#include <ILIGHT.h>

class DX11_SHADER;
class DX11_RASTERIZER_STATE;
class DX11_DEPTH_STENCIL_STATE;
class DX11_BLEND_STATE;
class RENDER_TARGET_CONFIG;

// POINT_LIGHT
//   For direct illumination a sphere geometry is rendered deferred into the accumulation render-target
//   of the GBuffer. For indirect illumination the voxel-grid of the GLOBAL_ILLUM post-processor is 
//   illuminated. 
//   Since often for point- and spots-lights shadows can be abandoned without causing unpleasant visual 
//   effects, in this demo point-lights do not cast shadows. However especially for large point-lights 
//   shadow maps have to be used same as for directional lights. According to directional lights the 
//   shadow maps, that have been generated for direct illumination, are reused for indirect illumination.
class POINT_LIGHT: public ILIGHT
{
public:
	POINT_LIGHT()
	{
		radius = 0.0f;
		multiplier = 0.0f;
    lightShader = NULL;
		lightGridShaders[FINE_GRID] = NULL;
		lightGridShaders[COARSE_GRID] = NULL;
		uniformBuffer = NULL;
		backCullRS = NULL;
		frontCullRS = NULL;
		noDepthWriteDSS = NULL;
		noDepthTestDSS = NULL;
		blendBS = NULL;
		rtConfig = NULL;
		cameraInVolume = false;
	}

	bool Create(const VECTOR3D &position,float radius,const COLOR &color,float multiplier);

	lightTypes GetLightType() const;

	virtual void Update();

	virtual void SetupShadowMapSurface(SURFACE *surface);

	virtual void AddLitSurface();

	virtual void AddGridSurfaces();

	virtual DX11_UNIFORM_BUFFER* GetUniformBuffer() const;

	void SetPosition(VECTOR3D &position);

	VECTOR3D GetPosition() const
	{
		return position;
	}

	void SetRadius(float radius);

	float GetRadius() const
	{
		return radius;
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
	bool IsSphereInVolume(const VECTOR3D &position,float radius);

	virtual void CalculateMatrices();

	virtual void UpdateUniformBuffer();

	// data for point-light uniform-buffer
	VECTOR3D position;
	float radius;
	COLOR color;
	MATRIX4X4 worldMatrix;
	float multiplier;

	DX11_SHADER *lightShader;
	DX11_SHADER *lightGridShaders[2];
	DX11_UNIFORM_BUFFER *uniformBuffer;
	DX11_RASTERIZER_STATE *backCullRS;
	DX11_RASTERIZER_STATE *frontCullRS;	
	DX11_DEPTH_STENCIL_STATE *noDepthWriteDSS;
	DX11_DEPTH_STENCIL_STATE *noDepthTestDSS;
	DX11_BLEND_STATE *blendBS;
	RENDER_TARGET_CONFIG *rtConfig;
	bool cameraInVolume; 

};

#endif

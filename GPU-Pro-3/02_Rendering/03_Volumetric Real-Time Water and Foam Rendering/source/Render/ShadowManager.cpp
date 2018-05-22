#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <windows.h>
#include "GL/glew.h"
#include <GL/gl.h>

#include <stdio.h>
#include <assert.h>

#include "ShadowManager.h"

#include "../Level/Camera.h"

#include "../Util/Math.h"
#include "../Util/Matrix4.h"
#include "../Util/Vector2.h"
#include "../Util/Vector4.h"
#include "../Util/ConfigLoader.h"

#include "../Render/RenderManager.h"
#include "../Render/FilterManager.h"
#include "../Render/ShaderManager.h"
#include "../Render/ShadowMap.h"

#include "../Main/DemoManager.h"

#include "../Input/InputManager.h"

#include <GL/glut.h>

GLenum buffers[] = {GL_COLOR_ATTACHMENT0_EXT,
					GL_COLOR_ATTACHMENT1_EXT,
					GL_COLOR_ATTACHMENT2_EXT,
					GL_COLOR_ATTACHMENT3_EXT,
					GL_COLOR_ATTACHMENT4_EXT,
					GL_COLOR_ATTACHMENT5_EXT,
					GL_COLOR_ATTACHMENT6_EXT,
					GL_COLOR_ATTACHMENT7_EXT
};

// -----------------------------------------------------------------------------
// ----------------------- ShadowManager::ShadowManager ------------------------
// -----------------------------------------------------------------------------
ShadowManager::ShadowManager(void) :
	isInizialized(false),
	mapSize(512),
	reconstructionOrder(16.0f)
{
}

// -----------------------------------------------------------------------------
// ----------------------------- ShadowManager::Init ---------------------------
// -----------------------------------------------------------------------------
void ShadowManager::Init(unsigned int _mapSize, float _reconstructionOrder)
{
	bool useMipMaps = true;

	assert(!isInizialized);
	assert((_reconstructionOrder >= 4.0f) && (_reconstructionOrder <= 16.0f));
	assert(((int)_reconstructionOrder % 4) == 0);

	mapSize = _mapSize;
	reconstructionOrder = _reconstructionOrder;

	//unsigned int recOrder = (int)(reconstructionOrder/4.0f);

	FilterManager::Instance()->Init(mapSize, reconstructionOrder, useMipMaps);

	//////////////////////////////////////////////////////////////////////////
	
	lights.push_back(DemoManager::Instance()->GetLight());
	shadowMaps.push_back(new ShadowMap(mapSize, reconstructionOrder, useMipMaps));

	//////////////////////////////////////////////////////////////////////////

	float smSize = (float)mapSize;

	Vector2 reconstructionOffset;
	if (Math::IsEqual(reconstructionOrder, 4.0f))
		reconstructionOffset = Vector2(0.0763f, 0.00763f);
	else if (Math::IsEqual(reconstructionOrder, 8.0f))
		reconstructionOffset = Vector2(0.05693333f, 0.005693333f);
	else if (Math::IsEqual(reconstructionOrder, 12.0f))
		reconstructionOffset = Vector2(0.03756666f, 0.003756666f);
	else
		reconstructionOffset = Vector2(0.0182f, 0.00182f);

	ShaderManager::Instance()->SetParameter1f(ShaderManager::SP_RECONSTRUCTION_ORDER, reconstructionOrder);
	ShaderManager::Instance()->SetParameter2f(ShaderManager::SP_RECONSTRUCTION_OFFSET, reconstructionOffset.comp);
	ShaderManager::Instance()->SetParameter1f(ShaderManager::SP_SHADOW_MAP_SIZE, smSize);
	ShaderManager::Instance()->SetParameter1f(ShaderManager::SP_SHADOW_MAP_SIZE_SQUARED, smSize*smSize);
	ShaderManager::Instance()->SetParameter1f(ShaderManager::SP_SHADOW_MAP_TEXEL_SIZE, 1.0f/smSize);
	ShaderManager::Instance()->SetParameter1f(ShaderManager::SP_USE_MIP_MAPS, (float)useMipMaps);

	ShaderManager::Instance()->SetParameter1f(ShaderManager::SP_SAT_TEXEL_SIZE, 1.0f/smSize);

	// fluid parameters
	ShaderManager::Instance()->SetParameter1f(ShaderManager::SP_FLUID_RECONSTRUCTION_ORDER, reconstructionOrder);
	ShaderManager::Instance()->SetParameter2f(ShaderManager::SP_FLUID_RECONSTRUCTION_OFFSET, reconstructionOffset.comp);
	ShaderManager::Instance()->SetParameter1f(ShaderManager::SP_FLUID_SHADOW_MAP_SIZE, smSize);
	ShaderManager::Instance()->SetParameter1f(ShaderManager::SP_FLUID_SHADOW_MAP_SIZE_SQUARED, smSize*smSize);
	ShaderManager::Instance()->SetParameter1f(ShaderManager::SP_FLUID_SHADOW_MAP_TEXEL_SIZE, 1.0f/smSize);
	ShaderManager::Instance()->SetParameter1f(ShaderManager::SP_FLUID_USE_MIP_MAPS, (float)useMipMaps);
	//////////////////////////////////////////////////////////////////////////

	isInizialized = true;
}

// -----------------------------------------------------------------------------
// ------------------------- ShadowManager::Update -----------------------------
// -----------------------------------------------------------------------------
void ShadowManager::Update(float deltaTime)
{
	
}

// -----------------------------------------------------------------------------
// -------------------- ShadowManager::UpdateSizeOfLight -----------------------
// -----------------------------------------------------------------------------
void ShadowManager::UpdateSizeOfLight(float sizeOfLight)
{
	unsigned int i;
	for (i=0; i<lights.size(); i++)
		lights[i]->SetLightRadiusWorld(sizeOfLight);
}

// -----------------------------------------------------------------------------
// ------------------------- ShadowManager::Exit -------------------------------
// -----------------------------------------------------------------------------
void ShadowManager::Exit(void)
{
	assert(isInizialized);

	FilterManager::Instance()->Exit();

	unsigned int i;

	for (i=0; i<shadowMaps.size(); i++)
	{
		if (shadowMaps[i])
			delete shadowMaps[i];
	}
	shadowMaps.clear();

	isInizialized = false;
}

// -----------------------------------------------------------------------------
// ----------------------- ShadowManager::BeginShadow --------------------------
// -----------------------------------------------------------------------------
void ShadowManager::BeginShadow(void)
{
	assert(isInizialized);

	glPushAttrib(GL_VIEWPORT_BIT);
	glShadeModel(GL_FLAT);

	//glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

	//glCullFace(GL_FRONT);

	// Set the viewport
	glViewport(0, 0, mapSize, mapSize);

	RenderManager::Instance()->SkipMaterials(true);
}

// -----------------------------------------------------------------------------
// ----------------------- ShadowManager::ShadowPass ---------------------------
// -----------------------------------------------------------------------------
void ShadowManager::ShadowPass(unsigned int passCount)
{
	assert(passCount < shadowMaps.size());

	shadowMaps[passCount]->SetFrameBuffer();

	float nearPlane = lights[passCount]->GetNearPlane();
	float farPlane = lights[passCount]->GetFarPlane();

	float planes[3];
	planes[0] = nearPlane;
	planes[1] = farPlane;
	planes[2] = nearPlane-farPlane;

	ShaderManager::Instance()->SetParameter3fv(ShaderManager::SP_SME_CLIP_PLANES, planes);
	ShaderManager::Instance()->EnableShader(ShaderManager::SHADER_EFFECT_SHADOW_MAP, "ComputeShadowMap");

	// Set ModelView Matrix
	glLoadIdentity();
	glMultMatrixf(lights[passCount]->GetViewMatrix().entry);

	// Set Projection Matrix
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glMultMatrixf(lights[passCount]->GetProjectionMatrix().entry);
	glMatrixMode(GL_MODELVIEW);

	// Clear buffers
	glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
}

// -----------------------------------------------------------------------------
// ------------------------ ShadowManager::EndShadow ---------------------------
// -----------------------------------------------------------------------------
void ShadowManager::EndShadow(void)
{
	assert(isInizialized);

	ShaderManager::Instance()->DisableShader();
	RenderManager::Instance()->SkipMaterials(false);

	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);

	glShadeModel(GL_SMOOTH);
	glPopAttrib();
}

// -----------------------------------------------------------------------------
// --------------------- ShadowManager::PrepareFinalPass -----------------------
// -----------------------------------------------------------------------------
void ShadowManager::PrepareFinalPass(Matrix4& viewMatrix, unsigned int passCount)
{
	Matrix4 bias;
	bias.SetTranslation(0.5f, 0.5f, 0.5f);
	bias.SetScale(0.5f, 0.5f, 0.5f);

	Matrix4 invViewMatrix = viewMatrix.Inverse();

	Matrix4 textureMatrix =
		invViewMatrix *
		lights[passCount]->GetViewMatrix() *
		lights[passCount]->GetProjectionMatrix() *
		bias;

	Matrix4 linearTextureMatrix =
		invViewMatrix *
		lights[passCount]->GetViewMatrix();

	ShaderManager::Instance()->SetParameterTexture(ShaderManager::SP_SHADOW_MAP, shadowMaps[passCount]->GetShadowMap());

	ShaderManager::Instance()->SetParameter1f(ShaderManager::SP_SHADOW_MAP_FOV, lights[passCount]->GetFieldOfView()*0.5f);
	ShaderManager::Instance()->SetMatrixParameterfc(ShaderManager::SP_SHADOW_MAP_TEXTURE_MATRIX, textureMatrix.entry);
	ShaderManager::Instance()->SetMatrixParameterfc(ShaderManager::SP_SHADOW_MAP_LINEAR_TEXTURE_MATRIX, linearTextureMatrix.entry);

	ShaderManager::Instance()->SetParameterTexture(ShaderManager::SP_SIN_MAP, shadowMaps[passCount]->GetSinTexture());
	ShaderManager::Instance()->SetParameterTexture(ShaderManager::SP_COS_MAP, shadowMaps[passCount]->GetCosTexture());

	//ShaderManager::Instance()->SetParameter1f(ShaderManager::SP_SIZE_OF_LIGHT, lights[passCount]->GetLightRadiusWorld());

	// set light stuff
	float nearPlane = lights[passCount]->GetNearPlane();
	float farPlane = lights[passCount]->GetFarPlane();

	float planes[3];
	planes[0] = nearPlane;
	planes[1] = farPlane;
	planes[2] = nearPlane-farPlane;

	ShaderManager::Instance()->SetParameter3fv(ShaderManager::SP_CLIP_PLANES, planes);

	// fluid parameters
	ShaderManager::Instance()->SetMatrixParameterfc(ShaderManager::SP_FLUID_SHADOW_MAP_TEXTURE_MATRIX, textureMatrix.entry);
	ShaderManager::Instance()->SetMatrixParameterfc(ShaderManager::SP_FLUID_SHADOW_MAP_LINEAR_TEXTURE_MATRIX, linearTextureMatrix.entry);

	ShaderManager::Instance()->SetParameter1f(ShaderManager::SP_FLUID_SHADOW_MAP_FOV, lights[passCount]->GetFieldOfView()*0.5f);

	ShaderManager::Instance()->SetParameterTexture(ShaderManager::SP_FLUID_SIN_MAP, shadowMaps[passCount]->GetSinTexture());
	ShaderManager::Instance()->SetParameterTexture(ShaderManager::SP_FLUID_COS_MAP, shadowMaps[passCount]->GetCosTexture());

	ShaderManager::Instance()->SetParameter3fv(ShaderManager::SP_FLUID_CLIP_PLANES, planes);
	//////////////////////////////////////////////////////////////////////////
}

// -----------------------------------------------------------------------------
// ------------------- ShadowManager::ComputeConvolutionMap --------------------
// -----------------------------------------------------------------------------
void ShadowManager::ComputeConvolutionMap(void)
{
	assert(isInizialized);

	glPushAttrib(GL_VIEWPORT_BIT | GL_COLOR_BUFFER_BIT);

	// Clear buffers
	//glClearColor (0.0, 0.0, 0.0, 1.0);
	glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);

	// Set the viewport
	glViewport(0, 0, mapSize, mapSize);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0, mapSize, mapSize, 0, -100, 100);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	//////////////////////////////////////////////////////////////////////////

	unsigned int i;
	for (i=0; i<shadowMaps.size(); i++)
	{
		ShaderManager::Instance()->SetParameterTexture(ShaderManager::SP_SME_SHADOW_MAP, shadowMaps[i]->GetShadowMap());

		shadowMaps[i]->ComputeCsm();
		shadowMaps[i]->PreFilter();
	}

	//////////////////////////////////////////////////////////////////////////

	glPopAttrib();
}

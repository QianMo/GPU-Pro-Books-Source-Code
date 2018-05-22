#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <windows.h>
#include "GL/glew.h"
#include <GL/gl.h>

#include <stdio.h>
#include <assert.h>

#include "ShadowMap.h"

#include "../Util/Math.h"
#include "../Util/Matrix4.h"
#include "../Util/Vector2.h"
#include "../Util/Vector4.h"

#include "../Render/ShaderManager.h"
#include "../Render/FilterManager.h"

#include <GL/glut.h>

GLenum colorAttachments[] = {GL_COLOR_ATTACHMENT0_EXT,
	GL_COLOR_ATTACHMENT1_EXT,
	GL_COLOR_ATTACHMENT2_EXT,
	GL_COLOR_ATTACHMENT3_EXT,
	GL_COLOR_ATTACHMENT4_EXT,
	GL_COLOR_ATTACHMENT5_EXT,
	GL_COLOR_ATTACHMENT6_EXT,
	GL_COLOR_ATTACHMENT7_EXT
};

// -----------------------------------------------------------------------------
// ---------------------------- ShadowMap::ShadowMap ---------------------------
// -----------------------------------------------------------------------------
ShadowMap::ShadowMap(unsigned int _mapSize, float _reconstructionOrder, bool _useMipMaps) :
	frameBuffer(0),
	csmSinBuffer(0),
	csmCosBuffer(0),
	shadowMapTexture(0),
	linearShadowMapTexture(0),
	csmSinTextureArray(0),
	csmCosTextureArray(0)
{
	assert((_reconstructionOrder >= 4.0f) && (_reconstructionOrder <= 16.0f));
	assert(((int)_reconstructionOrder % 4) == 0);

	mapSize = _mapSize;
	reconstructionOrder = _reconstructionOrder;
	useMipMaps = _useMipMaps;

	//////////////////////////////////////////////////////////////////////////

	unsigned int i;
	unsigned int recOrder = (int)(reconstructionOrder/4.0f);

	Vector4 borderColor(0.0f, 0.0f, 0.0f, 0.0f);

	glGenTextures(1, &shadowMapTexture);
	glBindTexture(GL_TEXTURE_2D, shadowMapTexture);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32, mapSize, mapSize, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER_ARB);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER_ARB);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	//glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor.comp);

	//////////////////////////////////////////////////////////////////////////

	if (useMipMaps)
	{
		glGenTextures(1, &linearShadowMapTexture);
		glBindTexture(GL_TEXTURE_2D, linearShadowMapTexture);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F_ARB, mapSize, mapSize, 0, GL_RGBA, GL_FLOAT, NULL);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER_ARB);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER_ARB);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glGenerateMipmapEXT(GL_TEXTURE_2D);
	}
	else
	{
		glGenTextures(1, &linearShadowMapTexture);
		glBindTexture(GL_TEXTURE_2D, linearShadowMapTexture);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F_ARB, mapSize, mapSize, 0, GL_RGBA, GL_FLOAT, NULL);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER_ARB);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER_ARB);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	}

	glBindTexture(GL_TEXTURE_2D, NULL);

	//////////////////////////////////////////////////////////////////////////

	if (useMipMaps)
	{
		glGenTextures(1, &csmSinTextureArray);
		glBindTexture(GL_TEXTURE_2D_ARRAY_EXT, csmSinTextureArray);
		glTexImage3D(GL_TEXTURE_2D_ARRAY_EXT, 0, GL_RGBA32F_ARB, mapSize, mapSize, recOrder, 0, GL_RGBA, GL_FLOAT, NULL);
		glTexParameteri(GL_TEXTURE_2D_ARRAY_EXT, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D_ARRAY_EXT, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D_ARRAY_EXT, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
		glTexParameteri(GL_TEXTURE_2D_ARRAY_EXT, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glGenerateMipmapEXT(GL_TEXTURE_2D_ARRAY_EXT);

		glGenTextures(1, &csmCosTextureArray);
		glBindTexture(GL_TEXTURE_2D_ARRAY_EXT, csmCosTextureArray);
		glTexImage3D(GL_TEXTURE_2D_ARRAY_EXT, 0, GL_RGBA32F_ARB, mapSize, mapSize, recOrder, 0, GL_RGBA, GL_FLOAT, NULL);
		glTexParameteri(GL_TEXTURE_2D_ARRAY_EXT, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D_ARRAY_EXT, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D_ARRAY_EXT, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
		glTexParameteri(GL_TEXTURE_2D_ARRAY_EXT, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glGenerateMipmapEXT(GL_TEXTURE_2D_ARRAY_EXT);
	}
	else
	{
		glGenTextures(1, &csmSinTextureArray);
		glBindTexture(GL_TEXTURE_2D_ARRAY_EXT, csmSinTextureArray);
		glTexImage3D(GL_TEXTURE_2D_ARRAY_EXT, 0, GL_RGBA32F_ARB, mapSize, mapSize, recOrder, 0, GL_RGBA, GL_FLOAT, NULL);
		glTexParameteri(GL_TEXTURE_2D_ARRAY_EXT, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D_ARRAY_EXT, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

		glGenTextures(1, &csmCosTextureArray);
		glBindTexture(GL_TEXTURE_2D_ARRAY_EXT, csmCosTextureArray);
		glTexImage3D(GL_TEXTURE_2D_ARRAY_EXT, 0, GL_RGBA32F_ARB, mapSize, mapSize, recOrder, 0, GL_RGBA, GL_FLOAT, NULL);
		glTexParameteri(GL_TEXTURE_2D_ARRAY_EXT, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D_ARRAY_EXT, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	}

	glBindTexture( GL_TEXTURE_2D_ARRAY_EXT, NULL);

	//////////////////////////////////////////////////////////////////////////

	glGenFramebuffersEXT(1, &frameBuffer);
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, frameBuffer);
	glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, linearShadowMapTexture, 0);
	glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, GL_TEXTURE_2D, shadowMapTexture, 0);
	CheckFrameBufferState();

	//////////////////////////////////////////////////////////////////////////

	glGenFramebuffersEXT(1, &csmSinBuffer);
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, csmSinBuffer);
	for (i=0; i<recOrder; i++)
		glFramebufferTextureLayerEXT(GL_FRAMEBUFFER_EXT, colorAttachments[i], csmSinTextureArray, 0, i);
	CheckFrameBufferState();

	glGenFramebuffersEXT(1, &csmCosBuffer);
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, csmCosBuffer);
	for (i=0; i<recOrder; i++)
		glFramebufferTextureLayerEXT(GL_FRAMEBUFFER_EXT, colorAttachments[i], csmCosTextureArray, 0, i);
	CheckFrameBufferState();

	//////////////////////////////////////////////////////////////////////////

	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
}

// -----------------------------------------------------------------------------
// --------------------------- ShadowMap::~ShadowMap ---------------------------
// -----------------------------------------------------------------------------
ShadowMap::~ShadowMap(void)
{
	glDeleteFramebuffersEXT(1, &frameBuffer);
	glDeleteFramebuffersEXT(1, &csmSinBuffer);
	glDeleteFramebuffersEXT(1, &csmCosBuffer);

	glDeleteTextures(1, &shadowMapTexture);
	glDeleteTextures(1, &linearShadowMapTexture);

	glDeleteTextures(1, &csmSinTextureArray);
	glDeleteTextures(1, &csmCosTextureArray);
}

// -----------------------------------------------------------------------------
// ------------------------- ShadowMap::SetFrameBuffer -------------------------
// -----------------------------------------------------------------------------
void ShadowMap::SetFrameBuffer(void) const
{
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, frameBuffer);
}

// -----------------------------------------------------------------------------
// --------------------------- ShadowMap::ComputeCsm ---------------------------
// -----------------------------------------------------------------------------
void ShadowMap::ComputeCsm(void)
{
	unsigned int recOrder = (int)(reconstructionOrder/4.0f);

	ShaderManager::Instance()->EnableShader(ShaderManager::SHADER_EFFECT_SHADOW_MAP, "ComputeConvolutionMap");
	ShaderManager::Instance()->SetParameter1f(ShaderManager::SP_SME_SIN_COS_FLAG, 0.0f);
	ShaderManager::Instance()->SetParameter1f(ShaderManager::SP_SME_CK_INDEX, 0.0f);

	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, csmSinBuffer);

	GLenum sBuffers[] = {GL_COLOR_ATTACHMENT0_EXT, GL_COLOR_ATTACHMENT1_EXT, GL_COLOR_ATTACHMENT2_EXT, GL_COLOR_ATTACHMENT3_EXT};
	glDrawBuffers(recOrder, sBuffers);

	DrawQuad();

	//////////////////////////////////////////////////////////////////////////

	ShaderManager::Instance()->SetParameter1f(ShaderManager::SP_SME_SIN_COS_FLAG, 1.0f);
	ShaderManager::Instance()->SetParameter1f(ShaderManager::SP_SME_CK_INDEX, 0.0f);

	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, csmCosBuffer);

	GLenum cBuffers[] = {GL_COLOR_ATTACHMENT0_EXT, GL_COLOR_ATTACHMENT1_EXT, GL_COLOR_ATTACHMENT2_EXT, GL_COLOR_ATTACHMENT3_EXT};
	glDrawBuffers(recOrder, cBuffers);

	DrawQuad();
}

// -----------------------------------------------------------------------------
// ---------------------------- ShadowMap::PreFilter ---------------------------
// -----------------------------------------------------------------------------
void ShadowMap::PreFilter(void)
{
	if (!useMipMaps)
	{
		FilterManager::Instance()->GenerateSummedAreaTable(linearShadowMapTexture);

		FilterManager::Instance()->GenerateSummedAreaTableArray(csmSinTextureArray);
		FilterManager::Instance()->GenerateSummedAreaTableArray(csmCosTextureArray);

		FilterManager::Instance()->SetStates();
	}

	//////////////////////////////////////////////////////////////////////////

	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
	ShaderManager::Instance()->DisableShader();

	//////////////////////////////////////////////////////////////////////////

	if (useMipMaps)
	{
		FilterManager::Instance()->GenerateMipMaps(linearShadowMapTexture);

		FilterManager::Instance()->GenerateMipMapsArray(csmSinTextureArray);
		FilterManager::Instance()->GenerateMipMapsArray(csmCosTextureArray);

		FilterManager::Instance()->SetStates();
	}
}

// -----------------------------------------------------------------------------
// ---------------------------- ShadowMap::DrawQuad ----------------------------
// -----------------------------------------------------------------------------
void ShadowMap::DrawQuad(void) const
{
	glBegin(GL_QUADS);
	{
		glTexCoord2f(0.0f, 1.0f);
		glVertex2f(0.0f, 0.0f);
		glTexCoord2f(0.0f, 0.0f);
		glVertex2f(0.0f, mapSize);
		glTexCoord2f(1.0f, 0.0f);
		glVertex2f(mapSize, mapSize);
		glTexCoord2f(1.0f, 1.0f);
		glVertex2f(mapSize, 0.0f);
	}
	glEnd();
}

// -----------------------------------------------------------------------------
// ---------------------- ShadowMap::CheckFrameBufferState ---------------------
// -----------------------------------------------------------------------------
void ShadowMap::CheckFrameBufferState(void) const
{
	GLenum status = glCheckFramebufferStatusEXT(GL_FRAMEBUFFER_EXT);
	switch( status )
	{
	case GL_FRAMEBUFFER_COMPLETE_EXT:
		break;
	case GL_FRAMEBUFFER_UNSUPPORTED_EXT:
		assert(false);
		break;
	default:
		assert(false);
	}
}
#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <windows.h>
#include "GL/glew.h"
#include <GL/gl.h>

#include <stdio.h>
#include <assert.h>

#include "ShadowMapping.h"

#include "../Level/Light.h"
#include "../Level/Camera.h"

#include "../Util/Matrix4.h"
#include "../Util/ConfigLoader.h"

#include "../Render/RenderManager.h"
#include "../Render/ShaderManager.h"

#include "../Main/DemoManager.h"

#include <GL/glut.h>

// -----------------------------------------------------------------------------
// ----------------------- ShadowMapping::ShadowMapping ------------------------
// -----------------------------------------------------------------------------
ShadowMapping::ShadowMapping(void) :
	shadowMapTexture(0),
	shadowMapDepthBuffer(0),
	width(0),
	height(0),
	light(NULL),
	camera(NULL)
{
}

// -----------------------------------------------------------------------------
// ----------------------- ShadowMapping::TextRenderManager --------------------
// -----------------------------------------------------------------------------
void ShadowMapping::Init(Light* light, Camera* camera, int width, int height)
{
	this->camera = camera;
	this->light = light;
	this->height = height;
	this->width = width;

	glGenFramebuffersEXT(1, &shadowMapDepthBuffer);
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, shadowMapDepthBuffer);

	GLuint depthbuffer;
	glGenRenderbuffersEXT(1, &depthbuffer);
	glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, depthbuffer);
	glRenderbufferStorageEXT(GL_RENDERBUFFER_EXT, GL_DEPTH_COMPONENT, width, height);
	glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, GL_RENDERBUFFER_EXT, depthbuffer);

	glGenTextures(1, &shadowMapTexture);
	glBindTexture(GL_TEXTURE_2D, shadowMapTexture);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, width, height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
	glGenerateMipmapEXT(GL_TEXTURE_2D);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_DEPTH_TEXTURE_MODE, GL_INTENSITY);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, GL_TEXTURE_2D, shadowMapTexture, 0);

	glReadBuffer(GL_NONE);
	glDrawBuffer(GL_NONE);

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

// -----------------------------------------------------------------------------
// ----------------------- ShadowMapping::TextRenderManager --------------------
// -----------------------------------------------------------------------------
void ShadowMapping::Begin(void)
{
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, shadowMapDepthBuffer);
	glPushAttrib(GL_VIEWPORT_BIT);
	glEnable(GL_POLYGON_OFFSET_FILL);
	glShadeModel(GL_FLAT);
	glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
	glCullFace(GL_FRONT);

	ShaderManager::Instance()->DisableShader();
	RenderManager::Instance()->SkipMaterials(true);

	// Set the viewport
	glViewport(0, 0, width, height);

	// Set ModelView Matrix
	glLoadIdentity();
	glMultMatrixf(light->GetViewMatrix().entry);

	// Set Projection Matrix
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glMultMatrixf(light->GetProjectionMatrix().entry);
	glMatrixMode(GL_MODELVIEW);

	// Clear buffers
	glClearColor (0.0, 0.0, 0.0, 1.0);
	glClear(GL_DEPTH_BUFFER_BIT|GL_COLOR_BUFFER_BIT);
}

// -----------------------------------------------------------------------------
// ----------------------- ShadowMapping::TextRenderManager --------------------
// -----------------------------------------------------------------------------
void ShadowMapping::End(void)
{
	RenderManager::Instance()->SkipMaterials(false);

	glCullFace(GL_BACK);
	glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
	glShadeModel(GL_SMOOTH);
	glDisable(GL_POLYGON_OFFSET_FILL);
	glPopAttrib();
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);

	Matrix4 bias;
	bias.SetTranslation(0.5f, 0.5f, 0.5f);
	bias.SetScale(0.5f, 0.5f, 0.5f);
	
	Matrix4 textureMatrix =
		camera->GetCameraMatrix().Inverse() *
		light->GetViewMatrix() *
		light->GetProjectionMatrix() *
		bias;

	ShaderManager::Instance()->SetParameterTexture(ShaderManager::SP_SHADOW_MAP, shadowMapTexture);
	ShaderManager::Instance()->SetMatrixParameterfc(ShaderManager::SP_SHADOW_MAP_TEXTURE_MATRIX, textureMatrix.entry);
}

// -----------------------------------------------------------------------------
// ----------------------- ShadowMapping::RenderDebug --------------------------
// -----------------------------------------------------------------------------
void ShadowMapping::RenderDebug(void)
{
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0, 1024, 768, 0, -100, 100);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, shadowMapTexture);
	glDisable(GL_BLEND);

	glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_NONE);

	glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
	glBegin(GL_QUADS);
		glTexCoord2f(0,1);
		glVertex2f(100, 100);
		glTexCoord2f(0,0);
		glVertex2f(100, 600);
		glTexCoord2f(1,0);
		glVertex2f(600, 600);
		glTexCoord2f(1,1);
		glVertex2f(600, 100);
	glEnd();

	glEnable(GL_TEXTURE_2D);
	glEnable(GL_BLEND);
}
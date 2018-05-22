#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <windows.h>
#include "GL/glew.h"
#include <GL/gl.h>

#include <stdio.h>
#include <assert.h>

#include "FilterManager.h"

#include "../Util/Math.h"
#include "../Util/Matrix4.h"
#include "../Util/Vector2.h"
#include "../Util/Vector4.h"
#include "../Util/ConfigLoader.h"

#include "../Render/ShaderManager.h"
#include "../Main/DemoManager.h"

#include <GL/glut.h>

const char* shaderH[] = {
	"GenerateSummedAreaTableH2",
	"GenerateSummedAreaTableH4",
	"GenerateSummedAreaTableH8",
	"GenerateSummedAreaTableH16"
};

const char* shaderV[] = {
	"GenerateSummedAreaTableV2",
	"GenerateSummedAreaTableV4",
	"GenerateSummedAreaTableV8",
	"GenerateSummedAreaTableV16"
};

const char* shaderArrayH[] = {
	"GenerateSummedAreaTableArrayH2",
	"GenerateSummedAreaTableArrayH4",
	"GenerateSummedAreaTableArrayH8",
	"GenerateSummedAreaTableArrayH16"
};

const char* shaderArrayV[] = {
	"GenerateSummedAreaTableArrayV2",
	"GenerateSummedAreaTableArrayV4",
	"GenerateSummedAreaTableArrayV8",
	"GenerateSummedAreaTableArrayV16"
};


// -----------------------------------------------------------------------------
// ----------------------- FilterManager::FilterManager ------------------------
// -----------------------------------------------------------------------------
FilterManager::FilterManager(void) :
	isInizialized(false),
	satBuffer(0),
	satTexture(0),
	mapSize(512),
	useMipMaps(false),
	satSampleCount(2),
	satPassCount(0)
{
}

// -----------------------------------------------------------------------------
// --------------------------- FilterManager::Init -----------------------------
// -----------------------------------------------------------------------------
void FilterManager::Init(unsigned int _mapSize, float _reconstructionOrder, bool _useMipMaps)
{
	assert(!isInizialized);

	assert((_reconstructionOrder >= 4.0f) && (_reconstructionOrder <= 16.0f));
	assert(((int)_reconstructionOrder % 4) == 0);

	mapSize = _mapSize;
	useMipMaps = _useMipMaps;

	reconstructionOrder = (int)(_reconstructionOrder/4.0f);

	satPassCount = ceilf(2.0f*Math::Log(mapSize)/Math::Log(Math::Pow(2.0f, (float)(satSampleCount+1))));

	if (!useMipMaps)
	{
		glGenTextures(1, &satTexture);
		glBindTexture(GL_TEXTURE_2D, satTexture);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F_ARB, mapSize, mapSize, 0, GL_RGBA, GL_FLOAT, NULL);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER_ARB);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER_ARB);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	}

	glBindTexture(GL_TEXTURE_2D, NULL);

	glGenFramebuffersEXT(1, &satBuffer);
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);

	isInizialized = true;
}

// -----------------------------------------------------------------------------
// ------------------------ FilterManager::Exit --------------------------------
// -----------------------------------------------------------------------------
void FilterManager::Exit(void)
{
	assert(isInizialized);

	glDeleteFramebuffersEXT(1, &satBuffer);

	if (!useMipMaps)
		glDeleteTextures(1, &satTexture);

	isInizialized = false;
}

// -----------------------------------------------------------------------------
// ------------------- FilterManager::GenerateMipMaps --------------------------
// -----------------------------------------------------------------------------
void FilterManager::GenerateMipMaps(unsigned int texture)
{
	glBindTexture(GL_TEXTURE_2D, texture);
	glGenerateMipmapEXT(GL_TEXTURE_2D);
}

// -----------------------------------------------------------------------------
// ----------------- FilterManager::GenerateMipMapsArray -----------------------
// -----------------------------------------------------------------------------
void FilterManager::GenerateMipMapsArray(unsigned int textureArray)
{
	glBindTexture(GL_TEXTURE_2D_ARRAY_EXT, textureArray);
	glGenerateMipmapEXT(GL_TEXTURE_2D_ARRAY_EXT);
}

// -----------------------------------------------------------------------------
// ---------------- FilterManager::GenerateSummedAreaTable ---------------------
// -----------------------------------------------------------------------------
void FilterManager::GenerateSummedAreaTable(unsigned int texture)
{
	unsigned int satTextures[2];
	satTextures[0] = texture;
	satTextures[1] = satTexture;

	glBindTexture(GL_TEXTURE_2D, satTextures[0]);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER_ARB);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER_ARB);
	glBindTexture(GL_TEXTURE_2D, NULL);

	unsigned int readTexture = 0;
	unsigned int writeTexture = 1;

	unsigned int i;
	for (i=0; i<satPassCount; i++)
	{
		ShaderManager::Instance()->SetParameter1f(ShaderManager::SP_SAT_PASS_INDEX, i);
		ShaderManager::Instance()->SetParameterTexture(ShaderManager::SP_SAT_MAP, satTextures[readTexture]);
		ShaderManager::Instance()->EnableShader(ShaderManager::SHADER_EFFECT_SAT, shaderH[satSampleCount]);

		glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, satBuffer);
		glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, satTextures[writeTexture], 0);
		glDrawBuffer(GL_COLOR_ATTACHMENT0_EXT);

		DrawQuad();

		// swap textures
		int tmp = readTexture;
		readTexture = writeTexture;
		writeTexture = tmp;
	}

	for (i=0; i<satPassCount; i++)
	{
		ShaderManager::Instance()->SetParameter1f(ShaderManager::SP_SAT_PASS_INDEX, i);
		ShaderManager::Instance()->SetParameterTexture(ShaderManager::SP_SAT_MAP, satTextures[readTexture]);
		ShaderManager::Instance()->EnableShader(ShaderManager::SHADER_EFFECT_SAT, shaderV[satSampleCount]);

		glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, satBuffer);
		glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, satTextures[writeTexture], 0);
		glDrawBuffer(GL_COLOR_ATTACHMENT0_EXT);

		DrawQuad();

		// swap textures
		int tmp = readTexture;
		readTexture = writeTexture;
		writeTexture = tmp;
	}

	glBindTexture(GL_TEXTURE_2D, satTextures[0]);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE_EXT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE_EXT);
	glBindTexture(GL_TEXTURE_2D, NULL);
}

// -----------------------------------------------------------------------------
// -------------- FilterManager::GenerateSummedAreaTableArray ------------------
// -----------------------------------------------------------------------------
void FilterManager::GenerateSummedAreaTableArray(unsigned int textureArray)
{
	glBindTexture(GL_TEXTURE_2D_ARRAY_EXT, textureArray);
	glTexParameteri(GL_TEXTURE_2D_ARRAY_EXT, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER_ARB);
	glTexParameteri(GL_TEXTURE_2D_ARRAY_EXT, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER_ARB);
	glBindTexture(GL_TEXTURE_2D_ARRAY_EXT, NULL);

	ShaderManager::Instance()->SetParameterTexture(ShaderManager::SP_SAT_ARRAY_MAP, textureArray);
	ShaderManager::Instance()->SetParameterTexture(ShaderManager::SP_SAT_MAP, satTexture);

	unsigned int textureIndex;
	unsigned int i;
	for (textureIndex=0; textureIndex<reconstructionOrder; textureIndex++)
	{
		ShaderManager::Instance()->SetParameter1f(ShaderManager::SP_SAT_INDEX, (float)textureIndex);

		for (i=0; i<satPassCount; i++)
		{
			if (i % 2 == 0)
			{
				ShaderManager::Instance()->SetParameter1f(ShaderManager::SP_SAT_PASS_INDEX, i);
				ShaderManager::Instance()->EnableShader(ShaderManager::SHADER_EFFECT_SAT, shaderArrayH[satSampleCount]);

				glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, satBuffer);
				glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, satTexture, 0);
				glDrawBuffer(GL_COLOR_ATTACHMENT0_EXT);

				DrawQuad();
			}
			else
			{
				ShaderManager::Instance()->SetParameter1f(ShaderManager::SP_SAT_PASS_INDEX, i);
				ShaderManager::Instance()->EnableShader(ShaderManager::SHADER_EFFECT_SAT, shaderH[satSampleCount]);

				glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, satBuffer);
				glFramebufferTextureLayerEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, textureArray, 0, textureIndex);
				glDrawBuffer(GL_COLOR_ATTACHMENT0_EXT);

				DrawQuad();
			}
		}

		for (i=0; i<satPassCount; i++)
		{
			if (i % 2 == 0)
			{
				ShaderManager::Instance()->SetParameter1f(ShaderManager::SP_SAT_PASS_INDEX, i);
				ShaderManager::Instance()->EnableShader(ShaderManager::SHADER_EFFECT_SAT, shaderArrayV[satSampleCount]);

				glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, satBuffer);
				glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, satTexture, 0);
				glDrawBuffer(GL_COLOR_ATTACHMENT0_EXT);

				DrawQuad();
			}
			else
			{
				ShaderManager::Instance()->SetParameter1f(ShaderManager::SP_SAT_PASS_INDEX, i);
				ShaderManager::Instance()->EnableShader(ShaderManager::SHADER_EFFECT_SAT, shaderV[satSampleCount]);

				glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, satBuffer);
				glFramebufferTextureLayerEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, textureArray, 0, textureIndex);
				glDrawBuffer(GL_COLOR_ATTACHMENT0_EXT);

				DrawQuad();
			}
		}
	}

	glBindTexture(GL_TEXTURE_2D_ARRAY_EXT, textureArray);
	glTexParameteri(GL_TEXTURE_2D_ARRAY_EXT, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE_EXT);
	glTexParameteri(GL_TEXTURE_2D_ARRAY_EXT, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE_EXT);
	glBindTexture(GL_TEXTURE_2D_ARRAY_EXT, NULL);
}

// -----------------------------------------------------------------------------
// ----------------------- FilterManager::SetStates ----------------------------
// -----------------------------------------------------------------------------
void FilterManager::SetStates(void) const
{
	glBindTexture(GL_TEXTURE_2D_ARRAY_EXT, NULL);
	glBindTexture(GL_TEXTURE_2D, NULL);
}

// -----------------------------------------------------------------------------
// ------------------- FilterManager::SetSatSampleCount ------------------------
// -----------------------------------------------------------------------------
void FilterManager::SetSatSampleCount(unsigned int _count)
{
	assert(_count < 4);
	satSampleCount = _count;
	satPassCount = ceilf(2.0f*Math::Log(mapSize)/Math::Log(Math::Pow(2.0f, (float)(satSampleCount+1))));
}

// -----------------------------------------------------------------------------
// ----------------------- FilterManager::DrawQuad -----------------------------
// -----------------------------------------------------------------------------
void FilterManager::DrawQuad(void) const
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
// ----------------- FilterManager::CheckFrameBufferState ----------------------
// -----------------------------------------------------------------------------
void FilterManager::CheckFrameBufferState(void) const
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
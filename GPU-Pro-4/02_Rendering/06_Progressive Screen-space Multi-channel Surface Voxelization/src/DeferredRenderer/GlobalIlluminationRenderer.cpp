#include "GlobalIlluminationRenderer.h"
#include "DeferredRenderer.h"

GlobalIlluminationRenderer::GlobalIlluminationRenderer()
{
	param_string = NULL;
	range = 1.0f;
	samples = 16;
	bounces = 1;
	factor = 1.0f;
	renderer = NULL;
	shader = NULL;
	initialized = false;
	type = DR_GI_METHOD_NONE;
}

GlobalIlluminationRenderer::~GlobalIlluminationRenderer()
{
	gi_lights.clear();
	if (param_string)
		free (param_string);
}

bool GlobalIlluminationRenderer::init(DeferredRenderer * _renderer)
{
	renderer = _renderer;
	if (!renderer)
		return false;
	
	for (int i=0; i<renderer->getNumLights(); i++)
	{
		if (renderer->getLight(i)->isGIEnabled())
			addGILight(i);
	}

	initialized = true;
	return true;
}

void GlobalIlluminationRenderer::update()
{
}

void GlobalIlluminationRenderer::draw()
{
	if (!initialized)
		return;

	glColor3f(0,0,0);
	glFrontFace(GL_CCW);
	glBegin(GL_QUADS);
		glColor3f(1,1,1);
		glNormal3f(0,0,1);
		glTexCoord2f(0,1);
		glVertex3f(-1,1,0);
		glTexCoord2f(0,0);
		glVertex3f(-1,-1,0);
		glTexCoord2f(1,0);
		glVertex3f(1,-1,0);
		glTexCoord2f(1,1);
		glVertex3f(1,1,0);
	glEnd();	
}

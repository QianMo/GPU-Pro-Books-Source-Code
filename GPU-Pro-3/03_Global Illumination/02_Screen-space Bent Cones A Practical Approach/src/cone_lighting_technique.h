#pragma once

#include "gbuffer.h"
#include "single_texture2d_rt.h"
#include "program.h"

class ConeLightingTechnique {
public:
	ConeLightingTechnique();
	~ConeLightingTechnique();

	void resize(unsigned int width, unsigned int height);

	void render(const GBuffer& gbuffer, const SingleTexture2DRT& BNAO, GLuint cubeMapArray, int arrayLayerCount);

	SingleTexture2DRT& output() { return output_; };

private:
	Program program_;
	SingleTexture2DRT output_;
};
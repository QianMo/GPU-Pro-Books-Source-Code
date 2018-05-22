#pragma once

#include "Texture2D.h"
#include "program.h"
#include "single_texture2d_rt.h"
#include <glm/glm.hpp>

class GBuffer;

class BentNormalBlurring {
public:
	BentNormalBlurring();
	~BentNormalBlurring();

	struct InputParameters {
		int kernelSize;
		float normalPower;
		float positionPower;
		int subSampling;
	};

	void resize(unsigned int width, unsigned int height);

	void setStaticParameters(const InputParameters& params);

	void render(const SingleTexture2DRT& input, const GBuffer* gbuffer);

	SingleTexture2DRT& output() { return output_; };

private:
	Texture2D seedTexture_;
	Program program_;

	SingleTexture2DRT pingPong_;
	SingleTexture2DRT output_;
};
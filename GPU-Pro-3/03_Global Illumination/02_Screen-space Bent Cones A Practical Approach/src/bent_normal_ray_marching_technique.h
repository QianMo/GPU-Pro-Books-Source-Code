#pragma once

#include "Texture2D.h"
#include "program.h"
#include "single_texture2d_rt.h"
#include <glm/glm.hpp>

class GBuffer;

class BentNormalRayMarchingTechnique {
public:
	BentNormalRayMarchingTechnique();
	~BentNormalRayMarchingTechnique();

	struct InputParameters {
		float sampleRadius;
		float maxDistance;
		unsigned int patternSize;
		unsigned int sampleCount;
		unsigned int numRayMarchingSteps;
		float rayMarchingBias;
	};

	void resize(unsigned int width, unsigned int height);

	void setStaticParameters(const InputParameters& params);

	void render(const glm::mat4& viewMatrix, const glm::mat4& viewProjectionMatrix, const GBuffer* gbuffer);

	SingleTexture2DRT& output() { return output_; };

private:
	Texture2D seedTexture_;
	Program program_;

	SingleTexture2DRT output_;
};
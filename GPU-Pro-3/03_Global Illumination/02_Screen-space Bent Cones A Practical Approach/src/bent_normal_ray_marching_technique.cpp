#include "bent_normal_ray_marching_technique.h"
#include "quad.h"
#include "gbuffer.h"
#include "tools.h"
#include <cmath>

using namespace glm;

BentNormalRayMarchingTechnique::BentNormalRayMarchingTechnique() :
output_(GL_RGBA8)
{
	program_.loadFiles("shaders/quad.vert", "shaders/bent_normal_ray_marching.frag");
	output_.init(1,1);
}

BentNormalRayMarchingTechnique::~BentNormalRayMarchingTechnique() {
}

void BentNormalRayMarchingTechnique::render(const mat4& viewMatrix, const mat4& viewProjectionMatrix, const GBuffer* gbuffer) {
	output_.bindDrawFBO();
	program_.use();
	glUniformMatrix4fv(glGetUniformLocation(program_.id(), "viewMatrix"), 1, GL_FALSE, &viewMatrix[0][0]);
	glUniformMatrix4fv(glGetUniformLocation(program_.id(), "viewProjectionMatrix"), 1, GL_FALSE, &viewProjectionMatrix[0][0]);

	gbuffer->bindTextures();

	seedTexture_.bind(2);

	glViewport(0, 0, output_.width(), output_.height());

	Quad::InstanceRef().render();
	program_.unUse();
	output_.unbindDrawFBO();
}

void BentNormalRayMarchingTechnique::setStaticParameters(const InputParameters& params) {
	program_.use();
	GLCHECK(glUniform1f(glGetUniformLocation(program_.id(), "sampleRadius"), params.sampleRadius));
	GLCHECK(glUniform1f(glGetUniformLocation(program_.id(), "maxDistance"), params.maxDistance));
	GLCHECK(glUniform1i(glGetUniformLocation(program_.id(), "patternSize"), params.patternSize));
	GLCHECK(glUniform1i(glGetUniformLocation(program_.id(), "sampleCount"), params.sampleCount));
	GLCHECK(glUniform1i(glGetUniformLocation(program_.id(), "numRayMarchingSteps"), params.numRayMarchingSteps));

	GLCHECK(glUniform1i(glGetUniformLocation(program_.id(), "positionTexture"), 0));
	GLCHECK(glUniform1i(glGetUniformLocation(program_.id(), "normalTexture"), 1));
	GLCHECK(glUniform1i(glGetUniformLocation(program_.id(), "seedTexture"), 2));
	program_.unUse();

	int patternSizeSquared = params.patternSize * params.patternSize;

	float* const seedPixels = new float[4 * params.sampleCount * patternSizeSquared];

	Tools::srandTimeSeed();

	for(int i = 0; i < patternSizeSquared; i++) {
		for(int j = 0; j < params.sampleCount; j++) {
			const int index = i * params.sampleCount + j;
			float xi0, xi1, xi2;
			xi0 = Tools::frand();
			while((xi1 = Tools::frand()) == xi0 || (xi1 == 0.0));
			while((xi2 = Tools::frand()) == xi1);

			vec3 direction;
			const vec2 hemisphericalDirection(2.0f * float(M_PI) * xi0, acosf(xi1));
			Tools::unitSphericalToCarthesian(hemisphericalDirection, direction);

			const float offset = xi2 / float(params.numRayMarchingSteps);

			seedPixels[index * 4 + 0] = direction.x;
			seedPixels[index * 4 + 1] = direction.y;
			seedPixels[index * 4 + 2] = direction.z;
			seedPixels[index * 4 + 3] = offset + params.rayMarchingBias;
		}
	}

	seedTexture_.bind();
	GLCHECK(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, params.sampleCount, patternSizeSquared, 0, internalFormatToFormat(GL_RGBA16F), internalFormatToType(GL_RGBA16F), seedPixels));
	GLCHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
	GLCHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
	GLCHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT));
	GLCHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT));
	seedTexture_.unbind();
	delete[] seedPixels;
}

void BentNormalRayMarchingTechnique::resize(unsigned int width, unsigned int height) {
	output_.resize(width, height);
}

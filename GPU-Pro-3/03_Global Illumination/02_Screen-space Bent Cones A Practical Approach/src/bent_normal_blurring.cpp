#include "bent_normal_blurring.h"
#include "quad.h"
#include "gbuffer.h"
#include "tools.h"
#include <cmath>

using namespace glm;

BentNormalBlurring::BentNormalBlurring() :
output_(GL_RGBA8),
pingPong_(GL_RGBA8)
{
	program_.loadFiles("shaders/quad_texcoord.vert", "shaders/bent_normal_blurring.frag");
	pingPong_.init(1,1);
	output_.init(1,1);
}

BentNormalBlurring::~BentNormalBlurring() {
}

void BentNormalBlurring::render(const SingleTexture2DRT& input, const GBuffer* gbuffer) {
	program_.use();
	pingPong_.bindDrawFBO();
	glUniform2i(glGetUniformLocation(program_.id(), "maskDirection"), 1, 0);

	glViewport(0, 0, output_.width(), output_.height());

	gbuffer->bindTextures();
	input.bindTexture(2);

	Quad::InstanceRef().render();

	//////////////////////////////////////////////////////////////////////////
	// other direction
	output_.bindDrawFBO();
	pingPong_.bindTexture(2);

	glUniform2i(glGetUniformLocation(program_.id(), "maskDirection"), 0, 1);
	Quad::InstanceRef().render();

	program_.unUse();
	output_.unbindDrawFBO();
}

void BentNormalBlurring::setStaticParameters(const InputParameters& params) {
	program_.use();
	GLCHECK(glUniform1f(glGetUniformLocation(program_.id(), "positionPower"), params.positionPower));
	GLCHECK(glUniform1f(glGetUniformLocation(program_.id(), "normalPower"), params.normalPower));
	GLCHECK(glUniform1i(glGetUniformLocation(program_.id(), "kernelSize"), params.kernelSize));
	GLCHECK(glUniform1i(glGetUniformLocation(program_.id(), "subSampling"), params.subSampling));

	GLCHECK(glUniform1i(glGetUniformLocation(program_.id(), "positionTexture"), 0));
	GLCHECK(glUniform1i(glGetUniformLocation(program_.id(), "normalTexture"), 1));
	GLCHECK(glUniform1i(glGetUniformLocation(program_.id(), "inputTexture"), 2));
	program_.unUse();
}

void BentNormalBlurring::resize(unsigned int width, unsigned int height) {
	pingPong_.resize(width, height);
	output_.resize(width, height);
}

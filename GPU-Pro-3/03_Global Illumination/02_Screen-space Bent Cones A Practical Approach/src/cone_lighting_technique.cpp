#include "cone_lighting_technique.h"
#include "quad.h"

ConeLightingTechnique::ConeLightingTechnique() :
output_(GL_RGBA16F)
{
	output_.init(1,1);

	program_.loadFiles("shaders/quad_texcoord.vert", "shaders/cone_lighting.frag");

	program_.use();
	GLCHECK(glUniform1i(glGetUniformLocation(program_.id(), "bnAOTexture"), 0));
	GLCHECK(glUniform1i(glGetUniformLocation(program_.id(), "normalTexture"), 1));
	GLCHECK(glUniform1i(glGetUniformLocation(program_.id(), "diffuseTexture"), 2));
	GLCHECK(glUniform1i(glGetUniformLocation(program_.id(), "convolvedEnvMapArray"), 3));
	program_.unUse();
}

ConeLightingTechnique::~ConeLightingTechnique() {
}

void ConeLightingTechnique::resize(unsigned int width, unsigned int height) {
	output_.resize(width, height);
}

void ConeLightingTechnique::render(const GBuffer& gbuffer, const SingleTexture2DRT& BNAO, GLuint cubeMapArray, int arrayLayerCount) {
	output_.bindDrawFBO();
	program_.use();
	GLCHECK(glUniform1f(glGetUniformLocation(program_.id(), "cubeMapArrayLayerCount"), float(arrayLayerCount)));
	
	gbuffer.bindTextures(0);
	BNAO.bindTexture(0); // overwrite binding on texture unit 0
	glActiveTexture(GL_TEXTURE3);
	glBindTexture(GL_TEXTURE_CUBE_MAP_ARRAY, cubeMapArray);

	Quad::InstanceRef().renderWithPreAndPost();

	glBindTexture(GL_TEXTURE_CUBE_MAP_ARRAY, 0);
	program_.unUse();
	output_.unbindDrawFBO();
}

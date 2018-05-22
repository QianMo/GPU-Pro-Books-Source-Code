#include "sky_renderer.h"
#include "quad.h"

SkyRenderer::SkyRenderer() {
	program_.loadFiles("shaders/sky.vert", "shaders/sky.frag");
	program_.use();
	GLCHECK(glUniform1i(glGetUniformLocation(program_.id(), "envMap"), 0));
	program_.unUse();
}

void SkyRenderer::render(const glm::mat4& invViewProjection, GLuint envMap, unsigned int width, unsigned int height) {
	glViewport(0, 0, width, height);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_CUBE_MAP, envMap);

	program_.use();
	glUniformMatrix4fv(glGetUniformLocation(program_.id(), "invViewProjection"), 1, GL_FALSE, &invViewProjection[0][0]);

	Quad::InstanceRef().renderWithPreAndPost();

	glBindTexture(GL_TEXTURE_CUBE_MAP, 0);
	program_.unUse();
	
}


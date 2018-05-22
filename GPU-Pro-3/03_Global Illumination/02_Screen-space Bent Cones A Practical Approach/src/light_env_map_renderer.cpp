#include "light_env_map_renderer.h"
#include "tools.h"
#include "quad.h"

LightEnvMapRenderer::LightEnvMapRenderer() {
	glGenTextures(1, &cubeMapOutput_);

	resize(1);
}

LightEnvMapRenderer::~LightEnvMapRenderer() {
	glDeleteTextures(1, &cubeMapOutput_);
}

void LightEnvMapRenderer::resize(unsigned int size) {
	cubeSize_ = size;
	glBindTexture(GL_TEXTURE_CUBE_MAP, cubeMapOutput_);

	for(int i=0; i<6; ++i) {
		glTexImage2D(
			GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 
			0,
			GL_RGB16F, 
			size, 
			size, 
			0, 
			internalFormatToFormat(GL_RGB16F), 
			internalFormatToType(GL_RGB16F), 
			NULL);
	}
	GLCHECK(glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
	GLCHECK(glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
	GLCHECK(glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_REPEAT));
	GLCHECK(glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_REPEAT));
	glBindTexture(GL_TEXTURE_CUBE_MAP, 0);

	fbo_.bindDraw();
	for(int i=0; i<6; ++i) {
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + i, GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, cubeMapOutput_, 0);
	}

	Tools::sequentialDrawBuffers(6);

	if(!checkFramebufferStatus(__LINE__, __FILE__, __FUNCTION__)) return;

	FrameBuffer::unbindDraw();
}

void LightEnvMapRenderer::render(const InputParameters& params) {
	std::vector<std::string> defines;
	defines.push_back(std::string("NUMLIGHTS ") + Tools::intToString(params.lights.size()));
	program_.loadFiles("shaders/quad_texcoord.vert", "shaders/env_map_renderer.frag", defines);
	program_.use();

	for(size_t i=0; i<params.lights.size(); ++i) {
		GLCHECK(glUniform3fv(glGetUniformLocation(program_.id(), ("lights[" + Tools::intToString(i) + "].color").c_str()), 1, &params.lights[i].color[0]));
		GLCHECK(glUniform3fv(glGetUniformLocation(program_.id(), ("lights[" + Tools::intToString(i) + "].direction").c_str()), 1, &params.lights[i].direction[0]));
		GLCHECK(glUniform1f(glGetUniformLocation(program_.id(), ("lights[" + Tools::intToString(i) + "].innerAngle").c_str()), params.lights[i].innerAngle));
		GLCHECK(glUniform1f(glGetUniformLocation(program_.id(), ("lights[" + Tools::intToString(i) + "].outerAngle").c_str()), params.lights[i].outerAngle));
	}
	
	fbo_.bindDraw();

	glViewport(0,0,cubeSize_,cubeSize_);
	Quad::InstanceRef().renderWithPreAndPost();

	FrameBuffer::unbindDraw();
	program_.unUse();

	//glBindTexture(GL_TEXTURE_CUBE_MAP, cubeMapOutput_);
	//glGenerateMipmap(GL_TEXTURE_CUBE_MAP);
	//glBindTexture(GL_TEXTURE_CUBE_MAP, 0);
}



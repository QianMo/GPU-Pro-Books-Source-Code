#include "env_map_convolution.h"
#include "framebuffer.h"
#include "tools.h"
#include "quad.h"
#include <cmath>

namespace {
	const unsigned int maxNumberOfReadsPerPass = 1920 * 1080 * 16; // Full HD * 16?
}

EnvMapConvolution::EnvMapConvolution() {
	glGenTextures(1, &cubeMapArrayOutput_);
	resize(1,1);
	program_.loadFiles("shaders/quad_texcoord.vert", "shaders/env_map_convolution.frag");

	program_.use();
	GLCHECK(glUniform1i(glGetUniformLocation(program_.id(), "inputCube"), 0));
	program_.unUse();
}

EnvMapConvolution::~EnvMapConvolution() {
	resizeFrameBuffers(0);
}

void EnvMapConvolution::render(const InputParameters& params, GLuint envMapCube) {
	program_.use();
	glViewport(0,0,cubeSize_,cubeSize_);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_CUBE_MAP, envMapCube);

	GLCHECK(glUniform1i(glGetUniformLocation(program_.id(), "sampleCount"), params.sampleCount));

	Quad::InstanceRef().preRender();

	for(size_t i=0; i<frameBuffers_.size(); ++i) {
		frameBuffers_[i]->bindDraw();
		float angle = (float)(frameBuffers_.size() - i) / (float)frameBuffers_.size() * float(M_PI_2);
		GLCHECK(glUniform1f(glGetUniformLocation(program_.id(), "cutOffAngle"), angle));

		unsigned int necessaryReads = cubeSize_ * cubeSize_ * 6 * params.sampleCount;

		if(maxNumberOfReadsPerPass > necessaryReads) { // 6 faces, sampleCount (= reads per pixel)
			Quad::InstanceRef().render();
		}
		else {
			unsigned int blockSize = unsigned int(float(cubeSize_) / std::sqrt(float(necessaryReads / maxNumberOfReadsPerPass + (necessaryReads % maxNumberOfReadsPerPass > 0u ? 1u : 0u))));
			blockSize = std::max(1u, blockSize); // just to make sure...
			//std::cout << "Blocksize: " << blockSize << std::endl;
			unsigned int steps = cubeSize_ / blockSize + (cubeSize_ % blockSize > 0 ? 1 : 0);
			glEnable(GL_SCISSOR_TEST);
			for(unsigned int x=0; x<steps; ++x) {
				for(unsigned int y=0; y<steps; ++y) {
					glScissor(GLint(x*blockSize), GLint(y*blockSize), GLint(blockSize), GLint(blockSize));
					Quad::InstanceRef().render();
					//std::cout << "Scissor: " << x << ", " << y << std::endl;
				}
			}
			glDisable(GL_SCISSOR_TEST);
		}
	}
	Quad::InstanceRef().postRender();
	FrameBuffer::unbindDraw();
	glBindTexture(GL_TEXTURE_CUBE_MAP, 0);
}

void EnvMapConvolution::resize(unsigned int size, unsigned int layerCount) {
	cubeSize_ = size;
	layerCount_ = layerCount;
	glBindTexture(GL_TEXTURE_CUBE_MAP_ARRAY, cubeMapArrayOutput_);

	GLint internalFormat = GL_RGB16F;
	glTexImage3D(GL_TEXTURE_CUBE_MAP_ARRAY, 0, internalFormat, size, size, layerCount * 6, 0, internalFormatToFormat(internalFormat), internalFormatToType(internalFormat), 0);
	GLCHECK(glTexParameteri(GL_TEXTURE_CUBE_MAP_ARRAY, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
	GLCHECK(glTexParameteri(GL_TEXTURE_CUBE_MAP_ARRAY, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
	GLCHECK(glTexParameteri(GL_TEXTURE_CUBE_MAP_ARRAY, GL_TEXTURE_WRAP_S, GL_REPEAT));
	GLCHECK(glTexParameteri(GL_TEXTURE_CUBE_MAP_ARRAY, GL_TEXTURE_WRAP_T, GL_REPEAT));
	glBindTexture(GL_TEXTURE_CUBE_MAP_ARRAY, 0);

	resizeFrameBuffers(layerCount);
	for(size_t i=0; i<frameBuffers_.size(); ++i) {
		frameBuffers_[i]->bindDraw();
		for(int j=0; j<6; ++j) {
			glFramebufferTextureLayer(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0+(GLenum)j, cubeMapArrayOutput_, 0, (int)i*6+j);
		}
		Tools::sequentialDrawBuffers(6);
		FrameBuffer::checkStatus();
	}
	FrameBuffer::unbindDraw();
}

void EnvMapConvolution::resizeFrameBuffers(size_t newSize) {
	if(newSize > frameBuffers_.size()) {
		for(size_t i=frameBuffers_.size(); i<newSize; ++i) {
			frameBuffers_.push_back(new FrameBuffer);
		}
	}
	else if(newSize < frameBuffers_.size()) {
		for(size_t i=newSize; i<frameBuffers_.size(); ++i) {
			delete frameBuffers_[i];
		}
		frameBuffers_.resize(newSize);
	}
}
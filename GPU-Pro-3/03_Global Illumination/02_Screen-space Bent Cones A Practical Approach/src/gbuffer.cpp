#include "gbuffer.h"
#include "config.h"
#include "glhelper.h"

namespace {
    const std::string gbufferVertexShaderFile = "shaders/gbuffer.vert";
    const std::string gbufferFragmentShaderFile = "shaders/gbuffer.frag";

    const size_t numTextures = 3;
}

GBuffer::GBuffer()
: fbo_(0),
depthRT_(0), colorRTs_(numTextures, 0)
{
}


void GBuffer::resize(unsigned int width, unsigned int height) {
	width_ = width;
	height_ = height;

	// create textures_
	GLCHECK(glBindTexture(GL_TEXTURE_2D, depthRT_));
	GLint format = GL_DEPTH_COMPONENT32;
	GLCHECK(glTexImage2D(GL_TEXTURE_2D, 0, format, width_, height_, 0, internalFormatToFormat(format), internalFormatToType(format), 0));
	GLCHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
	GLCHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
	GLCHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
	GLCHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));

	// position
	GLCHECK(glBindTexture(GL_TEXTURE_2D, colorRTs_[0]));
	format = GL_RGBA16F;
	GLCHECK(glTexImage2D(GL_TEXTURE_2D, 0, format, width_, height_, 0, internalFormatToFormat(format), internalFormatToType(format), 0));
	// normal texture
	GLCHECK(glBindTexture(GL_TEXTURE_2D, colorRTs_[1]));
	format = GL_RGB16F;
	GLCHECK(glTexImage2D(GL_TEXTURE_2D, 0, format, width_, height_, 0, internalFormatToFormat(format), internalFormatToType(format), 0));
	// diffuse
	GLCHECK(glBindTexture(GL_TEXTURE_2D, colorRTs_[2]));
	format = GL_RGBA8;
	GLCHECK(glTexImage2D(GL_TEXTURE_2D, 0, format, width_, height_, 0, internalFormatToFormat(format), internalFormatToType(format), 0));

	for(size_t i=0; i<colorRTs_.size(); ++i) {
		GLCHECK(glBindTexture(GL_TEXTURE_2D, colorRTs_[i]));
		GLCHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
		GLCHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
		GLCHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
		GLCHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));
	}

	GLCHECK(glBindTexture(GL_TEXTURE_2D, 0));
}


void GBuffer::init(unsigned int width, unsigned int height) {
    cleanup();

	GLCHECK(glGenTextures(1, &depthRT_));
	GLCHECK(glGenTextures(numTextures, &colorRTs_[0]));

	resize(width, height);

    // create FBO
    GLenum drawBuffers[numTextures];
    GLCHECK(glGenFramebuffers(1, &fbo_));
    GLCHECK(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo_));
    for(unsigned int i=0; i<numTextures; ++i) {
        drawBuffers[i] = GL_COLOR_ATTACHMENT0 + GLenum(i);
		GLCHECK(glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + GLenum(i), GL_TEXTURE_2D, colorRTs_[i], 0));
	}
	GLCHECK(glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depthRT_, 0));
	
	GLCHECK(glDrawBuffers(numTextures, drawBuffers));

	checkFramebufferStatus(__LINE__, __FILE__, __FUNCTION__);

	GLCHECK(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0));
}

void GBuffer::reloadShader() {
    program_.loadFiles(gbufferVertexShaderFile, gbufferFragmentShaderFile);

    // get locations
    GLCHECK(uniformMVP_ = glGetUniformLocation(program_.id(), "MVP"));
    GLCHECK(uniformNormalM_ = glGetUniformLocation(program_.id(), "normalM"));
    GLCHECK(uMaterialDiffuse_ = glGetUniformLocation(program_.id(), "material.diffuse"));
    GLCHECK(uGamma_ = glGetUniformLocation(program_.id(), "gamma"));

    program_.use();
	GLCHECK(glUniform1i(glGetUniformLocation(program_.id(), "diffuseTexture"), 0));
	program_.unUse();
}

void GBuffer::cleanup() {
    if(fbo_) {
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glDeleteFramebuffers(1, &fbo_);
        fbo_ = 0;
    }
    if(depthRT_) {
        glBindTexture(GL_TEXTURE_2D, 0);
        glDeleteTextures(1, &depthRT_);
        depthRT_ = 0;
    }
    for(size_t i=0; i<colorRTs_.size(); ++i) {
        if(colorRTs_[i]) {
            glBindTexture(GL_TEXTURE_2D, 0);
            glDeleteTextures(1, &colorRTs_[i]);
            colorRTs_[i] = 0;
        }
    }
}

void GBuffer::preRender(const float* MVPMatrix, const float* normalMatrix, float gammaCorrection) {
	GLCHECK(glViewport(0,0,width_, height_));
	GLCHECK(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo_));
	GLCHECK(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT));

	program_.use();

    GLCHECK(glUniformMatrix3fv(uniformNormalM_, 1, GL_FALSE, normalMatrix));
    GLCHECK(glUniformMatrix4fv(uniformMVP_, 1, GL_FALSE, MVPMatrix));

    GLCHECK(glUniform1f(uGamma_, gammaCorrection));

    //GLCHECK(glPolygonMode(GL_FRONT_AND_BACK, GL_LINE));
}

void GBuffer::postRender() {
    program_.unUse();
    GLCHECK(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0));
    //GLCHECK(glPolygonMode(GL_FRONT_AND_BACK, GL_FILL));
}

void GBuffer::bindFBO() {
    GLCHECK(glBindFramebuffer(GL_FRAMEBUFFER, fbo_));
}

void GBuffer::unbindFBO() {
    GLCHECK(glBindFramebuffer(GL_FRAMEBUFFER, 0));
}

void GBuffer::bindReadFBO() {
    GLCHECK(glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo_));
}

void GBuffer::unbindReadFBO() {
    GLCHECK(glBindFramebuffer(GL_READ_FRAMEBUFFER, 0));
}

void GBuffer::clearFBO() {
    GLCHECK(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT));
}

unsigned int GBuffer::width() const {
    return width_;
}

unsigned int GBuffer::height() const {
    return height_;
}

void GBuffer::bindTextures(unsigned int offset /*= 0*/) const {
	for(size_t i=0; i<colorRTs_.size(); ++i) {
		GLCHECK(glActiveTexture(GL_TEXTURE0 + offset + i));
		GLCHECK(glBindTexture(GL_TEXTURE_2D, colorRTs_[i]));
	}
}

void GBuffer::unbindTextures(unsigned int offset /*= 0*/) const {
	for(size_t i=0; i<colorRTs_.size(); ++i) {
		GLCHECK(glActiveTexture(GL_TEXTURE0 + offset + i));
		GLCHECK(glBindTexture(GL_TEXTURE_2D, 0));
	}
}

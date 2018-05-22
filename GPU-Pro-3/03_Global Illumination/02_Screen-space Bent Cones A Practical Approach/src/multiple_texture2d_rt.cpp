#include "multiple_texture2d_rt.h"

#include "config.h"
#include "glhelper.h"

MultipleTexture2DRT::MultipleTexture2DRT(const std::vector<GLenum>& textureFormats)
: fbo_(0),
texFormats_(textureFormats),
textures_(textureFormats.size(), 0),
width_(0), height_(0)
{
}

MultipleTexture2DRT::MultipleTexture2DRT(GLenum textureFormat0)
: fbo_(0),
texFormats_(std::vector<GLenum>(1, textureFormat0)),
textures_(1, 0),
width_(0), height_(0)
{
}

MultipleTexture2DRT::MultipleTexture2DRT(GLenum textureFormat0, GLenum textureFormat1)
: fbo_(0),
texFormats_(std::vector<GLenum>(2)),
textures_(2, 0),
width_(0), height_(0)
{
	texFormats_[0] = textureFormat0;
	texFormats_[1] = textureFormat1;
}

MultipleTexture2DRT::MultipleTexture2DRT(GLenum textureFormat0, GLenum textureFormat1, GLenum textureFormat2)
: fbo_(0),
texFormats_(std::vector<GLenum>(3)),
textures_(3, 0),
width_(0), height_(0)
{
	texFormats_[0] = textureFormat0;
	texFormats_[1] = textureFormat1;
	texFormats_[2] = textureFormat2;
}

MultipleTexture2DRT::~MultipleTexture2DRT() {
	cleanup();
}

void MultipleTexture2DRT::resize(unsigned int width, unsigned int height) {
	width_ = width;
	height_ = height;

	for(size_t i=0; i<textures_.size(); ++i) {
		GLCHECK(glBindTexture(GL_TEXTURE_2D, textures_[i]));
		GLCHECK(glTexImage2D(GL_TEXTURE_2D, 0, texFormats_[i], width_, height_, 0, internalFormatToFormat(texFormats_[i]), internalFormatToType(texFormats_[i]), 0));
		GLCHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
		GLCHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
		GLCHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT));
		GLCHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT));
	}
	GLCHECK(glBindTexture(GL_TEXTURE_2D, 0));
}


void MultipleTexture2DRT::init(unsigned int width, unsigned int height) {
    cleanup();

	// create textures_
	GLCHECK(glGenTextures(textures_.size(), &textures_[0]));

	resize(width, height);

	// create FBO
	GLCHECK(glGenFramebuffers(1, &fbo_));
	GLCHECK(glBindFramebuffer(GL_FRAMEBUFFER, fbo_));
	if(textures_.size() == 1) {
		GLCHECK(glDrawBuffer(GL_COLOR_ATTACHMENT0));
		GLCHECK(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, textures_[0], 0));
	}
	else {
		std::vector<GLenum> drawBuffers(textures_.size());
		for(size_t i=0; i<drawBuffers.size(); ++i) {
			drawBuffers[i] = GL_COLOR_ATTACHMENT0 + i;
			GLCHECK(glFramebufferTexture2D(GL_FRAMEBUFFER, drawBuffers[i], GL_TEXTURE_2D, textures_[i], 0));
		}
		GLCHECK(glDrawBuffers(drawBuffers.size(), &drawBuffers[0]));
	}

    if(!checkFramebufferStatus(__LINE__, __FILE__, __FUNCTION__)) return;

    GLCHECK(glBindFramebuffer(GL_FRAMEBUFFER, 0));
}

void MultipleTexture2DRT::cleanup() {
    if(fbo_) {
        GLCHECK(glBindFramebuffer(GL_FRAMEBUFFER, 0));
        GLCHECK(glDeleteFramebuffers(1, &fbo_));
        fbo_ = 0;
    }
	if(!textures_.empty() && textures_[0] != 0) {
		GLCHECK(glBindTexture(GL_TEXTURE_2D, 0));
		GLCHECK(glDeleteTextures(textures_.size(), &textures_[0]));
		textures_.assign(textures_.size(), 0);
	}
}

void MultipleTexture2DRT::bindTexture(size_t index, unsigned int offset) const {
    GLCHECK(glActiveTexture(GL_TEXTURE0 + offset));
    GLCHECK(glBindTexture(GL_TEXTURE_2D, textures_[index]));
}

void MultipleTexture2DRT::unbindTexture(unsigned int offset) const {
    GLCHECK(glActiveTexture(GL_TEXTURE0 + offset));
    GLCHECK(glBindTexture(GL_TEXTURE_2D, 0));
}

void MultipleTexture2DRT::bindTextures(unsigned int offset) const {
	for(size_t i=0; i<textures_.size(); ++i) {
		GLCHECK(glActiveTexture(GL_TEXTURE0 + offset + i));
		GLCHECK(glBindTexture(GL_TEXTURE_2D, textures_[i]));
	}
}

void MultipleTexture2DRT::unbindTextures(unsigned int offset) const {
	for(size_t i=0; i<textures_.size(); ++i) {
		GLCHECK(glActiveTexture(GL_TEXTURE0 + offset + i));
		GLCHECK(glBindTexture(GL_TEXTURE_2D, 0));
	}
}

void MultipleTexture2DRT::clearFBO(float r /*= 0.0f*/, float g /*= 0.0f*/, float b /*= 0.0f*/, float a /*= 0.0f*/) {
    // does not work
    //float color[4] = { r, g, b, a };
    //GLCHECK(glClearBufferfv(GL_COLOR, GL_DRAW_BUFFER0, color));

    glClearColor(r, g, b, a);
    glClear(GL_COLOR_BUFFER_BIT);
}

void MultipleTexture2DRT::bindDrawFBO() {
    GLCHECK(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo_));
}

void MultipleTexture2DRT::bindReadFBO() {
    GLCHECK(glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo_));
}

void MultipleTexture2DRT::unbindDrawFBO() {
    GLCHECK(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0));
}

void MultipleTexture2DRT::unbindReadFBO() {
    GLCHECK(glBindFramebuffer(GL_READ_FRAMEBUFFER, 0));
}

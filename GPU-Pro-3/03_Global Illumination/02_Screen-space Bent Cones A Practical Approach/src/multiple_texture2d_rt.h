#pragma once

#include <gl/glew.h>
#include <vector>

class MultipleTexture2DRT {
public:
	MultipleTexture2DRT(const std::vector<GLenum>& textureFormats);
	MultipleTexture2DRT(GLenum textureFormat0);
	MultipleTexture2DRT(GLenum textureFormat0, GLenum textureFormat1);
	MultipleTexture2DRT(GLenum textureFormat0, GLenum textureFormat1, GLenum textureFormat2);

	~MultipleTexture2DRT();

	void init(unsigned int width, unsigned int height);
	void resize(unsigned int width, unsigned int height);
    void cleanup();

    inline GLuint fboId() const { return fbo_; };
    inline GLuint rtId(size_t index) const { return textures_[index]; };

    void bindDrawFBO();
    void unbindDrawFBO();
    void bindReadFBO();
    void unbindReadFBO();
    void clearFBO(float r = 0.0f, float g = 0.0f, float b = 0.0f, float a = 0.0f);

    void bindTexture(size_t index, unsigned int offset = 0) const;
    void unbindTexture(unsigned int offset = 0) const;
	void bindTextures(unsigned int offset = 0) const;
	void unbindTextures(unsigned int offset = 0) const;

    inline GLsizei width() const { return width_; };
    inline GLsizei height() const { return height_; };

protected:
    GLsizei width_;
    GLsizei height_;

private:
    GLuint fbo_;
    std::vector<GLuint> textures_;
    std::vector<GLenum> texFormats_;
};


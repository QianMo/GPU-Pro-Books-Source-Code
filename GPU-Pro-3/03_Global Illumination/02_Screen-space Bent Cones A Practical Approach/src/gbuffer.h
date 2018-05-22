#pragma once

#include <gl/glew.h>

#include <vector>

#include "program.h"

class GBuffer {
public:

    unsigned int width() const;
    unsigned int height() const;

    GBuffer();

    void init(unsigned int width, unsigned int height);
	void resize(unsigned int width, unsigned int height);
    void cleanup();
    void reloadShader();

    void preRender(const float* MVPMatrix, const float* normalMatrix, float gammaCorrection);
    inline void setMaterialParameter(const float* diffuseColor) {
        glUniform3f(uMaterialDiffuse_, diffuseColor[0], diffuseColor[1], diffuseColor[2]);
    }
    void postRender();

    void bindFBO();
    void unbindFBO();
    void bindReadFBO();
    void unbindReadFBO();
    void clearFBO();

	GLuint depthRT() { return depthRT_; };

    void bindTextures(unsigned int offset = 0) const;
    void unbindTextures(unsigned int offset = 0) const;

private:
    bool useMaterial_;

    GLuint fbo_;

    Program program_;
    GLint uniformTexture0_;
    GLint uniformMVP_;
    GLint uniformNormalM_;
    GLint uMaterialDiffuse_;
    GLint uGamma_;

    GLuint depthRT_;
    std::vector<GLuint> colorRTs_;

    unsigned int width_;
    unsigned int height_;
};
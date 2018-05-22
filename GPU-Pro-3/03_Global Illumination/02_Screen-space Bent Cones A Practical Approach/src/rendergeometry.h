#pragma once

#include <gl/glew.h>
#include "config.h"

class RenderGeometry {
public:
    RenderGeometry() : vao(0), vbo_data(0), vbo_elements(0), elementsType_(GL_UNSIGNED_SHORT), initialized_(false) {}

    virtual void init() = 0;
    virtual void cleanup() = 0;

    inline void renderWithPreAndPost() const {
        GLCHECK(glBindVertexArray(vao));
        GLCHECK(glDrawElements(GL_TRIANGLES, numElements_, elementsType_, 0));
        GLCHECK(glBindVertexArray(0));
    };

    inline void render() const {
        GLCHECK(glDrawElements(GL_TRIANGLES, numElements_, elementsType_, 0));
    };

    inline void preRender() const {
        GLCHECK(glBindVertexArray(vao));
    }

    inline void postRender() const {
        GLCHECK(glBindVertexArray(0));
    }

protected:
    GLuint vao;
    GLuint vbo_data;
    GLuint vbo_elements;
    GLenum elementsType_;

    GLsizei numElements_;

    bool initialized_;
};

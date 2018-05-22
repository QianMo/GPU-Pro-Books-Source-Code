#include "quad.h"
#include "config.h"

#include <gl/glew.h>

Quad* Quad::instance_ = NULL;

Quad::Quad()
: RenderGeometry()
{
    numElements_ = 6;
}

void Quad::init() {
    if(initialized_) return;

    const GLfloat quadVertices[numVertices_ * 2] =
    {
        -1.0f,-1.0f,    //0.0f, 0.0f, // tex coord is reconstructed
         1.0f,-1.0f,    //1.0f, 0.0f,
         1.0f, 1.0f,    //1.0f, 1.0f,
        -1.0f, 1.0f,    //0.0f, 1.0f
    };

    const GLushort quadElements[6] = /* numElements_ = 6; */
    {
        0, 1, 2, 
        2, 3, 0
    };

    glGenBuffers(1, &vbo_elements);
    GLCHECK(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_elements));
    GLCHECK(glBufferData(GL_ELEMENT_ARRAY_BUFFER, numElements_*sizeof(GLushort), quadElements, GL_STATIC_DRAW));
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    glGenBuffers(1, &vbo_data);
    GLCHECK(glBindBuffer(GL_ARRAY_BUFFER, vbo_data));
    GLCHECK(glBufferData(GL_ARRAY_BUFFER, numVertices_*sizeof(GLfloat)*2, quadVertices, GL_STATIC_DRAW));
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    GLCHECK(glGenVertexArrays(1, &vao));
    GLCHECK(glBindVertexArray(vao));
    glBindBuffer(GL_ARRAY_BUFFER, vbo_data);
    glVertexAttribPointer(vertexAttrib::POSITION, 2, GL_FLOAT, GL_FALSE, sizeof(GLfloat)*2, GLH_BUFFER_OFFSET(0));
    //glVertexAttribPointer(vertexAttrib::TEXCOORD, 2, GL_FLOAT, GL_FALSE, sizeof(GLfloat)*4, GLH_BUFFER_OFFSET(sizeof(GLfloat)*2));
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_elements);

    GLCHECK(glEnableVertexAttribArray(vertexAttrib::POSITION));
    //GLCHECK(glEnableVertexAttribArray(vertexAttrib::TEXCOORD));
    GLCHECK(glBindVertexArray(0));

    initialized_ = true;
}

void Quad::cleanup() {
    if(vbo_data) {
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glDeleteBuffers(1, &vbo_data);
    }
    if(vbo_elements) {
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
        glDeleteBuffers(1, &vbo_elements);
    }
    if(vao) {
        glBindVertexArray(0);
        glDeleteVertexArrays(1, &vao);
    }

    initialized_ = false;
}

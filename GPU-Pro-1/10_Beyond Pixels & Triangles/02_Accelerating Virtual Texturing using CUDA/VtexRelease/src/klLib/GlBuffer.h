/**
 *
 *  This software module was originally developed for research purposes,
 *  by Multimedia Lab at Ghent University (Belgium).
 *  Its performance may not be optimized for specific applications.
 *
 *  Those intending to use this software module in hardware or software products
 *  are advized that its use may infringe existing patents. The developers of 
 *  this software module, their companies, Ghent Universtity, nor Multimedia Lab 
 *  have any liability for use of this software module or modifications thereof.
 *
 *  Ghent University and Multimedia Lab (Belgium) retain full right to modify and
 *  use the code for their own purpose, assign or donate the code to a third
 *  party, and to inhibit third parties from using the code for their products. 
 *
 *  This copyright notice must be included in all copies or derivative works.
 *
 *  For information on its use, applications and associated permission for use,
 *  please contact Prof. Rik Van de Walle (rik.vandewalle@ugent.be). 
 *
 *  Detailed information on the activities of
 *  Ghent University Multimedia Lab can be found at
 *  http://multimedialab.elis.ugent.be/.
 *
 *  Copyright (c) Ghent University 2004-2009.
 *
 **/

#ifndef GL_BUFFER_H
#define GL_BUFFER_H

#include <GL/glew.h>

struct klVertex {
    klVec3 xyz;
    klVec2 uv;
    unsigned int color;
    klVec3 normal;
    klVec3 tangent;
    klVec3 binormal;

    bool operator== (const klVertex &other) {
        /*return (xyz == other.xyz) &&
               (uv == other.uv) &&
               (color == other.color) &&
               (normal == other.normal) &&
               (tangent == other.tangent) &&
               (binormal == other.binormal);*/

        // this early-outs when one of the fields is non equal
        // helps a lot with debug mode :D
        /*if ( xyz[0] != other.xyz[0] ||
             xyz[1] != other.xyz[1] ||
             xyz[2] != other.xyz[2] ||
             uv[0] != other.uv[0] ||
             uv[1] != other.uv[1] ||
             normal[0] != other.normal[0] ||
             normal[1] != other.normal[1] ||
             normal[2] != other.normal[2] ||
             tangent[0] != other.tangent[0] ||
             tangent[1] != other.tangent[1] ||
             tangent[2] != other.tangent[2] ||
             binormal[0] != other.binormal[0] ||
             binormal[1] != other.binormal[1] ||
             binormal[2] != other.binormal[2]
        ) {
            return false;
        } else {
            return true;
        }*/
        return (memcmp(this,&other,sizeof(klVertex)) == 0);
    }
};

class klGlBuffer{
protected:
    GLuint hand;
    int size;
    int target;
public:

    klGlBuffer(int _target, const void *data, int _size) : target(_target), size(_size) {
        glGenBuffers(1, &hand);
        bind();
        glBufferData(target, size, data, GL_STATIC_DRAW_ARB);
        unbind();
    }

    void bind(void) {
        glBindBuffer(target,hand);
    }

    void unbind(void) {
        glBindBuffer(target,0);
    }

    unsigned int handle(void) {
        return hand;
    }

    void discard(void) {
        bind();
        glBufferData(target, size, 0, GL_DYNAMIC_DRAW);
    }

    size_t sizeInBytes(void) {
        return size;
    }

    ~klGlBuffer(void) {
        glDeleteBuffers(1, &hand);
    }
};

class klVertexBuffer : public  klGlBuffer {
public:

    klVertexBuffer(const klVertex *elements, int numElements)
        : klGlBuffer(GL_ARRAY_BUFFER_ARB,elements,numElements * sizeof(klVertex)) {}

    void setPointers(void) {
       // glColorPointer(4, GL_UNSIGNED_BYTE, sizeof(klVertex), (void *)offsetof(klVertex,color));
       // glNormalPointer(GL_FLOAT, sizeof(klVertex), (void *)offsetof(klVertex,normal));
        //glTexCoordPointer(2, GL_FLOAT, sizeof(klVertex), (void *)offsetof(klVertex,uv));
        
        glVertexAttribPointer(1, 2, GL_FLOAT, false, sizeof(klVertex), (void *)offsetof(klVertex,uv));
        glVertexAttribPointer(2, 4, GL_UNSIGNED_BYTE, true, sizeof(klVertex), (void *)offsetof(klVertex,color));
        glVertexAttribPointer(3, 3, GL_FLOAT, false, sizeof(klVertex), (void *)offsetof(klVertex,tangent));
        glVertexAttribPointer(4, 3, GL_FLOAT, false, sizeof(klVertex), (void *)offsetof(klVertex,binormal));
        glVertexAttribPointer(5, 3, GL_FLOAT, false, sizeof(klVertex), (void *)offsetof(klVertex,normal));

        //glVertexAttribPointer(1, 3, GL_FLOAT, false, sizeof(klVertex), (void *)offsetof(klVertex,tangent));
        //glVertexAttribPointer(2, 3, GL_FLOAT, false, sizeof(klVertex), (void *)offsetof(klVertex,binormal));

        glVertexPointer(3, GL_FLOAT, sizeof(klVertex), (void *)offsetof(klVertex,xyz));
    }

    void enableClientState(void) {
        glEnableClientState(GL_VERTEX_ARRAY);
       // glEnableClientState(GL_COLOR_ARRAY);
       // glEnableClientState(GL_NORMAL_ARRAY);
        //glEnableClientState(GL_TEXTURE_COORD_ARRAY);
        glEnableVertexAttribArray(1);
        glEnableVertexAttribArray(2);
        glEnableVertexAttribArray(3);
        glEnableVertexAttribArray(4);
        glEnableVertexAttribArray(5);
        //glEnableVertexAttribArray(1);
        //glEnableVertexAttribArray(2);
    }

};

class klIndexBuffer : public klGlBuffer {
    int numIndex;
    unsigned int maxIndex;
public:

    klIndexBuffer(const unsigned int *elements, int numElements)
        : klGlBuffer(GL_ELEMENT_ARRAY_BUFFER_ARB,elements,numElements*sizeof(int)), maxIndex(0), numIndex(numElements)
    {
        for( int i=0; i<numElements; i++ ) {
            if ( elements[i] > maxIndex ) maxIndex = elements[i];
        }
    }

    void drawElements(void) {
	    glDrawRangeElements(GL_TRIANGLES, 0, maxIndex, numIndex, GL_UNSIGNED_INT, (void *)(0) );
    }
};

#endif //GL_BUFER_H
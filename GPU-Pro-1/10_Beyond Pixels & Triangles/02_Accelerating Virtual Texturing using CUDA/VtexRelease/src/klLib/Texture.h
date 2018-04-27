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

#ifndef KLTEXTURE_H
#define KLTEXTURE_H

#include "GlBuffer.h"
#include "Manager.h"
#include "TextureFile.h"
#include "FileSystem.h"

class klTexture {
    klFileTimeStamp timeStamp;
    unsigned int handle;
    int width;
    int height;
    int depth;
    int layers;
    int internalFormat;
    int target; //1D,2D,3D,CUBE,...
public:  

    klTexture(void);
    klTexture(const char *fileName);

    void reload(const char *fileName);

    void bind(int unit);
    unsigned int getHandle(void) { return handle; }
    static void unBindAll(void);

    // Load the texture data from the specified file
    void setData( klTextureFileReader &reader );
    
    // Creates an texture with the given parameters
    // * Set type to 2D and depth >= 1 to create a texture array.
    // * Depth has to be 0 for a 2D texture or a
    //   3D-texture/texture-array with 1 layer will be created.
    // * Used mainly for render to texture
    // * Pixels can be null to only reserve space
    void setData(klTextureFileFormat format, klTextureType texType, int width, int height, int depth=0, void *pixels = NULL);

    int getWidth(void) const { return width; }
    int getHeight(void) const { return height; }

    int getSizeInBytes(void) const;
};

class klTextureManager : public klManager<klTexture> {
protected:
	virtual klTexture *getInstance(const char *name);
};

extern klTextureManager textureManager;

/*
    This is a surface we can render to but it can't be used as a texture.
*/
class klRenderBuffer {
    int width;
    int height;
    int format;
    unsigned int handle;
public:
    klRenderBuffer(const char *name);

    int getHandle(void) { return handle; }
    int getWidth(void) const { return width; }
    int getHeight(void) const { return height; }


};

class klRenderBufferManager : public klManager<klRenderBuffer> {
protected:
	virtual klRenderBuffer *getInstance(const char *name);
};

extern klRenderBufferManager renderBufferManager;

/*
    Encapsulates a render target configuration.

    Note: In increasing performance order.
    – Multiple FBOs
        * create a separate FBO for each texture you want to
        render to
        * switch using BindFramebuffer()
    – Single FBO, multiple texture attachments
        * textures should have same format and dimensions
        * use FramebufferTexture() to switch between
        textures
    – Single FBO, multiple texture attachments
        * attach textures to different color attachments
        * use glDrawBuffer() to switch rendering to different
        color attachments
*/
class klRenderTarget {
    int width;
    int height;
    int format;
    unsigned int handle;

    void checkSizes(int w, int h);
    static int currentHandle;
public:
    klRenderTarget(void);
    klRenderTarget(int _handle);
    ~klRenderTarget(void);

    bool checkComplete(void);

    void bind(void);

    void setDepth(klRenderBuffer *buff);
    void setDepth(klTexture *tex);

    void setColor(klRenderBuffer *buff, int index=0);
    void setColor(klTexture *tex, int index=0, int mipLevel=0);

    static const unsigned int DEPTH_ONY = (1<<31);
    static const unsigned int BUFFER0   = 1;
    static const unsigned int BUFFER1   = 2;
    static const unsigned int BUFFER2   = 4;
    static const unsigned int BUFFER3   = 8;
    static const unsigned int BUFFER4   = 16;
    static const unsigned int BUFFER5   = 32;
    static const unsigned int BUFFER6   = 64;
    static const unsigned int BUFFER7   = 128;
    static const unsigned int BUFFER8   = 256;

    /*
        Setup opengl for rendering to his buffer use the special
        DEPTH_ONY index to render to the depthbuffer only
    */
    void startRendering(unsigned int bufferMask=BUFFER0);

    int getWidth(void) const { return width; }
    int getHeight(void) const { return height; }

    void setSystemSize(int width, int height) {
        assert(handle == 0);
        this->width = width; this->height = height;
    }
};

/*
    Bind this render target to render to the system-window
    Note that calling any set* methots will fail on this 
    rendertarget.
*/
extern klRenderTarget systemRenderTarget;

class klPixelBuffer : public klGlBuffer {
    int width;
    int height;
    int bpp;
public:
    klPixelBuffer(int width, int height, int bytesPerPixel) : klGlBuffer(GL_PIXEL_PACK_BUFFER_ARB,NULL,width*height*bytesPerPixel) {
        this->width = width;
        this->height = height;
        bpp = bytesPerPixel;
    }

    void readPixels(void) {
        assert(bpp == 4);
        glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, hand);
        glReadPixels(0, 0, width, height, GL_BGRA, GL_UNSIGNED_BYTE, 0); 
        glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, 0);
    }

    void downloadToTex(klTexture *tex, int x, int y) {
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, hand);
        tex->bind(0);
        glTexSubImage2D(GL_TEXTURE_2D, 0, x, y, width, height, GL_BGRA, GL_UNSIGNED_BYTE, 0);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
    }

};


#endif //KLTEXTURE_H
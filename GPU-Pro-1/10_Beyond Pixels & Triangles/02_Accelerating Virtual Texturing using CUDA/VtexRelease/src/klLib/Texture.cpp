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

#include "shared.h"
#include "FileSystem.h"
#include "Texture.h"
#include "hdrloader.h"
#include "PixieFont.h"

// Define this so all float textures will use the 16bit "half"
// internal format
#define USE_16BIT_FLOAT

klTexture::klTexture( const char *fileName ) {
    glGenTextures(1, &handle);
    timeStamp = 0;
    reload(fileName);
}

void klTexture::reload( const char *fileName ) {
    // Build-in textures (starting with "_") never reload.
    if ( fileName[0] != '_' ) {

        klLog("Loading %s...",fileName);

        klFileName fname(fileName);
        klFileTimeStamp oldTimeStamp = timeStamp;
        std::istream *ss = fileSystem.openFile(fileName,&timeStamp);

        if ( !ss ) return;

        // If file was modifed so reload it...
        if ( oldTimeStamp != timeStamp ) {
            if ( fname.getExt() == "kltx" ) {
                klTextureFileReader r(*ss);
                setData(r);
            } else if ( fname.getExt() == "hdr" ) {
                HDRLoaderResult result;
                if ( HDRLoader::load(*ss,result) ) {
                    setData(KLTX_RGBA32F,KLTX_2D,result.width,result.height,0,result.pixels);
                } else {
                    klFatalError("Error loading .hdr file %s\n",fileName);
                }
            } else {
                klFatalError("Image file format not supported %s\n",fileName);
            }
        }

        delete ss;
    } else if ( strcmp(fileName,"_font")==0 ) {
        int totalCols = nr_chrs_S*charheight_S;
        int pixelSize = charheight_S*totalCols*4;
        unsigned char *buff = (unsigned char *)malloc(pixelSize);
        memset(buff,0,pixelSize);

        for ( int i=0;i<nr_chrs_S;i++ ) {
            DrawChar(buff+i*charheight_S*4, totalCols*4, i+firstchr_S, 0xFFFFFFFF);
        }

        setData(KLTX_RGBA8, KLTX_2D, totalCols, charheight_S, 0, buff);
        bind(0);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

        free(buff);
    }
}

void klTexture::setData( klTextureFileReader &reader ) {
    klTextureFileFormat fileFormat;
    int mipmaps;
    klTextureType texType;
    reader.readHeader(texType,fileFormat,width,height,depth,mipmaps,layers);
    int channelType = GL_UNSIGNED_BYTE;

    int format;
    bool compressed;
    switch (fileFormat) {
        case KLTX_RGB8:
            format = GL_RGB;
            internalFormat = GL_RGB8;
            compressed = false;
            break;
        case KLTX_RGBX8:
            format = GL_RGBA;
            internalFormat = GL_RGB8;
            compressed = false;
            break;
        case KLTX_RGBA8:
            format = GL_RGBA;
            internalFormat = GL_RGBA8;
            compressed = false;
            break;
        case KLTX_RGB_DXT1:
            format = 0;
            internalFormat = GL_COMPRESSED_RGB_S3TC_DXT1_EXT;
            compressed = true;
            break;
        case KLTX_RGBA_DXT1:
            format = 0;
            internalFormat = GL_COMPRESSED_RGBA_S3TC_DXT1_EXT;
            compressed = true;
            break;
        case KLTX_RGBA_DXT3:
            format = 0;
            internalFormat = GL_COMPRESSED_RGBA_S3TC_DXT3_EXT;
            compressed = true;
            break;
        case KLTX_RGBA_DXT5:
            format = 0;
            internalFormat = GL_COMPRESSED_RGBA_S3TC_DXT5_EXT;
            compressed = true;
            break;
        case KLTX_R_RGTC:
            format = 0;
            internalFormat = GL_COMPRESSED_RED_RGTC1_EXT;
            compressed = true;
            break;
        case KLTX_SR_RGTC:
            format = 0;
            internalFormat = GL_COMPRESSED_SIGNED_RED_RGTC1_EXT;
            compressed = true;
            break;
        case KLTX_DEPTH24:
            klFatalError("Depth buffers are not supported in texture files");
            break;
        case KLTX_L32F:
            format = GL_LUMINANCE;
            internalFormat = GL_LUMINANCE32F_ARB;
            channelType = GL_FLOAT;
            compressed = false;
            break;
        case KLTX_LA32F:
            format = GL_LUMINANCE_ALPHA;
            internalFormat = GL_LUMINANCE_ALPHA32F_ARB;
            channelType = GL_FLOAT;
            compressed = false;
            break;
        case KLTX_RGBA32F:
            format = GL_RGBA;
#ifdef USE_16BIT_FLOAT
            internalFormat = GL_RGBA16F_ARB;
#else
            internalFormat = GL_RGBA32F_ARB;
#endif
            channelType = GL_FLOAT;
            compressed = false;
            break;
        default:
            assert(0);
    }

    if ( depth == 0 ) {
        if ( layers > 1 ) {
            if ( texType == KLTX_CUBE ) {
                assert(layers==6);
                target = GL_TEXTURE_CUBE_MAP;
            } else {
                target = GL_TEXTURE_2D_ARRAY_EXT;
            }
        } else {
            target = GL_TEXTURE_2D;
        }
    } else {
        assert(texType==KLTX_3D);
        assert(layers==1);
        target = GL_TEXTURE_3D;
    }

    bind(0);

    glTexParameteri (target, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri (target, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri (target, GL_TEXTURE_WRAP_R, GL_REPEAT);
    glTexParameteri (target, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri (target, GL_TEXTURE_MIN_FILTER, (mipmaps <= 1) ? GL_LINEAR : GL_LINEAR_MIPMAP_LINEAR);

    if ( texType == KLTX_2D ) {
        float maxAniso;
        glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, &maxAniso);
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, maxAniso);
    }

    int levelWidth = width;
    int levelHeight = height;
    int levelDepth = depth;

    for ( int level=0; level<mipmaps; level++ ) {
        unsigned int dataSize;
        reader.readMipMap(dataSize);
        unsigned char *data = (unsigned char *)malloc(dataSize*layers);

        if ( !depth ) {
            for ( int i=0; i<layers; i++ ) {
                reader.readLayerData(data+i*dataSize);
            }

            if ( layers == 1 ) {
                if ( !compressed ) {
                    glTexImage2D(target,level,internalFormat,levelWidth,levelHeight,0,format,channelType,data);                    
                } else {
                    glCompressedTexImage2D(target,level,internalFormat,levelWidth,levelHeight,0,dataSize,data);
                }
            } else {
                if ( target == GL_TEXTURE_CUBE_MAP ) {
                    for ( int i=0;i<6; i++ ) {
                        if ( !compressed ) {
                            glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X+i,level,internalFormat,levelWidth,levelHeight,0,format,channelType,data+i*dataSize);                    
                        } else {
                            glCompressedTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X+i,level,internalFormat,levelWidth,levelHeight,0,dataSize,data+i*dataSize);
                        }
                    }
                } else {
                    if ( !compressed ) {
                        glTexImage3D(target,level,internalFormat,levelWidth,levelHeight,layers,0,format,channelType,data);
                    } else {
                        glCompressedTexImage3D(target,level,internalFormat,levelWidth,levelHeight,layers,0,dataSize*layers,data); 
                    }    
                }
            }
        } else {
            reader.readLayerData(data);

            if ( !compressed ) {
                glTexImage3D(target,level,internalFormat,levelWidth,levelHeight,levelDepth,0,format,channelType,data);
            } else {
                glCompressedTexImage3D(target,level,internalFormat,levelWidth,levelHeight,levelDepth,0,dataSize,data); 
            }
        }

        free(data);

        // Mipmaps
        levelWidth  = max(1,levelWidth>>1);
        levelHeight = max(1,levelHeight>>1);
        levelDepth  = max(1,levelDepth>>1);
    }

    klCheckGlErrors();
}

void klTexture::setData(klTextureFileFormat format, klTextureType texType, int width, int height, int depth, void *pixels) {
    int glFormat;
    bool compressed;
    int channelType = GL_UNSIGNED_BYTE;

    switch (format) {
        case KLTX_RGB8:
            glFormat = GL_RGB;
            internalFormat = GL_RGB8;
            compressed = false;
            break;
        case KLTX_RGBX8:
            glFormat = GL_RGBA;
            internalFormat = GL_RGB8;
            compressed = false;
            break;
        case KLTX_RGBA8:
            glFormat = GL_RGBA;
            internalFormat = GL_RGBA8;
            compressed = false;
            break;
        case KLTX_RGBA32F:
            glFormat = GL_RGBA;
#ifdef USE_16BIT_FLOAT
            internalFormat = GL_RGBA16F_ARB;
#else
            internalFormat = GL_RGBA32F_ARB;
#endif
            channelType = GL_FLOAT;
            compressed = false;
            break;
        case KLTX_DEPTH24:
            glFormat = GL_DEPTH_COMPONENT; 
            internalFormat = GL_DEPTH_COMPONENT24_ARB;
            channelType = GL_FLOAT;
            compressed = false;
            break;
        default:
            assert(0);
    }
    
    this->width = width;
    this->height = height;
    this->layers = 1;
    this->depth = 1;

    target = GL_TEXTURE_2D;
    if ( depth == 0 ) {
        if ( texType == KLTX_CUBE ) {
            target = GL_TEXTURE_CUBE_MAP;
        } else if ( texType == KLTX_2D ) {
            target = GL_TEXTURE_2D;            
        } else {
            assert(0);
        }
    } else {
        if ( texType == KLTX_3D ) {
            this->depth = depth;
            target = GL_TEXTURE_3D; 
        } else if ( texType == KLTX_2D ) {
            this->layers = depth;
            target = GL_TEXTURE_2D_ARRAY_EXT;
        } else {
            assert(0);
        }
    }

    // Now the texture is fully set-up we can bind it
    klCheckGlErrors();
    bind(0);
    klCheckGlErrors();

    // And now load "empty" data to it
    glTexParameteri (target, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri (target, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri (target, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri (target, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    klCheckGlErrors();

    if ( depth == 0 ) {
        if ( texType == KLTX_CUBE ) {
            for ( int i=0; i<6; i++ ) {
                glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X+i,0,internalFormat,
                                    width,height,0,glFormat,channelType,pixels);   
            }
        } else if ( texType == KLTX_2D ) {
            glTexImage2D(target,0,internalFormat,width,height,0,glFormat,channelType,pixels);                    
        }
    } else {
        if ( texType == KLTX_3D ) {
            glTexImage3D(target,0,internalFormat,width,height,depth,0,glFormat,channelType,pixels);   
        } else if ( texType == KLTX_2D ) {
            glTexImage3D(target,0,internalFormat,width,height,depth,0,glFormat,channelType,pixels); 
        }
    }
    klCheckGlErrors();
}


void klTexture::bind(int unit) {
    if ( glState.currentTextureUnit != unit ) {
        glActiveTexture(GL_TEXTURE0+unit);
        glState.currentTextureUnit = unit;
    } 
    glBindTexture(target,handle);
}

void klTexture::unBindAll(void) {
    for ( int i=0; i<8; i++ ) {
        glActiveTexture(GL_TEXTURE0+i);
        glBindTexture(GL_TEXTURE_2D,0);
    } 
    glActiveTexture(GL_TEXTURE0);
    glState.currentTextureUnit = 0;
}

int klTexture::getSizeInBytes(void) const {
    float bytesPerPixel;

    switch (internalFormat) {
        case GL_DEPTH_COMPONENT24_ARB:      bytesPerPixel = 4; break;
        case GL_RGBA16F_ARB:                bytesPerPixel = 8; break;
        case GL_RGBA8:                      bytesPerPixel = 4; break;
        case GL_RGBA32F_ARB:                bytesPerPixel = 16; break;
        case GL_LUMINANCE32F_ARB:           bytesPerPixel = 4; break;
        case GL_LUMINANCE_ALPHA32F_ARB:     bytesPerPixel = 8; break;
        case GL_COMPRESSED_SIGNED_RED_RGTC1_EXT: bytesPerPixel = 0.5; break;
        case GL_COMPRESSED_RED_RGTC1_EXT:        bytesPerPixel = 0.5; break;
        case GL_COMPRESSED_RGBA_S3TC_DXT1_EXT:   bytesPerPixel = 0.5; break;
        case GL_COMPRESSED_RGB_S3TC_DXT1_EXT:    bytesPerPixel = 0.5; break;
        case GL_COMPRESSED_RGBA_S3TC_DXT3_EXT:   bytesPerPixel = 0.5; break;
        case GL_COMPRESSED_RGBA_S3TC_DXT5_EXT:   bytesPerPixel = 1; break;
    }

    // fixme no mipmaps?
    int stack = max(max(layers,depth),1);
    return (int)(width*height*bytesPerPixel*stack);
}

klTexture *klTextureManager::getInstance(const char *name) {
    return new klTexture(name);
}

klTextureManager textureManager;

////////////////////////////////////// klRenderBuffer ///////////////////////////////////

klRenderBuffer *klRenderBufferManager::getInstance(const char *name) {
    return new klRenderBuffer(name);
}

klRenderBuffer::klRenderBuffer(const char *name) {
    /*Parses strings of format name_[format]_[width]_[height]*/

    std::vector<std::string> args;
    args = explode(name, "_");

    if( args.size() < 4 ) {
        klFatalError("Invalid surface name '%s'", name);        
    }

    if ( args[1] == "depth" ) {
        format = GL_DEPTH_COMPONENT24;
    } else if ( args[1]  == "rgba" ) {
        format = GL_RGBA;
    } else if ( args[1]  == "bgra" ) {
        format = GL_BGRA;
    } else {
        klFatalError("Surface name '%s' invalid format '%s'.", name, args[1].c_str() );    
    }

    width = atoi(args[2].c_str());
    height = atoi(args[3].c_str());

    glGenRenderbuffersEXT(1, &handle);
    glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, handle);
    glRenderbufferStorageEXT(GL_RENDERBUFFER_EXT, format, width, height);
}

klRenderBufferManager renderBufferManager;

/////////////////////////////// klRenderTarget /////////////////////////////////////////////

int klRenderTarget::currentHandle;
klRenderTarget systemRenderTarget(0);

klRenderTarget::klRenderTarget(int _handle) : width(0), height(0), format(0), handle(_handle) {}

klRenderTarget::klRenderTarget(void) : width(0), height(0), format(0) {
    glGenFramebuffersEXT(1, &handle);
}

klRenderTarget::~klRenderTarget(void) {
    glDeleteFramebuffersEXT(1, &handle);
}

void klRenderTarget::bind(void) {
    if ( currentHandle == handle ) return;
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, handle);
    currentHandle = handle;
}

void klRenderTarget::checkSizes(int w, int h) {
    if ( width == 0 ) {
        width = w;
    } else if ( width != w ) {
        assert(0);
    }
    if ( height == 0 ) {
        height = h;
    } else if ( height != h ) {
        assert(0);
    }
}

void klExitHardwareNotSupported(void);

bool klRenderTarget::checkComplete(void) {
    int status = glCheckFramebufferStatusEXT(GL_FRAMEBUFFER_EXT);
    switch(status) {
        case GL_FRAMEBUFFER_COMPLETE_EXT:
            return true;
        case GL_FRAMEBUFFER_UNSUPPORTED_EXT:
            klExitHardwareNotSupported();
            break;
        default:
            klFatalError("Frame buffer error %x",status);
            break;
    } 
    return false;
}

void klRenderTarget::setDepth(klRenderBuffer *buff) {
    assert(handle > 0);
    checkSizes(buff->getWidth(),buff->getHeight());
    bind();
    glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT,
                                GL_RENDERBUFFER_EXT, buff->getHandle());
    klCheckGlErrors();
}

void klRenderTarget::setDepth(klTexture *tex) {
    assert(handle > 0);
    checkSizes(tex->getWidth(),tex->getHeight());
    bind();
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT,
                              GL_TEXTURE_2D, tex->getHandle(), 0);
    klCheckGlErrors();
}

void klRenderTarget::setColor(klRenderBuffer *buff, int index) {
    assert(handle > 0);
    klCheckGlErrors();
    checkSizes(buff->getWidth(),buff->getHeight());
    bind();
    glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT+index,
                                 GL_RENDERBUFFER_EXT, buff->getHandle());
    klCheckGlErrors();
}

void klRenderTarget::setColor(klTexture *tex, int index, int mipLevel) {
    assert(handle > 0);
    klCheckGlErrors();
    if ( mipLevel == 0 ) {
        checkSizes(tex->getWidth(),tex->getHeight());
    }
    bind();
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT+index,
                              GL_TEXTURE_2D, tex->getHandle(), mipLevel);
    klCheckGlErrors();
}

void klRenderTarget::startRendering(unsigned int bufferMask) {
    bind();
    if ( handle == 0 ) {
        // Set it back to the window system back buffer
        glDrawBuffer(GL_BACK);
        glReadBuffer(GL_BACK);
        return;
    }

    if ( bufferMask == DEPTH_ONY ) {
        glDrawBuffer(GL_NONE);
        glReadBuffer(GL_NONE);
    } else {
        unsigned int drawBuffers[31];
        int numDrawBuffers = 0;

        for ( int i=0;i<31; i++ ) {
            int mask = 1<<i;
            if (bufferMask&mask) {
                drawBuffers[numDrawBuffers] = GL_COLOR_ATTACHMENT0_EXT+i;
                numDrawBuffers++;
            }
            assert(numDrawBuffers);
            glDrawBuffers(numDrawBuffers,drawBuffers);
            glReadBuffer(drawBuffers[0]);
        }

        //glDrawBuffer(GL_COLOR_ATTACHMENT0_EXT+index);
        //glReadBuffer(GL_COLOR_ATTACHMENT0_EXT+index);
    }

#ifdef _DEBUG
    checkComplete();
#endif 

    klCheckGlErrors();
}
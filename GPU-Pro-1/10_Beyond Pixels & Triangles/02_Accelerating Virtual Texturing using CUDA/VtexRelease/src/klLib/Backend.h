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

#ifndef KLBACKEND_H
#define KLBACKEND_H

#include "Model.h"
#include "Camera.h"

struct klInstanceInfo {
    static const int NUM_PARAMETERS = 16;

    klMaterial *       maskMaterial;               //If nonnull override only this material
    klMaterial *       overrideMaterial;           //Override material stored in model with this material
    float              parameters[NUM_PARAMETERS]; //Can be fed to shaders

    klInstanceInfo(void) {
        overrideMaterial = 0;
        maskMaterial = 0;
        memset(parameters,0,sizeof(float) * NUM_PARAMETERS);
    }
};

struct klBackendSurface {
    klVertexBuffer *vertices;
    klIndexBuffer  *indices;
    klMaterial     *material;
    klInstanceParameters parm;
};

struct klScissorRect {
    int x;
    int y;
    int width;
    int height;
    klScissorRect(int _x, int _y, int _w, int _h) : x(_x), y(_y), width(_w), height(_h) {}
};

class klRenderBackend {
public:

    enum MaterialTransform {
        MT_NONE,
        MT_SHADOW
    };

    klRenderBackend(void);

    // Start collecting frame data
    void startFrame(float time = 0.0f);

    // Set destination buffer(s) to render to & clear if needed
    void setDestination(klRenderTarget *tgt, int clearBits = GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT, int bufferMask = 1);

    // Start a new view
    void startView(const klCamera &camera);

    // Set the scissor rect
    void setScissor(const klScissorRect &rect);

    // Set material transform (applied to any folowing drawmodel calls)
    void setMaterialTransform(MaterialTransform mt) { currentMaterialTransform = mt; }

    // Draw the specified model
    void drawModel(const klModel *model,
                   const klMatrix4x4 &modelTransform,
                   const klInstanceInfo *instance = NULL);

    // Actually execute all queued drawing commands, this can be used when external GL code needs 
    // to be sure something actually got drawn etc ... and we don't want to wait till the end of the 
    // current frame
    void flush(void);

    // End the frame drawing, this flushes the backend and draws debug data on top
    void finishFrame(void);

    // Draw a line for debugging
    void debugLine(const klVec3 &start, const klVec3 &end) {
        DebugLine l;
        l.start = start; l.end = end;
        debugLines.push_back(l);
    }

    // Draw a string for debugging
    void drawString(int x, int y, const char *str, int color) {
        DebugString s;
        s.x = x; s.y = y;
        s.color = color;
        s.str = str;
        debugStrings.push_back(s);
    }

    static size_t AllocInstanceId(void); 

private:
    void *buffer;
    size_t used;
    size_t limit;
    klCamera currentCamera;

    struct DebugLine {
        klVec3 start;
        klVec3 end;
    };

    struct DebugString {
        int x;
        int y;
        int color;
        std::string  str;
    };

    std::vector<DebugLine> debugLines;
    klMaterial *lineMaterial;
    std::vector<DebugString> debugStrings;
    klMaterial *fontMaterial;

    static size_t instanceIdCounter;
    static const int REALLOC_DELTA = (1024*1024);

    void *tempAlloc(size_t bytes);
    size_t tempAllocMark(void) { return used; }
    void *tempAllocMarkTrans(size_t mark) { return ((unsigned char *)buffer)+mark; }
    void clearTempAlloc(void);

    enum CommandType {
        CT_CAMERA,
        CT_DEST,
        CT_SURFS,
        CT_SCISSOR,
        CT_END
    };

    void putCommand(CommandType t);
    void putPointer(void *p);
    void putInteger(int i);

    //store an offset because tempalloc may reallocate the array
    size_t surfaceListOffset;
    int numSurfaces;

    void closeSurfaceList(void);

    static int compareSurfaces(const void *av,const void *bv);
    void *drawSurfaces(unsigned char *commands);

    klMaterial *transformMaterial(klMaterial *original, const klInstanceInfo *instance);

    float frameTime;
    MaterialTransform currentMaterialTransform;
};

extern klRenderBackend renderBackend;

#endif //KLBACKEND_H
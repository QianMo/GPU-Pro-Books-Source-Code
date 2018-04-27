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
#include "Backend.h"
#include "PixieFont.h"

size_t klRenderBackend::instanceIdCounter = 1;

size_t klRenderBackend::AllocInstanceId(void) {
    //WARNING: Not thread safe
    return ++instanceIdCounter;
}

void *klRenderBackend::tempAlloc(size_t bytes) {
    if ( (used+bytes) > limit ) {
        if ( bytes < REALLOC_DELTA ) {
            limit += REALLOC_DELTA;
        } else {
            limit += bytes+REALLOC_DELTA;
        }
        buffer = realloc(buffer,limit);
    }

    void *result = ((unsigned char *)buffer)+used;
    used += bytes;
    return result;
}

void klRenderBackend::clearTempAlloc(void) {
    used = 0;
}

void klRenderBackend::putCommand(CommandType t) {
    CommandType *ct = (CommandType *)tempAlloc(sizeof(t));
    *ct = t;
}

void klRenderBackend::putPointer(void *p) {
    void** pt = (void **)tempAlloc(sizeof(void *));
    *pt = p;
}

void klRenderBackend::putInteger(int i) {
    int* ip = (int *)tempAlloc(sizeof(int));
    *ip = i;
}

void klRenderBackend::startFrame(float time) {
    clearTempAlloc();
    frameTime = time;
    currentMaterialTransform = MT_NONE;
}

void klRenderBackend::setDestination(klRenderTarget *tgt, int clearBits, int bufferMask) {
    closeSurfaceList();
    putCommand(CT_DEST);
    putPointer((void *)tgt);
    putInteger(bufferMask);
    putInteger(clearBits);
}

void klRenderBackend::setScissor(const klScissorRect &rect) {
    closeSurfaceList();
    putCommand(CT_SCISSOR);
    klScissorRect *r = (klScissorRect*)tempAlloc(sizeof(klScissorRect));
    *r = rect;   
}

void klRenderBackend::startView(const klCamera &camera) {
    closeSurfaceList();
    putCommand(CT_CAMERA);
    currentCamera = camera;
    klCamera* cam = (klCamera*)tempAlloc(sizeof(klCamera));
    *cam = camera;
}

int klRenderBackend::compareSurfaces(const void *av,const void *bv) {
    const klBackendSurface *a = (const klBackendSurface *)av;
    const klBackendSurface *b = (const klBackendSurface *)bv;

    if ( a->material->getSort() != b->material->getSort() ) {
        return (int)a->material->getSort() - (int)b->material->getSort();
    }

    if ( a->material != b->material ) {
        return a->material - b->material;
    }
    
    if (a->parm.instanceId != b->parm.instanceId ) {
        return a->vertices - b->vertices;
    }

    if (a->vertices != b->vertices ) {
        return a->vertices - b->vertices;
    }

    if (a->indices != b->indices ) {
        return a->indices - b->indices;
    }

    // They are equal with respect to render order
    return 0;
}

void klRenderBackend::closeSurfaceList(void) {
    if ( surfaceListOffset == 0 ) return;

    void *firstSurface = tempAllocMarkTrans(surfaceListOffset);

    // Sort the surfaces
    qsort(firstSurface, numSurfaces, sizeof(klBackendSurface), compareSurfaces);

    // Add sentinel
    klBackendSurface *surf = (klBackendSurface *)tempAlloc(sizeof(klBackendSurface));
    surf->material = NULL;

    // Empty list
    surfaceListOffset = 0;
    numSurfaces = 0;  
}

klMaterial *klRenderBackend::transformMaterial(klMaterial *original, const klInstanceInfo *instance) {
    // Overriden by game code
    if (instance && instance->overrideMaterial ) {
        if ( instance->maskMaterial ) {
            if (instance->maskMaterial == original) {
                original = instance->overrideMaterial;
            }
        } else {
            original = instance->overrideMaterial;
        }
    }

    if ( currentMaterialTransform == MT_SHADOW ) {
        return original->getShadowMaterial();
    }

    return original;
}

void klRenderBackend::drawModel(const klModel *model,
                                const klMatrix4x4 &modelTransform,
                                const klInstanceInfo *instance)
{
    if ( !model || model->numSurfaces() == 0 ) return;

    // We calculate these here so they get only calculated once per model not
    // once per surface...
    size_t instanceId = AllocInstanceId();
    klMatrix4x4 finalTransform = currentCamera.getProjMatrix() * currentCamera.getViewMatrix() * modelTransform;
    klMatrix4x4 worldToModel = modelTransform.inverseRigid();
    klVec3 modelCamera = worldToModel * currentCamera.getOrigin();
    klVec3 modelLight;// = worldToModel * currentLight.getOrigin();

    // Start a new surface list if needed
    if ( surfaceListOffset == 0 ) {
        putCommand(CT_SURFS);
        surfaceListOffset = tempAllocMark();
        numSurfaces = 0;  
    }

    // Allocate the rest of the surfaces
    for ( int i=0; i<model->numSurfaces(); i++ ) {
        if ( !model->getSurface(i).material ) continue;
        klMaterial *mat = transformMaterial(model->getSurface(i).material,instance);
        if ( !mat) continue;

        klBackendSurface *surf = (klBackendSurface *)tempAlloc(sizeof(klBackendSurface));
        surf->vertices = model->getSurface(i).vertices;
        surf->indices = model->getSurface(i).indices;
        surf->material = mat;
        surf->parm.instanceId = instanceId;
        surf->parm.modelViewProjection = finalTransform;
        surf->parm.worldToModel = worldToModel;
        surf->parm.modelToWorld = modelTransform;
        surf->parm.modelSpaceCamera = modelCamera;
        surf->parm.modelSpaceLight = modelLight;
        if ( instance ) {
            memcpy(surf->parm.userParams,instance->parameters,sizeof(float)*klInstanceInfo::NUM_PARAMETERS);
        }
        numSurfaces++;
    }
}

void *klRenderBackend::drawSurfaces(unsigned char *commands) {
    klVertexBuffer *currentVertices = NULL;
    klIndexBuffer  *currentIndices = NULL;
    klMaterial     *currentMaterial = NULL;

    klBackendSurface *surf = (klBackendSurface *)commands;
    while ( surf->material ) {

        if ( surf->vertices != currentVertices ) {
            surf->vertices->bind();
            surf->vertices->setPointers();
            surf->vertices->enableClientState();
            currentVertices = surf->vertices;
        }

        if ( surf->indices != currentIndices ) {
            surf->indices->bind();
            currentIndices = surf->indices;
        }

        if ( surf->material != currentMaterial ) {
            if ( currentMaterial ) currentMaterial->reset();
            currentMaterial = surf->material;
            currentMaterial->setup();
        }

          
        currentMaterial->setupInstance(surf->parm);
        //  currentMaterial->setup();
        surf->indices->drawElements();
        //  currentMaterial->reset();

        surf++;
    }

    if ( currentMaterial ) currentMaterial->reset();
    return surf+1;
}

void klRenderBackend::flush(void) {
    closeSurfaceList();
    putCommand(CT_END);

    unsigned char *commands = (unsigned char *)buffer;
    CommandType command;
    int viewWidth;
    int viewHeight;
    
    effectManager.setGlobalParameter(klEffectManager::PK_TIME,&frameTime);

    // Do some gl state management here
    glDisable(GL_SCISSOR_TEST);

    while ( (command = *(CommandType *)commands) != CT_END ) {
        commands += 4;
        switch (command) {
            case CT_CAMERA: {
                currentCamera = *(klCamera *)commands;
                
                klMatrix4x4 viewProjection = currentCamera.getProjMatrix() * currentCamera.getViewMatrix();
                klMatrix4x4 invViewProjection = viewProjection.inverse();
                klMatrix4x4 invProjection = currentCamera.getProjMatrix().inverse();

                effectManager.setGlobalParameter(klEffectManager::PK_INVPROJECTION,invProjection.toCPtr());
                effectManager.setGlobalParameter(klEffectManager::PK_INVVIEWPROJECTION,invViewProjection.toCPtr());

                commands += sizeof(klCamera);
                break;
            }
            case CT_DEST: {
                klRenderTarget *tgt = *(klRenderTarget **)commands;
                commands += sizeof(void*);

                unsigned int buffers = *(int *)commands;
                commands += sizeof(int);

                int clearBits = *(int *)commands;
                commands += sizeof(int); 

                tgt->startRendering(buffers);
                glViewport(0,0,tgt->getWidth(),tgt->getHeight());
                if ( clearBits ) glClear(clearBits);
                break;
            }
            case CT_SCISSOR: {
                klScissorRect *rect = (klScissorRect *)commands;
                glScissor(rect->x, rect->y, rect->width, rect->height);
                glEnable(GL_SCISSOR_TEST);
                commands += sizeof(klScissorRect);
                break;
            }
            case CT_SURFS: {
                commands = (unsigned char*)drawSurfaces(commands);
                break;
            }
            default:
                assert(0);
        }
    }
    clearTempAlloc();
    klCheckGlErrors();
}

void klRenderBackend::finishFrame(void) {
    flush();

    // Debug lines
    if (debugLines.size()) {
        if (!lineMaterial) {
            lineMaterial = materialManager.getForName("debuglines");
        }
        if (lineMaterial) {
            klInstanceParameters lineParams;
            lineParams.instanceId = AllocInstanceId();
            lineParams.modelSpaceCamera = currentCamera.getOrigin();
            lineParams.modelSpaceLight.zero();
            lineParams.modelToWorld = klIdentityMatrix();
            lineParams.modelViewProjection = currentCamera.getProjMatrix() * currentCamera.getViewMatrix();
            lineParams.worldToModel = klIdentityMatrix();

            lineMaterial->setup();
            lineMaterial->setupInstance(lineParams);
            glBegin(GL_LINES);
            for ( int i=0; i<debugLines.size(); i++ ) {
                glVertex3fv(debugLines[i].start.toPtr());
                glVertex3fv(debugLines[i].end.toPtr());
            }
            glEnd();
            lineMaterial->reset();
        } else {
            klFatalError("No debug line material found");
        }
    }

    // Strings
    if (debugStrings.size()) {
        if (!fontMaterial) {
            fontMaterial = materialManager.getForName("font");
        }
        if (fontMaterial) {
            klInstanceParameters params;
            params.instanceId = AllocInstanceId();
            fontMaterial->setup();
            fontMaterial->setupInstance(params);


            
            float viewPort[4];
            glGetFloatv(GL_VIEWPORT,viewPort);
            float texScale = 1.0f / (nr_chrs_S*charheight_S);

            viewPort[2] *= 0.5f;
            viewPort[3] *= 0.5f;

            glBegin(GL_QUADS);
            for ( int i=0;i<debugStrings.size();i++ ) {
                std::string  &str = debugStrings[i].str;
                float x = debugStrings[i].x / viewPort[2] - 1.0f;
                float y = (-debugStrings[i].y) / viewPort[3] + 1.0f;
                for ( int j=0;j<str.size();j++) {
                    int theChar = str[j];
                    if ( theChar < firstchr_S || theChar > (nr_chrs_S+firstchr_S) ) {
                        continue;
                    }
                    int charAdj = theChar-firstchr_S;
                    float u0 = (charAdj*charheight_S)*texScale;
                    float u1 = (charAdj*charheight_S+lentbl_S[charAdj])*texScale;
                    float dx = lentbl_S[charAdj] / viewPort[2];
                    float dy = charheight_S /viewPort[3];

                    glTexCoord2f(u0,1.0);
                    glVertex3f(x,y,0.0);

                    glTexCoord2f(u1,1.0);
                    glVertex3f(x+dx,y,0.0);

                    glTexCoord2f(u1,0.0);
                    glVertex3f(x+dx,y+dy,0.0);

                    glTexCoord2f(u0,0.0);
                    glVertex3f(x,y+dy,0.0);

                    x+= ((lentbl_S[charAdj]+1) / viewPort[2]);
                }                
            }

            glEnd();
            fontMaterial->reset();
        } else {
            klFatalError("No font material found");
        }
    }

    // We clear this here so even strings drawn outside of
    // backend.begin/end will be drawn
    debugLines.clear();
    debugStrings.clear();
}

klRenderBackend::klRenderBackend(void) {
    surfaceListOffset = 0;
    numSurfaces = 0;
    buffer = NULL;
    used = 0;
    limit = 0;
    currentMaterialTransform = MT_NONE;
}

klRenderBackend renderBackend;
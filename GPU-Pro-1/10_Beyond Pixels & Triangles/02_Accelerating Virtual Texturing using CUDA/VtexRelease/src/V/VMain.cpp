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
#include "../klLib/Klubnika.h"
#include "../klLib/RenderWindow.h"
#include "PageResolver.h"
#include "PageProvider.h"
#include "PageCache.h"
#include "FpsCounter.h"

double time;
unsigned int lastFrameTime;
klFpsCounter fpsCounter;

klModel *worldModel;
klRenderTarget *deferredRenderTarget;

// Virtual texturing objects
AbstractPageResolver *pageResolver;
AbstractPageCache    *pageCache;
AbstractPageProvider *pageProvider;

// Models and material for postprocessing
klModel *quad;
klMaterial *postProcessMaterial;
klMaterial *postProcessMaterialPageId;

// Camera util class
struct CamInfo {
    klVec3 cameraPosition;
    klVec3 cameraDirection;
    float pitch;
    float yaw;

    void deriveDirection(void) {
        klMatrix3x3 rot;

        rot = klAxisAngleMatrix(klVec3(0.0,0.0,1.0),yaw);
        cameraDirection = rot * klVec3(1.0,0.0,0.0);
        klVec3 yAxis = rot * klVec3(0.0,1.0,0.0);
        rot = klAxisAngleMatrix(yAxis,pitch);
        cameraDirection = rot * cameraDirection;
    }
};

CamInfo currentCamInfo;

// Window listener class
class VListener : public RenderWindow::Listener {
    int lastX;
    int lastY;

    bool mouseIsDown;
    bool leftDown;
    bool rightDown;
    bool upDown;
    bool downDown;
    bool shiftDown;
public:

    float aspect;
    int width;
    int height;

    VListener(void) {
        leftDown = rightDown = upDown = downDown = shiftDown = false;
        mouseIsDown = false;
        lastX = lastY = -1;
    }

    virtual bool glInit(void) {
        glClearColor(1.0,1.0,1.0,0.0);

        // Initialize CUDA (should be done after opengl, as per nvidia docs)
        klGpuInit();

        // Start up the engine's material and effect manager
        materialManager.loadMaterials();
        postProcessMaterial = materialManager.getForName("postprocess");
        postProcessMaterialPageId = materialManager.getForName("postprocess_pageid");

        // Create the deferred rendertarget, we just create one with a resolution of 1280*1024
        // so we never have to reallocate it
        klTexture *diffuseRenderTexture = textureManager.getForName("_diffuseRT");
        diffuseRenderTexture->setData(KLTX_RGBA8, KLTX_2D, 1280, 1024, 0, NULL);
        klTexture *pageIdRenderTexture = textureManager.getForName("_pageidRT");
        pageIdRenderTexture->setData(KLTX_RGBA8, KLTX_2D, 1280, 1024, 0, NULL);
        pageIdRenderTexture->bind(0);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        klRenderBuffer *depthRenderBuffer  = renderBufferManager.getForName("deferred_depth_1280_1024");
        
        klTexture *testTexture = textureManager.getForName("_font");

        deferredRenderTarget = new klRenderTarget();
        deferredRenderTarget->setDepth(depthRenderBuffer);
        deferredRenderTarget->setColor(pageIdRenderTexture,0,0);
        deferredRenderTarget->setColor(diffuseRenderTexture,1,0);

        // Create the virtual texturing objects
        pageProvider = new DiskPageProvider("./base/pagefiles/test16k.pages");
        pageResolver = new GpuPageResolver(pageProvider->getTextureInfo(),deferredRenderTarget);
        pageCache    = new SimplePageCache(pageProvider->getTextureInfo());

        pageCache->setProvider(pageProvider);
        pageProvider->setCache(pageCache);
        pageResolver->setCache(pageCache);

        // Force it to load the lowest miplevel
        pageCache->flush();

        // Load some models to render
        worldModel = modelManager.getForName("./base/models/test.pbj");
        quad = modelManager.getForName("__quad");

        // Place the camera
        currentCamInfo.cameraPosition.set(200.0f,200.0f, 80.781f);
        currentCamInfo.pitch = 0.61100062f;
        currentCamInfo.yaw = -40.219006f;
        currentCamInfo.deriveDirection();

        lastFrameTime = GetTickCount();
        time = 0;

        return true;
    }

    virtual bool glFree(void) {
        delete pageCache;
        delete pageResolver;
        delete pageProvider;
        delete deferredRenderTarget;
        return true;        
    }

    virtual void resize(size_t width, size_t height, float aspect) {
        systemRenderTarget.setSystemSize(width, height);
        this->aspect = aspect;
        this->width = width;
        this->height = height;
    }

    virtual void mouseMove(int x, int y) {
        if ( lastX >= 0 && mouseIsDown ) {
            int dx = x-lastX;
            int dy = y-lastY;

            currentCamInfo.yaw += -dx*0.01f;
            currentCamInfo.pitch += dy*0.01f;
            currentCamInfo.pitch = min(max(-M_PI / 2.0f + 0.01f,currentCamInfo.pitch),M_PI / 2.0f);

            currentCamInfo.deriveDirection();
        }

        lastX = x;
        lastY = y;    
    }

    virtual void mouseDown(MouseButtonType btn) {
        if ( btn == RenderWindow::Listener::MOUSE_LEFT ) {
            mouseIsDown = true;
        }
    }

    virtual void mouseUp(MouseButtonType btn) {
        if ( btn == RenderWindow::Listener::MOUSE_LEFT ) {
            mouseIsDown = false;
        }    
    }
 
    virtual void onChar(int theChar) {
        if ( theChar == '`' ) {
            console.show();
        }
    }

    virtual void keyDown(size_t key) {
        switch (key) {
            case VK_UP:
                upDown = true;
                break;
            case VK_DOWN:
                downDown = true;
                break;
            case VK_LEFT:
                leftDown = true;
                break;
            case VK_RIGHT:
                rightDown = true;
                break;
        }
    }

    virtual void keyUp(size_t key) {
        switch (key) {
            case VK_UP:
                upDown = false;
                break;
            case VK_DOWN:
                downDown = false;
                break;
            case VK_LEFT:
                leftDown = false;
                break;
            case VK_RIGHT:
                rightDown = false;
                break;
        }
    }

    void animateCamera(float dt) {
        float scale = ((shiftDown) ? 1600.0f : 400.0f) * dt;

        klVec3 left;
        left.cross(currentCamInfo.cameraDirection,klVec3(0.0f,0.0f,1.0f));
        left.normalize();
        
        if ( upDown ) {
            currentCamInfo.cameraPosition += currentCamInfo.cameraDirection * scale;
        }

        if ( downDown ) {
            currentCamInfo.cameraPosition -= currentCamInfo.cameraDirection * scale;
        }

        if ( leftDown ) {
            currentCamInfo.cameraPosition -= left * scale;
        } 

        if ( rightDown ) {
            currentCamInfo.cameraPosition += left * scale;
        } 
    }
};

void DrawGuiElement(int x, int y, int width, int height, int viewWidth, int viewHeight, char *materialName) {
    float fx = (x+(width/2.0f)-(viewWidth/2.0f )) / (viewWidth/2.0f);
    float fy = (y+(height/2.0f)-(viewHeight/2.0f)) / (viewHeight/2.0f);


    float fw = (width / (float)viewWidth) * 1.0f;
    float fh = -(height / (float)viewHeight) * 1.0f;

    klInstanceInfo ifo;
    ifo.parameters[0] = fw;
    ifo.parameters[1] = fh;
    ifo.parameters[2] = fx;
    ifo.parameters[3] = fy;
    ifo.overrideMaterial = materialManager.getForName(materialName);

    renderBackend.drawModel(quad,klIdentityMatrix(),&ifo);     
}

void FlushCache_Con(const char *args) {
    pageCache->flush();
}

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow) {

    // Init klubnika library
    klInit();

    // Create opengl window
    VListener listener;
    RenderWindow::Open(&listener, "Test Window" );

    console.registerCommand("vt_flush",FlushCache_Con);
    klConVariable *vt_showPhysical  = console.getVariable("vt_showPhysical", "0");
    klConVariable *vt_showTranslate = console.getVariable("vt_showTranslate", "0");
    klConVariable *vt_showPageId    = console.getVariable("vt_showPageId", "0");

    klPrint(
        "Some interesting commands\n"
        "=========================\n\n"
        "vt_showPhysical             Show the physical pages texture\n"
        "vt_showTranslate            Show the page translation texture\n"
        );

    // Main loop
    while ( RenderWindow::Update() ) {
        fpsCounter.frame();

        // Draw our frame info (draws MS)
        renderBackend.drawString(2,10,fpsCounter.getString(),0xFFFFFFFF);

        unsigned int frameTime = GetTickCount();
        double dt = (frameTime-lastFrameTime)*0.001;
        time += dt;
        listener.animateCamera((float)dt);

        glEnable(GL_DEPTH_TEST);

        //  Camera set up
        klCamera mainCam;
        mainCam.setNear(20.0f);
        mainCam.setFar(10000.0f);
        //mainCam.setFov((zoom) ? 30.0f : 80.0f);
        mainCam.setFov(80.0f);
        mainCam.setOrigin(currentCamInfo.cameraPosition);
        mainCam.setDirection(currentCamInfo.cameraDirection,klVec3(0.0,0.0,1.0));
        mainCam.setAspect(listener.aspect);

        /////////////////////////////////
        /// Rendering starts here
        /////////////////////////////////

        glClearColor(0.79f,0.9f,1.0f,0.0f);
        renderBackend.startFrame(time);

        // Draw the scene to the G-Buffer
        renderBackend.setDestination(deferredRenderTarget,GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT,
                                        klRenderTarget::BUFFER0 | klRenderTarget::BUFFER1);

        renderBackend.startView(mainCam);
        renderBackend.drawModel(worldModel,klIdentityMatrix());

        // Force the back-end to send everything to the glDriver
        renderBackend.flush();

        // Capture the pageId buffer for the resolver
        pageResolver->captureBuffer();

        // Do deferred lighting or whatever...

        // Render some transparent geometry...

        // Render to the window-system back buffer
        renderBackend.setDestination(&systemRenderTarget,GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        
        // Full screen quad post processing...
        klInstanceInfo ifo;
        if ( vt_showPageId->getValueInt() ) {
            ifo.overrideMaterial = postProcessMaterialPageId;
        } else {
            ifo.overrideMaterial = postProcessMaterial;
        }
        renderBackend.drawModel(quad,klIdentityMatrix(),&ifo);

        // Render some debug gui stuff on top
        if ( vt_showPhysical->getValueInt() ) {
            DrawGuiElement(0,0,512,512,listener.width,listener.height,"show_texture_cache");
        }
        if ( vt_showTranslate->getValueInt() ) {
            DrawGuiElement(0,0,512,512,listener.width,listener.height,"show_page_table");
        }

        // and finish it off
        renderBackend.finishFrame();
        //RenderWindow::SwapBuffers();

        /////////////////////////////////
        /// Now do all the cuda stuff to update our cache
        /////////////////////////////////

        // First find what tiles are used by this frame
        // and request them from the cache, this will put us in CUDA context
        pageResolver->resolve();

        // Send any newly arrived tiles to the cache (this starts of in CUDA mode then switches to GL again)
        // Note: This will leave the cache in an invalid state untill frameSync is called on the cache...
#if 1
        pageProvider->frameSync();
#else
        DiskPageProvider *p = (DiskPageProvider *)pageProvider;
        p->benchmarkUpload();
#endif

        // Allow the cache to update it's internal data structures based on newly arrived tiles (uses GL only)
        pageCache->frameSync();


        lastFrameTime = frameTime;
        RenderWindow::SwapBuffers();
    }

    // Free the window
    RenderWindow::Close();

    // Close log
    klShutDown();

    return 0;
}
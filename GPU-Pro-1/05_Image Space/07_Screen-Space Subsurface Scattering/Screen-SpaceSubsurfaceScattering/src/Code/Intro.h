#ifndef INTRO_H
#define INTRO_H

#include "Camera.h"
#include "SplashScreen.h"

class Intro {
    public:
        Intro(Camera &camera, Camera &light1, Camera &light2);

        void render(float t, bool updateCamera);
        void cameraReleased(float t) { tReleased = t; }

        bool hasFinished(float t) const { return t > 57.0f; } 

    private:
        void bounceCamera(float t, bool updateCamera, D3DXVECTOR2 &target);

        Camera &camera, &light1, &light2;

        float tReleased;
};

#endif

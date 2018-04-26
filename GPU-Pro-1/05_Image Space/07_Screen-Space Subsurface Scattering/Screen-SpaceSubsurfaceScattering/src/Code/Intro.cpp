/*
 * Copyright (C) 2009 Jorge Jimenez (jim@unizar.es)
 * Copyright (C) 2009 Diego Gutierrez (diegog@unizar.es)
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without 
 * modification, are permitted provided that the following conditions are met:
 * 
 * 1. Redistributions of source code must retain the above copyright notice, 
 *    this list of conditions and the following disclaimer.
 * 
 * 2. Redistributions in binary form must display the names 'Jorge Jimenez'
 *    and 'Diego Gutierrez' as 'Real-Time Rendering R&D' in the credits of the
 *    application, if such credits exist. The authors of this work must be
 *    notified via email (jim@unizar.es) in this case of redistribution.
 * 
 * 3. Neither the name of copyright holders nor the names of its contributors 
 *    may be used to endorse or promote products derived from this software 
 *    without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS 
 * IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, 
 * THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR 
 * PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL COPYRIGHT HOLDERS OR CONTRIBUTORS 
 * BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include <DXUT.h>
#include <string>
#include "Intro.h"
#include "Fade.h"
#include "Animation.h"
using namespace std;


Intro::Intro(Camera &camera, Camera &light1, Camera &light2)
    : camera(camera), light1(light1), light2(light2), tReleased(-6.0f) {
}


void Intro::render(float t, bool updateCamera) {
    if (t < 11.0f) {
        bounceCamera(t, updateCamera, Animation::linear(t, 0.0f, 11.0f, D3DXVECTOR2(0.0f, 1.1f), D3DXVECTOR2(0.0f, 0.0f)));
        
        light1.setDistance(1.35f);
        light1.setAngle(Animation::push(t, 0.0f, D3DXVECTOR2(-4.805f, -0.056f), D3DXVECTOR2(-0.1981f, -0.0925f)));

        light2.setDistance(1.45f);
        light2.setAngle(D3DXVECTOR2(2.011f, -0.245f));

        Fade::render(t, 0.0f, 9.0f, 2.0f, 2.0f);
    } else if (t < 26.0f) {
        light1.setDistance(1.8f);
        light1.setAngle(Animation::push(t, 11.0f, D3DXVECTOR2(-0.7f, -0.359f), D3DXVECTOR2(-80.0f, 1.0f) / 138.0f));
        
        light2.setDistance(2.24f);
        light2.setAngle(Animation::push(t, 11.0f, D3DXVECTOR2(-0.350f, -0.813f), D3DXVECTOR2(-60.0f, 1.0f) / 138.0f));

        Fade::render(t, 13.0f, 24.0f, 2.0f, 2.0f);
    } else {
        bounceCamera(t, updateCamera, D3DXVECTOR2(Animation::linear(t, 27.0f, 57.0f, D3DXVECTOR2(-1.974f, -0.367f), D3DXVECTOR2(1.456f, 0.079f))));
        
        light1.setAngle(Animation::push(t, 11.0f, D3DXVECTOR2(-0.7f, -0.359f), D3DXVECTOR2(-80.0f, 1.0f) / 138.0f));
        light2.setAngle(Animation::push(t, 11.0f, D3DXVECTOR2(-0.350f, -0.813f), D3DXVECTOR2(-60.0f, 1.0f) / 138.0f));

        Fade::render(t, 27.0f, 54.0f, 2.0f, 3.0f);
    }
}


void Intro::bounceCamera(float t, bool updateCamera, D3DXVECTOR2 &target) {
    if (updateCamera) {
        camera.setAngle(Animation::smooth(t, tReleased, tReleased + 6.0f, camera.getAngle(), target));
    }
}

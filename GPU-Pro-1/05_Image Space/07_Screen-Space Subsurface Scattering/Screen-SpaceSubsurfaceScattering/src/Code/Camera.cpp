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
#include "Camera.h"


LRESULT Camera::handleMessages(HWND hwnd, UINT msg, WPARAM wparam, LPARAM lparam) {
    switch(msg) {
        case WM_LBUTTONDOWN: {
            POINT point;
            GetCursorPos(&point);
            mousePos = D3DXVECTOR2((float) point.x, (float) point.y);
            draggingLeft = true;
            attenuation = 4.0f;
            SetCapture(hwnd);
            return true;
        }
        case WM_LBUTTONUP:
            draggingLeft = false;
            if (wparam & MK_CONTROL) {
                attenuation = 0.0f;
            } else {
                attenuation = 4.0f;
            }
            ReleaseCapture();
            return true;
        case WM_RBUTTONDOWN: {
            POINT point;
            GetCursorPos(&point);
            mousePos = D3DXVECTOR2((float) point.x, (float) point.y);
            draggingRight = true;
            SetCapture(hwnd);
            return true;
        }
        case WM_RBUTTONUP: {
            draggingRight = false;
            ReleaseCapture();
            return true;
        }
        case WM_MOUSEMOVE: {
            POINT point;
            GetCursorPos(&point);
            D3DXVECTOR2 newMousePos = D3DXVECTOR2((float) point.x, (float) point.y);
            if (draggingLeft) {
                D3DXVECTOR2 delta = newMousePos - mousePos;
                angleVelocity -= delta;
                mousePos = newMousePos;
            }
            if (draggingRight) {
                distance += (newMousePos.y - mousePos.y) / 75.0f;
                mousePos = newMousePos;
            }
            return true;
        }
        case WM_CAPTURECHANGED: {
            if ((HWND) lparam != hwnd) {
                draggingLeft = false;
                draggingRight = false;
            }
            break;
        }
    }
    return 0;
}


void Camera::frameMove(FLOAT elapsedTime) {
    angle += angleVelocity * elapsedTime / 150.0f;
    angleVelocity = angleVelocity / (1.0f + attenuation * elapsedTime);
    changed = true;
}


void Camera::setProjection(float fov, float aspect, float nearPlane, float farPlane) {
    D3DXMatrixPerspectiveFovLH(&projection, fov, aspect, nearPlane, farPlane);
}


void Camera::build() {
    if (changed) {
        D3DXMatrixTranslation(&view, 0.0f, 0.0f, distance);

        D3DXMATRIX t;
        D3DXMatrixRotationX(&t, angle.y);
        view = t * view;

        D3DXMatrixRotationY(&t, angle.x);
        view = t * view;

        D3DXMATRIX viewInverse;
        float det;
        D3DXMatrixInverse(&viewInverse, &det, &view);
        eyePosition = D3DXVECTOR3(viewInverse._41, viewInverse._42, viewInverse._43);

        changed = false;
    }
}

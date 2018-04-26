// Copyright (C) 2009 Jorge Jimenez (jim@unizar.es)

#ifndef CAMERA_H
#define CAMERA_H

class Camera {
    public:
        Camera() :
            distance(0.0f),
            angle(0.0f, 0.0f),
            angleVelocity(0.0f, 0.0f),
            changed(true),
            draggingLeft(false),
            draggingRight(false) {}

        LRESULT handleMessages(HWND hwnd, UINT msg, WPARAM wparam, LPARAM lparam);

        void frameMove(FLOAT elapsedTime);

        void setDistance(float distance) { this->distance = distance; changed = true; }
        float getDistance() const { return distance; }

        void setAngle(const D3DXVECTOR2 &angle) { this->angle = angle; changed = true; }
        const D3DXVECTOR2 &getAngle() const { return angle; }

        void setAngleVelocity(const D3DXVECTOR2 &angleVelocity) { this->angleVelocity = angleVelocity; }
        const D3DXVECTOR2 &getAngleVelocity() const { return angleVelocity; }

        void setProjection(float fov, float aspect, float nearPlane, float farPlane);

        const D3DXMATRIX &getViewMatrix() { build(); return view; }
        const D3DXMATRIX &getProjectionMatrix() const { return projection; }

        const D3DXVECTOR3 &getEyePosition() { build(); return eyePosition; } 

        D3DXVECTOR3 getLookAtPosition() const { return D3DXVECTOR3(0.0f, 0.0f, 0.0f); } 

    private:
        void build();

        float distance;
        D3DXVECTOR2 angle, angleVelocity;
        bool changed;

        D3DXMATRIX view, projection;
        D3DXVECTOR3 eyePosition;

        D3DXVECTOR2 mousePos;
        float attenuation;
        bool draggingLeft, draggingRight;
};

#endif

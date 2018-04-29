// Copyright 2010 Intel Corporation
// All Rights Reserved
//
// Permission is granted to use, copy, distribute and prepare derivative works of this
// software for any purpose and without fee, provided, that the above copyright notice
// and this statement appear in all copies.  Intel makes no representations about the
// suitability of this software for any purpose.  THIS SOFTWARE IS PROVIDED "AS IS."
// INTEL SPECIFICALLY DISCLAIMS ALL WARRANTIES, EXPRESS OR IMPLIED, AND ALL LIABILITY,
// INCLUDING CONSEQUENTIAL AND OTHER INDIRECT DAMAGES, FOR THE USE OF THIS SOFTWARE,
// INCLUDING LIABILITY FOR INFRINGEMENT OF ANY PROPRIETARY RIGHTS, AND INCLUDING THE
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.  Intel does not
// assume any responsibility for any errors which may appear in this software nor any
// responsibility to update it.

#ifndef _CAMERA_H
#define _CAMERA_H

extern float gMoveRate;
extern float gRotateRate;

class Camera
{
    friend class Effect;
    friend class Frustum;

protected:
    float mFOV;
    float mAspectRatio;
    float mNearClipDistance;
    float mFarClipDistance;
    float mYaw;
    float mPitch;

    D3DXMATRIX mMatProjection;
    D3DXMATRIX mMatView;
    D3DXMATRIX mMatViewProjection;

    D3DXVECTOR3 mPosition;
    D3DXVECTOR3 mLook;
    D3DXVECTOR3 mUp;
    D3DXVECTOR3 mRight;

    D3DXMATRIX  mShadowViewProjection;
    D3DXVECTOR3 mShadowCameraPosition;
    D3DXVECTOR3 mPointOnShadowPlane;
    D3DXVECTOR3 mpShadowPlaneNormals[4];

public:
    Camera(
        float fov,
        float aspectRatio,
        float nearClipDistance,
        float farClipDistance,
        D3DXVECTOR3 &position,
        D3DXVECTOR3 &look,
        D3DXVECTOR3 &up
    );
    ~Camera() {}

    void SetPositionAndOrientation(
        float fov,
        float aspectRatio,
        float nearClipDistance,
        float farClipDistance,
        const D3DXVECTOR3 &position,
        const D3DXVECTOR3 &look,
        const D3DXVECTOR3 &up
    );

    void Update( float deltaSeconds );
    void UpdateShadowViewProjection();

    virtual const D3DXMATRIX *GetViewProjectionMatrix() { return &mMatViewProjection; }
    virtual const D3DXVECTOR3 GetPosition() { return mPosition; }
    const D3DXVECTOR3 GetLook() { return mLook; }
    const D3DXVECTOR3 GetUp() { return mUp; }
    const D3DXVECTOR3 GetRight() { return mRight; }
    const D3DXMATRIX *GetShadowViewProjection() { return &mShadowViewProjection; } // { return &mMatViewProjection; }// 
    const D3DXVECTOR3 GetPointOnShadowPlane() { return mPointOnShadowPlane; }
    const D3DXVECTOR3 *GetShadowPlaneNormals() { return mpShadowPlaneNormals; }

    void MoveForward( float deltaSeconds ){ mPosition += mLook * deltaSeconds * gMoveRate; }
    void MoveBack( float deltaSeconds )   { mPosition -= mLook * deltaSeconds * gMoveRate; }
    void MoveUp( float deltaSeconds )     { mPosition +=   mUp * deltaSeconds * gMoveRate; }
    void MoveDown( float deltaSeconds )   { mPosition -=   mUp * deltaSeconds * gMoveRate; }

    void TurnLeft( float deltaSeconds ) { mYaw   += gRotateRate * deltaSeconds;}
    void TurnRight( float deltaSeconds ){ mYaw   -= gRotateRate * deltaSeconds;}
    void TurnUp( float deltaSeconds )   { mPitch += gRotateRate * deltaSeconds;}
    void TurnDown( float deltaSeconds ) { mPitch -= gRotateRate * deltaSeconds;}

    void MoveRight( float deltaSeconds ){
        D3DXVECTOR3 right;
        D3DXVec3Cross( &right, &mUp, &mLook );
        mPosition += right * deltaSeconds * gMoveRate;
    }
    void MoveLeft( float deltaSeconds ){
        D3DXVECTOR3 right;
        D3DXVec3Cross( &right, &mUp, &mLook );
        mPosition -= right * deltaSeconds * gMoveRate;
    }

};

//-----------------------------------------------
class ShadowCamera : public Camera
{
public:
    ShadowCamera(
        float fov,
        float aspectRatio,
        float nearClipDistance,
        float farClipDistance,
        D3DXVECTOR3 &position,
        D3DXVECTOR3 &look,
        D3DXVECTOR3 &up
        ) : Camera(fov, aspectRatio, nearClipDistance, farClipDistance, position, look, up )
    {
    }
    ~ShadowCamera() {}

    virtual const D3DXMATRIX *GetViewProjectionMatrix() { return GetShadowViewProjection(); }
    virtual const D3DXVECTOR3 GetPosition() { return mShadowCameraPosition; }
};

#endif // _CAMERA_H

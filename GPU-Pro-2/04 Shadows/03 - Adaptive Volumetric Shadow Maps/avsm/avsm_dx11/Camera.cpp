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

#include "DXUT.h"
#include "Camera.h"

float gMoveRate = 200.0f;
float gRotateRate = 1.0f;
//-----------------------------------------------
Camera::Camera(
    float fov,
    float aspectRatio,
    float nearClipDistance,
    float farClipDistance,
    D3DXVECTOR3 &position,
    D3DXVECTOR3 &look,
    D3DXVECTOR3 &up
)
{
    SetPositionAndOrientation(
        fov,
        aspectRatio,
        nearClipDistance,
        farClipDistance,
        position,
        look,
        up
    );
}

//-----------------------------------------------
void Camera::SetPositionAndOrientation(
    float fov,
    float aspectRatio,
    float nearClipDistance,
    float farClipDistance,
    const D3DXVECTOR3 &position,
    const D3DXVECTOR3 &look,
    const D3DXVECTOR3 &up
)
{
    mFOV = fov;
    mAspectRatio = aspectRatio;
    mNearClipDistance = nearClipDistance;
    mFarClipDistance = farClipDistance;
    mPosition = position;
    mPitch = mYaw = 0.0f;

	// Setup Frustrum
	D3DXMatrixPerspectiveFovLH
    (
        &mMatProjection,
        mFOV,
        mAspectRatio,
        mNearClipDistance,
        mFarClipDistance
    );

    mLook = look;
    mUp = up;
    D3DXVec3Cross( &mRight, &mLook, &mUp );
    D3DXVECTOR3 lookPoint = mPosition + mLook;

    D3DXMatrixLookAtLH
    (
        &mMatView,
        &mPosition,
        &lookPoint,
        &mUp
    );

    mMatViewProjection = mMatView * mMatProjection;

    UpdateShadowViewProjection();
}

//-----------------------------------------------
void Camera::Update(float deltaSeconds)
{
    D3DXMATRIX rotate;
    D3DXMatrixRotationYawPitchRoll( &rotate, mYaw, -mPitch, 0.0f );
    mLook = D3DXVECTOR3( rotate._31, rotate._33, rotate._32);
    D3DXVec3Normalize( &mLook, &mLook );

    D3DXVECTOR3 immediate(0.0f, 0.0f, 1.0f);
    D3DXVec3Cross( &mRight, &mLook, &immediate);
    D3DXVec3Normalize( &mRight, &mRight );

    D3DXVec3Cross( &mUp, &mRight, &mLook );

    D3DXVECTOR3 lookPoint = mPosition + mLook;
	D3DXMatrixLookAtLH
    (
        &mMatView,
        &mPosition,
        &lookPoint,
        &mUp
    );
 	D3DXMatrixPerspectiveFovLH
    (
        &mMatProjection,
        mFOV,
        mAspectRatio,
        mNearClipDistance,
        mFarClipDistance
    );

    mMatViewProjection = mMatView * mMatProjection;

    // Update shadow matrices
    UpdateShadowViewProjection();
}

// Move the shadow camera back to narrow the FOV
// = shift pixel density toward the far clip plane
static float gShift = 150.0;

//-----------------------------------------------
void Camera::UpdateShadowViewProjection()
{
    mShadowCameraPosition = mPosition - mLook * gShift;

    // The clip planes stay where they were.  That means they shift
    // relative to the new posiiton.  Note, the near-clip plane gets
    // bigger, but the far-clip plane satys the same size.
    float newNearClipDistance = mNearClipDistance + gShift;
    float newFarClipDistance  = mFarClipDistance  + gShift;

    // Calculate the new FOV
    float tanHalfOriginalFov    = tan(0.5f*mFOV);
    float halfOriginalFarHeight = mFarClipDistance * tanHalfOriginalFov;
    float tanHalfFov            = halfOriginalFarHeight/newFarClipDistance;
    float newFov                = 2.0f * atan(tanHalfFov);

    // *******************************
    // Compute values from the new camera's frustum
    // *******************************
    // Compute the position of the center of the near and far clip planes.
    D3DXVECTOR3 nearCenter = mShadowCameraPosition + newNearClipDistance * mLook;
    D3DXVECTOR3 farCenter  = mShadowCameraPosition + newFarClipDistance  * mLook;

    // Compute the width and height of the near and far clip planes
    float halfNearHeight = newNearClipDistance * tanHalfFov;
    float halfNearWidth  = halfNearHeight      * mAspectRatio;
    float halfFarHeight  = newFarClipDistance  * tanHalfFov;
    float halfFarWidth   = halfFarHeight       * mAspectRatio;

    // Create two vectors each for the near and far clip planes.
    // These are the scaled up and right vectors.
    D3DXVECTOR3 nearUp    = halfNearHeight * mUp;
    D3DXVECTOR3 nearRight = halfNearWidth  * mRight;
    D3DXVECTOR3 farUp     = halfFarHeight  * mUp;
    D3DXVECTOR3 farRight  = halfFarWidth   * mRight;

    // *******************************
    // Compute values for new frustum
    // This transforms one of the original camera's frustum planes to a square
    // by positioning a new camera such that it's position, direction and FOV
    // achieve the desired projection.
    // *******************************
    // Use the original camera's position as the point on our planes.
    mPointOnShadowPlane = mPosition;


    // Calculate the normal for each of the four shadow planes.

    D3DXVECTOR3 nearRightCenter  = nearCenter + nearRight;
    D3DXVECTOR3 nearLeftCenter   = nearCenter - nearRight;
    D3DXVECTOR3 nearTopCenter    = nearCenter + nearUp;
    D3DXVECTOR3 nearBottomCenter = nearCenter - nearUp;

    D3DXVECTOR3 farRightCenter   = farCenter  + farRight;
    D3DXVECTOR3 farLeftCenter    = farCenter  - farRight;
    D3DXVECTOR3 farTopCenter     = farCenter  + farUp;
    D3DXVECTOR3 farBottomCenter  = farCenter  - farUp;

    // Compute the shadow plane's normal.
    D3DXVECTOR3 shadowPlaneForward;

    shadowPlaneForward = farTopCenter - nearBottomCenter;
    D3DXVec3Normalize( &shadowPlaneForward, &shadowPlaneForward );
    D3DXVec3Cross( &mpShadowPlaneNormals[0], &shadowPlaneForward, &mRight );

    shadowPlaneForward = farRightCenter - nearLeftCenter;
    D3DXVec3Normalize( &shadowPlaneForward, &shadowPlaneForward );
    D3DXVec3Cross( &mpShadowPlaneNormals[1], &mUp, &shadowPlaneForward );

    shadowPlaneForward = farBottomCenter - nearTopCenter;
    D3DXVec3Normalize( &shadowPlaneForward, &shadowPlaneForward );
    D3DXVec3Cross( &mpShadowPlaneNormals[2], &mRight, &shadowPlaneForward );

    shadowPlaneForward = farLeftCenter - nearRightCenter;
    D3DXVec3Normalize( &shadowPlaneForward, &shadowPlaneForward );
    D3DXVec3Cross( &mpShadowPlaneNormals[3], &shadowPlaneForward, &mUp );


    // Construct the new view matrix
    D3DXMATRIX newView;
    D3DXVECTOR3 lookPoint = mShadowCameraPosition + mLook;
    D3DXMatrixLookAtLH
    (
        &newView,
        &mShadowCameraPosition,
        &lookPoint,
        &mUp
    );

    D3DXMATRIX newProjection;
 	D3DXMatrixPerspectiveFovLH
    (
        &newProjection,
        newFov,
        mAspectRatio,
        newNearClipDistance,
        newFarClipDistance
    );

    mShadowViewProjection = newView * newProjection;
}


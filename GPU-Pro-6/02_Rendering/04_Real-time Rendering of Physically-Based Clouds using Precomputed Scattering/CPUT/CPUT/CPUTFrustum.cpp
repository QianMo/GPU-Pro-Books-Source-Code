//--------------------------------------------------------------------------------------
// Copyright 2012 Intel Corporation
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
//--------------------------------------------------------------------------------------
#include "CPUTFrustum.h"
#include "CPUTCamera.h"

//-----------------------------------------------
void CPUTFrustum::InitializeFrustum( CPUTCamera *pCamera )
{
    InitializeFrustum(
        pCamera->GetNearPlaneDistance(),
        pCamera->GetFarPlaneDistance(),
        pCamera->GetAspectRatio(),
        pCamera->GetFov(),
        pCamera->GetPosition(),
        pCamera->GetLook(),
        pCamera->GetUp()
    );
}

//-----------------------------------------------
void CPUTFrustum::InitializeFrustum
(
    float nearClipDistance,
    float farClipDistance,
    float aspectRatio,
    float fov,
    const float3 &position,
    const float3 &look,
    const float3 &up
)
{
    // ******************************
    // This function computes the position of each of the frustum's eight points.
    // It also computes the normal of each of the frustum's six planes.
    // ******************************

    mNumFrustumVisibleModels = 0;
    mNumFrustumCulledModels  = 0;

    // We have the camera's up and look, but we also need right.
    float3 right = cross3( up, look );

    // Compute the position of the center of the near and far clip planes.
    float3 nearCenter = position + look * nearClipDistance;
    float3 farCenter  = position + look * farClipDistance;

    // Compute the width and height of the near and far clip planes
    float tanHalfFov = tanf(0.5f*fov);
    float halfNearHeight = nearClipDistance * tanHalfFov;
    float halfNearWidth  = halfNearHeight * aspectRatio;

    float halfFarHeight  = farClipDistance * tanHalfFov;
    float halfFarWidth   = halfFarHeight  * aspectRatio;

    // Create two vectors each for the near and far clip planes.
    // These are the scaled up and right vectors.
    float3 upNear      = up    * halfNearHeight;
    float3 rightNear   = right * halfNearWidth;
    float3 upFar       = up    * halfFarHeight;
    float3 rightFar    = right * halfFarWidth;

    // Use the center positions and the up and right vectors
    // to compute the positions for the near and far clip plane vertices (four each)
    mpPosition[0] = nearCenter + upNear - rightNear; // near top left
    mpPosition[1] = nearCenter + upNear + rightNear; // near top right
    mpPosition[2] = nearCenter - upNear + rightNear; // near bottom right
    mpPosition[3] = nearCenter - upNear - rightNear; // near bottom left
    mpPosition[4] = farCenter  + upFar  - rightFar;  // far top left
    mpPosition[5] = farCenter  + upFar  + rightFar;  // far top right
    mpPosition[6] = farCenter  - upFar  + rightFar;  // far bottom right
    mpPosition[7] = farCenter  - upFar  - rightFar;  // far bottom left

    // Compute some of the frustum's edge vectors.  We will cross these
    // to get the normals for each of the six planes.
    float3 nearTop     = mpPosition[1] - mpPosition[0];
    float3 nearLeft    = mpPosition[3] - mpPosition[0];
    float3 topLeft     = mpPosition[4] - mpPosition[0];
    float3 bottomRight = mpPosition[2] - mpPosition[6];
    float3 farRight    = mpPosition[5] - mpPosition[6];
    float3 farBottom   = mpPosition[7] - mpPosition[6];

    mpNormal[0] = cross3(nearTop,     nearLeft).normalize();    // near clip plane
    mpNormal[1] = cross3(nearLeft,    topLeft).normalize();     // left
    mpNormal[2] = cross3(topLeft,     nearTop).normalize();     // top
    mpNormal[3] = cross3(farBottom,   bottomRight).normalize(); // bottom
    mpNormal[4] = cross3(bottomRight, farRight).normalize();    // right
    mpNormal[5] = cross3(farRight,    farBottom).normalize();   // far clip plane
}

//-----------------------------------------------
bool CPUTFrustum::IsVisible(
    const float3 &center,
    const float3 &half
){
    // TODO:  There are MUCH more efficient ways to do this.
    float3 pBoundingBoxPosition[8];
    pBoundingBoxPosition[0] = center + float3(  half.x,  half.y,  half.z );
    pBoundingBoxPosition[1] = center + float3(  half.x,  half.y, -half.z );
    pBoundingBoxPosition[2] = center + float3(  half.x, -half.y,  half.z );
    pBoundingBoxPosition[3] = center + float3(  half.x, -half.y, -half.z );
    pBoundingBoxPosition[4] = center + float3( -half.x,  half.y,  half.z );
    pBoundingBoxPosition[5] = center + float3( -half.x,  half.y, -half.z );
    pBoundingBoxPosition[6] = center + float3( -half.x, -half.y,  half.z );
    pBoundingBoxPosition[7] = center + float3( -half.x, -half.y, -half.z );

    // Test each bounding box point against each of the six frustum planes.
    // Note: we need a point on the plane to compute the distance to the plane.
    // We only need two of our frustum's points to do this.  A corner vertex is on
    // three of the six planes.  We need two of these corners to have a point
    // on all six planes.
    UINT pPointIndex[6] = {0,0,0,6,6,6};
    UINT ii;
    for( ii=0; ii<6; ii++ )
    {
        bool allEightPointsOutsidePlane = true;
        float3 *pNormal = &mpNormal[ii];
        float3 *pPlanePoint = &mpPosition[pPointIndex[ii]];
        float3 planeToPoint;
        float distanceToPlane;
        UINT jj;
        for( jj=0; jj<8; jj++ )
        {
            planeToPoint = pBoundingBoxPosition[jj] - *pPlanePoint;
            distanceToPlane = dot3( *pNormal, planeToPoint );
            if( distanceToPlane < 0.0f )
            {
                allEightPointsOutsidePlane = false;
                break; // from for.  No point testing any more points against this plane.
            }
        }
        if( allEightPointsOutsidePlane )
        {
            mNumFrustumCulledModels++;
            return false;
        }
    }

    // Tested all eight points against all six planes and none of the planes
    // had all eight points outside.
    mNumFrustumVisibleModels++;
    return true;
}


#pragma once

#include "../Core/Rendering/Scene/Camera.h"
#include "AABBox.h"

namespace Utils
{
    inline void LookAt( const Vec3& pos, const Vec3& dir, Camera& camera )
    {
        camera.SetViewMatrix( Mat4x4::LookAtD3D( pos, pos + dir, c_YAxis ) );
    }

    inline void AlignBBox( CAABBox& BBox, float alignment )
    {
        Vec3 boxMin = BBox.m_min / alignment; boxMin = Vec3( floorf( boxMin.x ) * alignment, floorf( boxMin.y ) * alignment, 0.0f );
        Vec3 boxMax = BBox.m_max / alignment; boxMax = Vec3( ceilf( boxMax.x ) * alignment, ceilf( boxMax.y ) * alignment, 0.0f );
        BBox = CAABBox( boxMin, boxMax );
    }

    inline float CalcDepthBias(
        const Mat4x4& orthoProjMat,
        const Vec3& kernelSize,
        int viewportWidth,
        int viewportHeight,
        int depthBitsPerPixel )
    {
        Vec3 texelSizeWS(
            fabsf( 2.0f / ( orthoProjMat.e11 * float( viewportWidth ) ) ),
            fabsf( 2.0f / ( orthoProjMat.e22 * float( viewportHeight ) ) ),
            fabsf( 1.0f / ( orthoProjMat.e33 * float( 1 << depthBitsPerPixel ) ) ) );
        Vec3 kernelSizeWS = texelSizeWS * kernelSize;
        float kernelSizeMax = std::max( std::max( kernelSizeWS.x, kernelSizeWS.y ), kernelSizeWS.z );
        return kernelSizeMax * fabsf( orthoProjMat.e33 );
    }
}

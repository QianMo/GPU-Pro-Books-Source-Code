#pragma once

#include "AABBox.h"
#include "../Core/Rendering/Scene/Camera.h"

template< unsigned int maxVertices = 5 >
class CConvexHull2D
{
public:
    static const unsigned int MAX_VERTICES = maxVertices;
    Vec2 m_vertices[ maxVertices ];
    int m_size;

    CConvexHull2D()
    {
        Reset();
    }

    CConvexHull2D( int numVertices, const Vec2* pVertices )
    {
        m_size = FindConvexHull( numVertices, pVertices, m_vertices );
    }

    void Reset()
    {
        m_size = 0;
    }

    const Vec2 FindFrustumConvexHull( const Camera& frustum, float frustumZMaxOverride, const Mat4x4& viewProj )
    {
        static const unsigned int numVertices = 5;
        Vec2 vertices[ numVertices ] = { Vec3::Project( frustum.GetPosition(), viewProj ) };

        float hz = Vec3::Project( Vec3( 0, 0, frustumZMaxOverride ), frustum.GetProjection() ).z;

        const Vec3 frustumCorners[] = 
        {
            Vec3(-1.0f, -1.0f, hz ),
            Vec3(+1.0f, -1.0f, hz ),
            Vec3(+1.0f, +1.0f, hz ),
            Vec3(-1.0f, +1.0f, hz ),
        };

        Mat4x4 tm = frustum.GetViewProjectionInverse() * viewProj;
        for( unsigned int i = 1; i < numVertices; ++i )
            vertices[ i ] = Vec3::Project( frustumCorners[ i - 1 ], tm );

        m_size = FindConvexHull( numVertices, vertices, m_vertices );
        _ASSERT( m_size > 0 );

        return vertices[0];
    }

    bool Intersects( const CAABBox& BBox ) const
    {
        if( m_size == 0 ) return false;

        static const Vec2 normals[] =
        {
            Vec2( 1, 0),
            Vec2( 0, 1),
            Vec2(-1, 0),
            Vec2( 0,-1),
        };

        Vec2 vb1[ maxVertices * 2 ];
        Vec2 vb2[ maxVertices * 2 ];

        const Vec2* v = m_vertices;
        int n = m_size;

        int j, index[2];
        float d[2];
        for( int i = 0; i < 4; ++i )
        {
            float pw = -Vec2::Dot( normals[i], i < 2 ? BBox.m_min : BBox.m_max );
            index[1] = n - 1;
            d[1] = Vec2::Dot( normals[i], v[ index[ 1 ] ] ) + pw;
            for( j = 0; j < n; j++ )
            {
                index[0] = index[1];
                index[1] = j;
                d[0] = d[1];
                d[1] = Vec2::Dot( normals[i], v[ index[1] ] ) + pw;
                if( d[1] > 0 && d[0] < 0 ) break;
            }
            if( j < n )
            {
                int k = 0;
                Vec2* tmp = v == vb1 ? vb2 : vb1;
                tmp[ k++ ] = Vec2::Lerp( v[ index[1] ], v[ index[0] ], d[1] / ( d[1] - d[0] ) );
                do
                {
                    index[0] = index[1];
                    index[1] = ( index[1] + 1 ) % n;
                    d[0] = d[1];
                    d[1] = Vec2::Dot( normals[i], v[ index[1] ] ) + pw;
                    tmp[ k++ ] = v[ index[0] ];
                } while( d[1] > 0 );
                tmp[ k++ ] = Vec2::Lerp( v[ index[1] ], v[ index[0] ], d[1] / (d[1] - d[0] ) );
                n = k;
                v = tmp;
            }
            else
            {
                if( d[1] < 0 ) return false;
            }
        }
        return n > 0;
    }

    static int FindConvexHull( int numVertices, const Vec2* pVertices, Vec2* pHull )
    {
        _ASSERT( numVertices <= maxVertices );
        const float eps = 1e-5f;
        const float epsSq = eps * eps;
        int leftmostIndex = 0;
        for( int i = 1; i < numVertices; ++i )
        {
            float f = pVertices[ leftmostIndex ].x - pVertices[ i ].x;
            if( fabsf(f) < epsSq )
            {
                if( pVertices[ leftmostIndex ].y> pVertices[ i ].y )
                    leftmostIndex = i;
            }
            else if( f > 0 )
            {
                leftmostIndex = i;
            }
        }
        Vec2 dir0( 0, -1 );
        int hullSize = 0;
        int index0 = leftmostIndex;
        do
        {
            float maxCos = -FLT_MAX;
            int index1 = -1;
            Vec2 dir1;
            for( int j = 1; j < numVertices; ++j )
            {
                int k = ( index0 + j ) % numVertices;
                Vec2 v = pVertices[ k ] - pVertices[ index0 ];
                float l = Vec2::LengthSq( v );
                if( l > epsSq )
                {
                    Vec2 d = Vec2::Normalize( v );
                    float f = Vec2::Dot( d, dir0 );
                    if( maxCos < f )
                    {
                        maxCos = f;
                        index1 = k;
                        dir1 = d;
                    }
                }
            }
            if( index1 < 0 || hullSize >= numVertices )
            {
                //_ASSERT(!"epic fail");
                return 0;
            }
            pHull[ hullSize++ ] = pVertices[ index1 ];
            index0 = index1;
            dir0 = dir1;
        } while( Vec2::LengthSq( pVertices[ index0 ] - pVertices[ leftmostIndex ] ) > epsSq );
        return hullSize;
    }
};

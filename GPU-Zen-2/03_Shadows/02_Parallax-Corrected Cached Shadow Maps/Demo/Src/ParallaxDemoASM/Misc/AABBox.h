#pragma once

#include "../../Core/Math/Math.h"

class CAABBox : public MathLibObject
{
public:
    Vec3 m_min;
    Vec3 m_max;

    inline CAABBox() { Reset(); }
    inline CAABBox( const Vec3& boxMin, const Vec3& boxMax ) : m_min( boxMin ), m_max( boxMax ) { }

    void Reset()
    {
        m_min = Vec3( FLT_MAX);
        m_max = Vec3(-FLT_MAX);
    }

    bool IsInvalid() const
    {
        return !( m_min <= m_max );
    }

    void Add( const Vec3& v )
    {
        m_min = Vec3::Min( m_min, v );
        m_max = Vec3::Max( m_max, v );
    }

    void Add( const CAABBox& aabb )
    {
        m_min = Vec3::Min( m_min, aabb.m_min );
        m_max = Vec3::Max( m_max, aabb.m_max );
    }

    const Vec3 GetSize() const { return m_max - m_min; }
    const Vec3 GetCenter() const { return ( m_max + m_min ) * 0.5f; }

    float GetSizeX() const { return m_max.x - m_min.x; }
    float GetSizeY() const { return m_max.y - m_min.y; }
    float GetSizeZ() const { return m_max.z - m_min.z; }

    bool IsIntersectBox( const CAABBox& b ) const
    {
        if( ( m_min.x > b.m_max.x ) || ( b.m_min.x > m_max.x ) ) return false;
        if( ( m_min.y > b.m_max.y ) || ( b.m_min.y > m_max.y ) ) return false;
        if( ( m_min.z > b.m_max.z ) || ( b.m_min.z > m_max.z ) ) return false;
        return true;
    }

    bool ContainsPoint( const Vec3& point ) const
    {
        return ( point >= m_min ) & ( point <= m_max );
    }

    bool operator != ( const CAABBox& rhs ) const
    {
        return ( m_min != rhs.m_min ) | ( m_max != rhs.m_max );
    }

    bool operator == ( const CAABBox& rhs ) const
    {
        return ( m_min == rhs.m_min ) & ( m_max == rhs.m_max );
    }

    void TransformTo( const Mat4x4& tm, CAABBox& result ) const
    {
        Vec3 boxSize = m_max - m_min;

        Vec4 vx = tm.r[0] * Vec4::Swizzle<::x, ::x, ::x, ::x>( boxSize );
        Vec4 vy = tm.r[1] * Vec4::Swizzle<::y, ::y, ::y, ::y>( boxSize );
        Vec4 vz = tm.r[2] * Vec4::Swizzle<::z, ::z, ::z, ::z>( boxSize );

        Vec4 newMin = m_min * tm;
        Vec4 newMax = newMin;

        newMin += Vec3::Min( vx, Vec3::Zero() );
        newMax += Vec3::Max( vx, Vec3::Zero() );

        newMin += Vec3::Min( vy, Vec3::Zero() );
        newMax += Vec3::Max( vy, Vec3::Zero() );

        newMin += Vec3::Min( vz, Vec3::Zero() );
        newMax += Vec3::Max( vz, Vec3::Zero() );

        result = CAABBox( newMin, newMax );
    }
};

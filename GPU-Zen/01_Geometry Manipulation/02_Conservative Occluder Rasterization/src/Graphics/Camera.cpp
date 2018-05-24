#include "Camera.h"

#include <DirectXMathVector.inl>

namespace NGraphics
{
    CCamera::CCamera() :
        m_Position( DirectX::XMVectorSet( 0.0f, 0.0f, 0.0f, 1.0f ) ),
        m_Right( DirectX::XMVectorSet( 1.0f, 0.0f, 0.0f, 0.0f ) ),
        m_Up( DirectX::XMVectorSet( 0.0f, 1.0f, 0.0f, 0.0f ) ),
        m_Look( DirectX::XMVectorSet( 0.0f, 0.0f, 1.0f, 0.0f ) ),
        m_View( DirectX::XMMatrixIdentity() ),
        m_Projection( DirectX::XMMatrixIdentity() ),
        m_NearZ( 0.0f ),
        m_FarZ( 0.0f )
    {
    }

    void CCamera::Strafe( FLOAT distance )
    {
        m_Position = DirectX::XMVectorMultiplyAdd( DirectX::XMVectorReplicate( distance ), m_Right, m_Position );
    }

    void CCamera::Walk( FLOAT distance )
    {
        m_Position = DirectX::XMVectorMultiplyAdd( DirectX::XMVectorReplicate( distance ), m_Look, m_Position );
    }

    void CCamera::Pitch( FLOAT angle )
    {
        DirectX::XMVECTOR rotation = DirectX::XMQuaternionRotationAxis( m_Right, angle );
        m_Up = DirectX::XMVector3Rotate( m_Up, rotation );
        m_Look = DirectX::XMVector3Rotate( m_Look, rotation );
    }

    void CCamera::Yaw( FLOAT angle )
    {
        DirectX::XMVECTOR rotation = DirectX::XMQuaternionRotationAxis( DirectX::XMVectorSet( 0.0f, 1.0f, 0.0f, 0.0f ), angle );
        m_Right = DirectX::XMVector3Rotate( m_Right, rotation );
        m_Up = DirectX::XMVector3Rotate( m_Up, rotation );
        m_Look = DirectX::XMVector3Rotate( m_Look, rotation );
    }

    void CCamera::UpdateViewMatrix()
    {
        DirectX::XMVECTOR target = DirectX::XMVectorAdd( m_Position, m_Look );
        m_View = DirectX::XMMatrixLookAtRH( m_Position, target, m_Up );
    }

    void CCamera::ExtractFrustumPlanes( DirectX::XMFLOAT4 planes[ 6 ] )
    {
        DirectX::XMFLOAT4X4 view_projection;
        DirectX::XMStoreFloat4x4( &view_projection, GetViewProjection() );

        // Left
        planes[ 0 ].x = view_projection( 0, 3 ) + view_projection( 0, 0 );
        planes[ 0 ].y = view_projection( 1, 3 ) + view_projection( 1, 0 );
        planes[ 0 ].z = view_projection( 2, 3 ) + view_projection( 2, 0 );
        planes[ 0 ].w = view_projection( 3, 3 ) + view_projection( 3, 0 );

        // Right
        planes[ 1 ].x = view_projection( 0, 3 ) - view_projection( 0, 0 );
        planes[ 1 ].y = view_projection( 1, 3 ) - view_projection( 1, 0 );
        planes[ 1 ].z = view_projection( 2, 3 ) - view_projection( 2, 0 );
        planes[ 1 ].w = view_projection( 3, 3 ) - view_projection( 3, 0 );

        // Bottom
        planes[ 2 ].x = view_projection( 0, 3 ) + view_projection( 0, 1 );
        planes[ 2 ].y = view_projection( 1, 3 ) + view_projection( 1, 1 );
        planes[ 2 ].z = view_projection( 2, 3 ) + view_projection( 2, 1 );
        planes[ 2 ].w = view_projection( 3, 3 ) + view_projection( 3, 1 );

        // Top
        planes[ 3 ].x = view_projection( 0, 3 ) - view_projection( 0, 1 );
        planes[ 3 ].y = view_projection( 1, 3 ) - view_projection( 1, 1 );
        planes[ 3 ].z = view_projection( 2, 3 ) - view_projection( 2, 1 );
        planes[ 3 ].w = view_projection( 3, 3 ) - view_projection( 3, 1 );

        // Near
        planes[ 4 ].x = view_projection( 0, 2 );
        planes[ 4 ].y = view_projection( 1, 2 );
        planes[ 4 ].z = view_projection( 2, 2 );
        planes[ 4 ].w = view_projection( 3, 2 );

        // Far
        planes[ 5 ].x = view_projection( 0, 3 ) - view_projection( 0, 2 );
        planes[ 5 ].y = view_projection( 1, 3 ) - view_projection( 1, 2 );
        planes[ 5 ].z = view_projection( 2, 3 ) - view_projection( 2, 2 );
        planes[ 5 ].w = view_projection( 3, 3 ) - view_projection( 3, 2 );

        // Normalize the planes
        for ( UINT i = 0; i < 6; i++ )
        {
            DirectX::XMStoreFloat4( &planes[ i ], DirectX::XMPlaneNormalize( DirectX::XMLoadFloat4( &planes[ i ] ) ) );
        }
    }

    CCamera::SRay CCamera::ComputeRay( const Sint32* mouse_position )
    {
        float x = static_cast< float >( mouse_position[ 0 ] );
        float y = static_cast< float >( mouse_position[ 1 ] );

        float normalized_x = 2.0f * x / m_Width - 1.0f;
        float normalized_y = 1.0f - 2.0f * y / m_Height;

        DirectX::XMVECTOR direction = DirectX::XMVectorSet( normalized_x, normalized_y, 1.0f, 1.0f );
        
        DirectX::XMVECTOR dummy;
        DirectX::XMMATRIX inverse_view = DirectX::XMMatrixInverse( &dummy, m_View );
        DirectX::XMMATRIX inverse_projection = DirectX::XMMatrixInverse( &dummy, m_Projection );
        
        direction = DirectX::XMVector3TransformCoord( direction, inverse_projection );
        direction = DirectX::XMVector3TransformNormal( direction, inverse_view );
        direction = DirectX::XMVector3Normalize( direction );

        SRay ray;
        DirectX::XMStoreFloat3( &ray.m_Origin, m_Position );
        DirectX::XMStoreFloat3( &ray.m_Direction, direction );
        return ray;
    }

    void CCamera::SetPosition( DirectX::XMFLOAT3 position )
    {
        m_Position = XMLoadFloat3( &position );
    }

    void CCamera::SetPerspective( FLOAT fov_y, FLOAT width, FLOAT height, FLOAT near_z, FLOAT far_z )
    {
        m_FovY = fov_y;
        m_Width = width;
        m_Height = height;
        m_NearZ = near_z;
        m_FarZ = far_z;
        m_Projection = DirectX::XMMatrixPerspectiveFovRH( m_FovY, m_Width / m_Height, m_NearZ, m_FarZ );
        m_ProjectionFlippedZ = DirectX::XMMatrixPerspectiveFovRH( m_FovY, m_Width / m_Height, m_FarZ, m_NearZ );
    }

    const DirectX::XMMATRIX CCamera::GetView()
    {
        return m_View;
    }

    const DirectX::XMMATRIX CCamera::GetProjection()
    {
        return m_Projection;
    }

    const DirectX::XMMATRIX CCamera::GetProjectionFlippedZ()
    {
        return m_ProjectionFlippedZ;
    }

    const DirectX::XMMATRIX CCamera::GetViewProjection()
    {
        return DirectX::XMMatrixMultiply( m_View, m_Projection );
    }

    const DirectX::XMMATRIX CCamera::GetViewProjectionFlippedZ()
    {
        return DirectX::XMMatrixMultiply( m_View, m_ProjectionFlippedZ );
    }

    const DirectX::XMVECTOR CCamera::GetPosition()
    {
        return m_Position;
    }

    const DirectX::XMVECTOR CCamera::GetRight()
    {
        return m_Right;
    }

    const DirectX::XMVECTOR CCamera::GetUp()
    {
        return m_Up;
    }

    const DirectX::XMVECTOR CCamera::GetLook()
    {
        return m_Look;
    }

    const FLOAT CCamera::GetNearZ()
    {
        return m_NearZ;
    }

    const FLOAT CCamera::GetFarZ()
    {
        return m_FarZ;
    }
}
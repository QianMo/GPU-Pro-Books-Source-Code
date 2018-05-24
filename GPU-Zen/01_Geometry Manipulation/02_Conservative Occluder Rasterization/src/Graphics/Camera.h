#pragma once

#include <d3d12.h>
#include <DirectXMath.h>
#include <SDL.h>

namespace NGraphics
{
    class CCamera
    {
    public:
        struct SRay
        {
            DirectX::XMFLOAT3 m_Origin;
            DirectX::XMFLOAT3 m_Direction;
        };

    private:
        DirectX::XMVECTOR m_Position;
        DirectX::XMVECTOR m_Right;
        DirectX::XMVECTOR m_Up;
        DirectX::XMVECTOR m_Look;

        DirectX::XMMATRIX m_View;
        DirectX::XMMATRIX m_Projection;
        DirectX::XMMATRIX m_ProjectionFlippedZ;

        FLOAT m_FovY;
        FLOAT m_Width;
        FLOAT m_Height;
        FLOAT m_NearZ;
        FLOAT m_FarZ;

    public:
        CCamera();

        void Strafe( FLOAT distance );
        void Walk( FLOAT distance );
        void Pitch( FLOAT angle );
        void Yaw( FLOAT angle );

        void UpdateViewMatrix();

        void ExtractFrustumPlanes( DirectX::XMFLOAT4 planes[ 6 ] );

        SRay ComputeRay( const Sint32* mouse_position );

        void SetPosition( DirectX::XMFLOAT3 position );
        void SetPerspective( FLOAT fov_y, FLOAT width, FLOAT height, FLOAT near_z, FLOAT far_z );

        const DirectX::XMMATRIX GetView();
        const DirectX::XMMATRIX GetProjection();
        const DirectX::XMMATRIX GetProjectionFlippedZ();
        const DirectX::XMMATRIX GetViewProjection();
        const DirectX::XMMATRIX GetViewProjectionFlippedZ();

        const DirectX::XMVECTOR GetPosition();
        const DirectX::XMVECTOR GetRight();
        const DirectX::XMVECTOR GetUp();
        const DirectX::XMVECTOR GetLook();

        const FLOAT GetNearZ();
        const FLOAT GetFarZ();
    };
}
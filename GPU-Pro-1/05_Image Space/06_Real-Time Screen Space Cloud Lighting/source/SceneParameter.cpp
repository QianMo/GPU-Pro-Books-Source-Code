//--------------------------------------------------------------------------------------
// SceneParameter.cpp
// 
// Implementation of the camera, light and scattering class.
// 
// Copyright (C) Kaori Kubota. All rights reserved.
//--------------------------------------------------------------------------------------
#include "DXUT.h"
#include "SceneParameter.h"
#include "Ground.h"


//--------------------------------------------------------------------------------------

CSceneCamera::CSceneCamera()
: m_pGround(NULL)
{
}



//--------------------------------------------------------------------------------------
// Update the view matrix based on user input & elapsed time
//--------------------------------------------------------------------------------------
VOID CSceneCamera::FrameMove( FLOAT fElapsedTime )
{
    if( DXUTGetGlobalTimer()->IsStopped() )
        fElapsedTime = 1.0f / DXUTGetFPS();

    if( IsKeyDown( m_aKeys[CAM_RESET] ) )
        Reset();

    // Get keyboard/mouse/gamepad input
    GetInput( m_bEnablePositionMovement, ( m_nActiveButtonMask & m_nCurrentButtonMask ) || m_bRotateWithoutButtonDown,
              true, m_bResetCursorAfterMove );

    // Get amount of velocity based on the keyboard input and drag (if any)
    UpdateVelocity( fElapsedTime );

    // Simple euler method to calculate position delta
    D3DXVECTOR3 vPosDelta = m_vVelocity * fElapsedTime;

    // If rotating the camera 
    if( ( m_nActiveButtonMask & m_nCurrentButtonMask ) ||
        m_bRotateWithoutButtonDown ||
        m_vGamePadRightThumb.x != 0 ||
        m_vGamePadRightThumb.z != 0 )
    {
        // Update the pitch & yaw angle based on mouse movement
        float fYawDelta = m_vRotVelocity.x;
        float fPitchDelta = m_vRotVelocity.y;

        // Invert pitch if requested
        if( m_bInvertPitch )
            fPitchDelta = -fPitchDelta;

        m_fCameraPitchAngle += fPitchDelta;
        m_fCameraYawAngle += fYawDelta;

        // Limit pitch to straight up or straight down
        m_fCameraPitchAngle = __max( -D3DX_PI / 2.0f, m_fCameraPitchAngle );
        m_fCameraPitchAngle = __min( +D3DX_PI / 2.0f, m_fCameraPitchAngle );
    }

    // Make a rotation matrix based on the camera's yaw & pitch
    D3DXMATRIX mCameraRot;
    D3DXMatrixRotationYawPitchRoll( &mCameraRot, m_fCameraYawAngle, m_fCameraPitchAngle, 0 );

    // Transform vectors based on camera's rotation matrix
    D3DXVECTOR3 vWorldUp, vWorldAhead;
    D3DXVECTOR3 vLocalUp = D3DXVECTOR3( 0, 1, 0 );
    D3DXVECTOR3 vLocalAhead = D3DXVECTOR3( 0, 0, 1 );
    D3DXVec3TransformCoord( &vWorldUp, &vLocalUp, &mCameraRot );
    D3DXVec3TransformCoord( &vWorldAhead, &vLocalAhead, &mCameraRot );

    // Transform the position delta by the camera's rotation 
    D3DXVECTOR3 vPosDeltaWorld;
    if( !m_bEnableYAxisMovement )
    {
        // If restricting Y movement, do not include pitch
        // when transforming position delta vector.
        D3DXMatrixRotationYawPitchRoll( &mCameraRot, m_fCameraYawAngle, 0.0f, 0.0f );
    }
    D3DXVec3TransformCoord( &vPosDeltaWorld, &vPosDelta, &mCameraRot );

    // Move the lookAt position 
    m_vLookAt += vPosDeltaWorld;
	if (m_pGround != NULL) {
		FLOAT fGround = m_pGround->GetHeight( m_vLookAt.x, m_vLookAt.z );
	    m_vLookAt.y = fGround+100.0f;//__max( m_vLookAt.y, fGround+100.0f );
	}

    // Update the eye position based on the lookAt position 
    m_vEye = m_vLookAt - vWorldAhead;

    if( m_bClipToBoundary )
        ConstrainToBoundary( &m_vEye );

	// Modify height of the eye position so that camera is always above the ground.
	if (m_pGround != NULL) {
		FLOAT fGround = m_pGround->GetHeight( m_vEye.x, m_vEye.z );
	    m_vEye.y = __max( m_vEye.y, fGround );
	}


    // Update the view matrix
    D3DXMatrixLookAtLH( &m_mView, &m_vEye, &m_vLookAt, &vWorldUp );

    D3DXMatrixInverse( &m_mCameraWorld, NULL, &m_mView );

	// Update the view and projection matrix  
	D3DXMatrixMultiply( &m_mW2C, &m_mView, GetProjMatrix() );
}





//--------------------------------------------------------------------------------------
// SBoundingBox::Transform
//  Compute bounding box in the indicated coordinate.
//  pMat must not be a projection matrix.
//--------------------------------------------------------------------------------------
VOID SBoundingBox::Transform(SBoundingBox& box, const D3DXMATRIX* pMat) const
{
	D3DXVECTOR3 vCenter;
	D3DXVECTOR3 vDiagonal;
	Centroid( &vCenter );
	HalfDiag( &vDiagonal );
	D3DXVECTOR3 vAxisX( vDiagonal.x, 0.0f, 0.0f );
	D3DXVECTOR3 vAxisY( 0.0f, vDiagonal.y, 0.0f );
	D3DXVECTOR3 vAxisZ( 0.0f, 0.0f, vDiagonal.z );

	// Transform the center position and the axis vectors.
	D3DXVec3TransformCoord( &vCenter, &vCenter, pMat );
	D3DXVec3TransformNormal( &vAxisX, &vAxisX, pMat );
	D3DXVec3TransformNormal( &vAxisY, &vAxisY, pMat );
	D3DXVec3TransformNormal( &vAxisZ, &vAxisZ, pMat );

	vDiagonal = D3DXVECTOR3(
		fabsf( vAxisX.x ) + fabsf( vAxisY.x ) + fabsf( vAxisZ.x ),
		fabsf( vAxisX.y ) + fabsf( vAxisY.y ) + fabsf( vAxisZ.y ),
		fabsf( vAxisX.z ) + fabsf( vAxisY.z ) + fabsf( vAxisZ.z ) );
	D3DXVec3Add( &box.vMax, &vCenter, &vDiagonal );
	D3DXVec3Subtract( &box.vMin, &vCenter, &vDiagonal );
}





//--------------------------------------------------------------------------------------
// SSceneParamter::GetShaderParam
//  Get scatterign parameters for pixel shader 
//--------------------------------------------------------------------------------------
VOID SSceneParamter::GetShaderParam(SScatteringShaderParameters& param) const
{
	static const FLOAT PI = 3.14159265f;

	// vRayleigh : rgb : 3/(16*PI) * Br           w : -2*g
	D3DXVec3Scale( (D3DXVECTOR3*)&param.vRayleigh, &m_vRayleigh, 3.0f/(16.0f*PI) );
	param.vRayleigh.w = -2.0f * m_fG;

	// vMie : rgb : 1/(4*PI) * Bm * (1-g)^2  w : (1+g^2)
	FLOAT       fG = 1.0f -m_fG;
	D3DXVec3Scale( (D3DXVECTOR3*)&param.vMie, &m_vMie, fG*fG/(4.0f*PI) );
	param.vMie.w = 1.0f + m_fG * m_fG;

	D3DXVECTOR3 vSum;
	D3DXVec3Add( (D3DXVECTOR3*)&vSum, &m_vRayleigh, &m_vMie );

	// vESun : rgb : Esun/(Br+Bm)             w : R
	param.vESun.x = m_fLightScale * m_vLightColor.x/vSum.x;
	param.vESun.y = m_fLightScale * m_vLightColor.y/vSum.y;
	param.vESun.z = m_fLightScale * m_vLightColor.z/vSum.z;
	param.vESun.w = m_fEarthRadius;
	
	// vSum  : rgb : (Br+Bm)                  w : h(2R+h)
	// scale by inverse of farclip to apply constant scattering effect in case farclip is changed.
	D3DXVec3Scale( (D3DXVECTOR3*)&param.vSum, (D3DXVECTOR3*)&vSum, 1.0f/m_pCamera->GetFarClip() );
	param.vSum.w = m_fAtomosHeight * (2.0f*m_fEarthRadius + m_fAtomosHeight);

	// ambient term of scattering
	D3DXVec3Scale( (D3DXVECTOR3*)&param.vAmbient, &m_vAmbientLight, m_fAmbientScale );
	param.vAmbient.w = 1.0f/sqrtf( param.vSum.w );
}

//--------------------------------------------------------------------------------------
VOID SSceneParamter::SetTime(FLOAT fTimeOfADay)
{
	static const FLOAT PI = 3.141592f;
	FLOAT fAngle = ( 45.0f ) * PI / 180.0f;
	D3DXVECTOR3 vRotAxis( 0.0f, sinf( fAngle ), cosf( fAngle ) );

	D3DXMATRIX matRot;
	D3DXMatrixRotationAxis( &matRot, &vRotAxis, fTimeOfADay * (1.0f*PI) );

	D3DXVECTOR3 v( -1.0f, 0.0f, 0.0f );
	D3DXVec3TransformNormal( &m_vLightDir, &v, &matRot );
}


//--------------------------------------------------------------------------------------
// SceneParameter.h
//
// Declaration of the camera, light and scattering class.
//
// Copyright (C) Kaori Kubota. All rights reserved.
// 
// 
// Computation of daylight scattering is based on 
// "Rendering Outdoor Light Scattering in Real Time" by Naty Hoffman and Arcot J Preetham .
// The paper's URL is http://ati.amd.com/developer/dx9/ATI-LightScattering.pdf .
//--------------------------------------------------------------------------------------


#if !defined(__INCLUDED_CAMERA_H__)
#define __INCLUDED_CAMERA_H__

#include "DXUTcamera.h"

class CGround;


//--------------------------------------------------------------------------------------
// CSceneCamera 
//  Camera object moving on the ground in this demo scene.
//--------------------------------------------------------------------------------------
class CSceneCamera : public CFirstPersonCamera {
public :
	CSceneCamera();

	virtual VOID FrameMove( FLOAT fElapsedTime );

	inline const D3DXMATRIX* GetWorld2ViewMatrix() const;
	inline const D3DXMATRIX* GetWorld2ProjMatrix() const;
	inline VOID SetGround( CGround* pGround );

protected :
	CGround*   m_pGround;
	D3DXMATRIX m_mW2C;
};

//--------------------------------------------------------------------------------------
// Return the view matrix 
//--------------------------------------------------------------------------------------
const D3DXMATRIX* CSceneCamera::GetWorld2ViewMatrix() const 
{
	return &m_mView;
}

//--------------------------------------------------------------------------------------
// Return the view and projection matrix 
//--------------------------------------------------------------------------------------
const D3DXMATRIX* CSceneCamera::GetWorld2ProjMatrix() const
{
	return &m_mW2C;
}

//--------------------------------------------------------------------------------------
// Initialize 
//--------------------------------------------------------------------------------------
VOID CSceneCamera::SetGround( CGround* pGround )
{
	m_pGround = pGround;
}



//--------------------------------------------------------------------------------------
// SBoundingBox 
//  Bounding box for shadow
//--------------------------------------------------------------------------------------
struct SBoundingBox {
	D3DXVECTOR3 vMin;
	D3DXVECTOR3 vMax;

	inline VOID Centroid(D3DXVECTOR3* p) const;
	inline VOID HalfDiag(D3DXVECTOR3* p) const;
	VOID Transform(SBoundingBox& box, const D3DXMATRIX* pMat) const;
};

//--------------------------------------------------------------------------------------
// Return center position
//--------------------------------------------------------------------------------------
VOID SBoundingBox::Centroid(D3DXVECTOR3* p) const
{
	D3DXVec3Add( p, &vMin, &vMax );
	D3DXVec3Scale( p, p, 0.5f );
}

//--------------------------------------------------------------------------------------
// Return Half diagonal vector
//--------------------------------------------------------------------------------------
VOID SBoundingBox::HalfDiag(D3DXVECTOR3* p) const
{
	D3DXVec3Subtract( p, &vMax, &vMin );
	D3DXVec3Scale( p, p, 0.5f );
}




//--------------------------------------------------------------------------------------
// SScatteringShaderParameters
//      Parameters for scattering in pixel shader 
// 
//   Scattering equation is 
//      L(s,theta) = L0 * Fex(s) + Lin(s,theta).
//   where,
//      Fex(s) = exp( -s * (Br+Bm) )
//      Lin(s,theta) = (Esun * ((Br(theta)+Bm(theta))/(Br+Bm)) + ambient)* (1.0f - exp( -s * (Br+Bm) ))
//      Br(theta) = 3/(16*PI) * Br * (1+cos^2(theta))
//      Bm(theta) = 1/(4*PI) * Bm * ((1-g)^2/(1+g^2-2*g*cos(theta))^(3/2))
// 
//   Distance light goes through the atomosphere in a certain ray is 
//      Distance(phi) = -R*sin(phi) + sqrt( (R*sin(phi))^2 + h * (2*R+h) )
//   where,
//      R   : Earth radius
//      h   : atomosphere height
//      phi : angle between a ray vector and a horizontal plane.
//--------------------------------------------------------------------------------------
struct SScatteringShaderParameters {
	D3DXVECTOR4 vRayleigh;   // rgb : 3/(16*PI) * Br           w : -2*g
	D3DXVECTOR4 vMie;        // rgb : 1/(4*PI) * Bm * (1-g)^2  w : (1+g^2)
	D3DXVECTOR4 vESun;       // rgb : Esun/(Br+Bm)             w : R
	D3DXVECTOR4 vSum;        // rgb : (Br+Bm)                  w : h(2R+h)
	D3DXVECTOR4 vAmbient;    // rgb : ambient
};


//--------------------------------------------------------------------------------------
// SSceneParamter 
//  scene parameters
//--------------------------------------------------------------------------------------
struct SSceneParamter {
	// Camera
	CSceneCamera* m_pCamera;
	// light
	D3DXVECTOR3 m_vLightDir;    // sun light direction
	D3DXVECTOR3 m_vLightColor;  // sun light color
	D3DXVECTOR3 m_vAmbientLight;// ambient light
	// scattering
	D3DXVECTOR3 m_vRayleigh;     // rayleigh scattering
	D3DXVECTOR3 m_vMie;          // mie scattering
	FLOAT       m_fG;            // eccentricity of mie scattering 
	FLOAT       m_fLightScale;   // scaling parameter of light
	FLOAT       m_fAmbientScale; // scaline parameter of ambient term
	// distance for scattering
	FLOAT       m_fEarthRadius;  // radius of the earth
	FLOAT       m_fAtomosHeight; // height of the atomosphere
	FLOAT       m_fCloudHeight;  // height of cloud 


	VOID GetShaderParam(SScatteringShaderParameters& param) const;
	VOID SetTime(FLOAT fTime);

	inline VOID GetCloudDistance(D3DXVECTOR2& v) const;
};


//--------------------------------------------------------------------------------------
// Return parameters for computing distance to the cloud 
//--------------------------------------------------------------------------------------
VOID SSceneParamter::GetCloudDistance(D3DXVECTOR2& v) const
{
	v.x = m_fEarthRadius;
	v.y = m_fCloudHeight * (2.0f*m_fEarthRadius + m_fCloudHeight);
}




#endif 

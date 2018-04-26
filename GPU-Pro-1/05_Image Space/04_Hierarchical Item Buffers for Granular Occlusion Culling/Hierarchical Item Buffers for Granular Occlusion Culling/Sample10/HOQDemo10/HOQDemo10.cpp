#include "HOQDemo10.h"




void Camera::SetLookAt( const D3DXVECTOR3& eye, const D3DXVECTOR3& at, const D3DXVECTOR3& up ) {
	this->eye	= eye;
	this->at	= at;
	this->up	= up;

	D3DXMatrixLookAtLH( &viewMat, &eye, &at, &up );
}

void Camera::SetPerspective( float aspect, float fovy, float znear, float zfar ) {
	this->aspect	= aspect;
	this->fovy		= fovy;
	this->znear		= znear;
	this->zfar		= zfar;

	D3DXMatrixPerspectiveFovLH( &projMat, fovy * D3DX_PI / 180.f, aspect, znear, zfar );
}

void Camera::SetViewport( UINT width, UINT height ) {
	this->viewport.Width	= width;
	this->viewport.Height	= height;
	this->viewport.MinDepth	= 0.f;
	this->viewport.MaxDepth	= 1.f;
	this->viewport.TopLeftX	= 0;
	this->viewport.TopLeftY	= 0;
}

D3DXMATRIX* Camera::GetViewProjMat( D3DXMATRIX* pOut ) {
	D3DXMatrixMultiply( pOut, &viewMat, &projMat );
	return pOut;
}

D3DXVECTOR4* Camera::GetEyePos( D3DXVECTOR4* pOut ) {
	pOut->x = eye.x;
	pOut->y = eye.y;
	pOut->z = eye.z;
	pOut->w = 1.f;
	return pOut;
}


void CreateTransformationsFromParticle( D3DXMATRIX* o2w, D3DXMATRIX* n2w, Particle* p )
{
	float r = ( 0.3f + abs(sin( p->radius )) * 0.7f ) * 20.f;
	float x = cos( p->angle ) * r;
	float z = sin( p->angle ) * r;
	float y = (0.1f + 0.7f * abs( cos( p->height ) )) * 20.f;

	D3DXMATRIX tmp;
	D3DXMatrixScaling( o2w, 0.2f, 0.2f, 0.2f );
	D3DXMatrixTranslation( &tmp, x-6.f, y, z );
	D3DXMatrixMultiply( o2w, o2w, &tmp );

	FLOAT det = D3DXMatrixDeterminant( o2w );
	D3DXMatrixInverse( n2w, &det, o2w );
	D3DXMatrixTranspose( n2w, n2w );
}
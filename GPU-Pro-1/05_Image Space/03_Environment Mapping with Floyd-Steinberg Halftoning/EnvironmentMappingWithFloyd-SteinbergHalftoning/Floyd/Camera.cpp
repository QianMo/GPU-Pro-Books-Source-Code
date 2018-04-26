#include "DXUT.h"
#include "Camera.h"


D3DXVECTOR3 Camera::getAhead()
{
	const D3DXMATRIX& viewMatrix = getViewMatrix();
	D3DXMATRIX viewMatrixInverse;
	D3DXMatrixInverse(&viewMatrixInverse, NULL, &viewMatrix);
	D3DXVECTOR3 ahead;
	D3DXVec3TransformNormal(&ahead, &D3DXVECTOR3(0, 0, 1), &viewMatrixInverse);
	return ahead;
}

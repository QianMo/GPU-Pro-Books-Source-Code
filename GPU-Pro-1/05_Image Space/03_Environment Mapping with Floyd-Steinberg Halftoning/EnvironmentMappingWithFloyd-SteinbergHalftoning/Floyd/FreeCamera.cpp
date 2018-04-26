#include "DXUT.h"
#include "FreeCamera.h"

const D3DXVECTOR3& FreeCamera::getEyePosition()
{
	return *camera.GetEyePt();
}

const D3DXMATRIX& FreeCamera::getOrientProjMatrixInverse()
{
	static D3DXMATRIX vpm;
	const D3DXVECTOR3& eyePosition = getEyePosition();
	D3DXMATRIX eyePosTranslationMatrix;
	D3DXMatrixTranslation(&eyePosTranslationMatrix, eyePosition.x, eyePosition.y, eyePosition.z);
	D3DXMatrixInverse(&vpm, NULL, &(eyePosTranslationMatrix * *camera.GetViewMatrix() * *camera.GetProjMatrix()));
	return vpm;
}

const D3DXMATRIX& FreeCamera::getViewMatrix()
{
	return *camera.GetViewMatrix();
}

const D3DXMATRIX& FreeCamera::getProjMatrix()
{
	return *camera.GetProjMatrix();
}

void FreeCamera::handleInput(HWND hWnd,  UINT uMsg, WPARAM wParam, LPARAM lParam)
{
	camera.HandleMessages(hWnd, uMsg, wParam, lParam);
}

void FreeCamera::animate(double dt)
{
	camera.FrameMove(dt);
}

FreeCamera::FreeCamera(D3DXVECTOR3& eye, D3DXVECTOR3& lookAt)
{
	camera.SetViewParams(&eye, &lookAt);
	camera.SetProjParams(1.57, 1, 0.1, 8000);
//	camera.SetScalers(0.01, 2.0);
	camera.SetScalers(0.01, 10.0);
}

void FreeCamera::setAspect(double aspect)
{
	camera.SetProjParams(1.57, aspect, 0.1, 8000);
}

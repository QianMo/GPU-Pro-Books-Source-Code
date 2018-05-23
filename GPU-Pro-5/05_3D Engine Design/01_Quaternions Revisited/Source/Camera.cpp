/****************************************************************************

  GPU Pro 5 : Quaternions revisited - sample code
  All sample code written from scratch by Sergey Makeev specially for article.

  Copyright (c) 2013, Sergey Makeev

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software.

  2. If you use this software in a non-commercial product, an acknowledgment
     in the product documentation would be appreciated but is not required.

  3. If you use this software in a commercial product you requried to credit
     the author.

  4. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.

  5. This notice may not be removed or altered from any source distribution.
    

   Please let me know if you use the code in your products or have any questions or suggestions.

   e-mail: sergeymakeev@inbox.ru
   http://linkedin.com/in/sergeymakeev/

****************************************************************************/
#include "Camera.h"


FreeCamera::FreeCamera()
{
	mouseLookSavedPosition.x = 0;
	mouseLookSavedPosition.y = 0;
	isMouseLook = false;
	cameraPosition = D3DXVECTOR3(9.2548857f, 2.2040517f, 12.318352f);
	cameraRotation = D3DXQUATERNION(-0.026506374f, 0.95308179f, -0.087554961f, -0.28853607f);
}

FreeCamera::~FreeCamera()
{
}

void FreeCamera::Update(float deltaTime)
{
	float cameraMoveSpeed = 6.0f;
	const float cameraRotationSpeed = 0.08f;

	if (isMouseLook)
	{
		if (GetAsyncKeyState(VK_SHIFT) < 0)
		{
			cameraMoveSpeed = cameraMoveSpeed * 10.0f;
		}

		if (GetAsyncKeyState('W') < 0)
		{
			D3DXVECTOR3 forward = GetForward();
			cameraPosition += forward * deltaTime * cameraMoveSpeed;
		}

		if (GetAsyncKeyState('S') < 0)
		{
			D3DXVECTOR3 forward = GetForward();
			cameraPosition -= forward * deltaTime * cameraMoveSpeed;
		}

		if (GetAsyncKeyState('D') < 0)
		{
			D3DXVECTOR3 strafe = GetStrafe();
			cameraPosition -= strafe * deltaTime * cameraMoveSpeed;
		}

		if (GetAsyncKeyState('A') < 0)
		{
			D3DXVECTOR3 strafe = GetStrafe();
			cameraPosition += strafe * deltaTime * cameraMoveSpeed;
		}
	}

	int mouseDeltaX = 0;
	int mouseDeltaY = 0;

	if (GetAsyncKeyState(VK_RBUTTON) < 0)
	{
		if (isMouseLook)
		{
			POINT mousePos;
			GetCursorPos(&mousePos);

			mouseDeltaX = mousePos.x - 640;
			mouseDeltaY = mousePos.y - 360;
		} else
		{
			GetCursorPos(&mouseLookSavedPosition);
			ShowCursor(FALSE);
		}

		SetCursorPos(640, 360);
		isMouseLook = true;

	} else
	{
		if (isMouseLook)
		{
			SetCursorPos(mouseLookSavedPosition.x, mouseLookSavedPosition.y);
			ShowCursor(TRUE);
		}

		isMouseLook = false;
	}


	if (mouseDeltaY != 0)
	{
		D3DXQUATERNION deltaRotation;
		D3DXQuaternionRotationYawPitchRoll(&deltaRotation, 0.0f, cameraRotationSpeed * mouseDeltaY * deltaTime, 0.0f);
		D3DXQuaternionMultiply(&cameraRotation, &deltaRotation, &cameraRotation);
	}

	if (mouseDeltaX != 0)
	{
		D3DXQUATERNION deltaRotation;
		D3DXQuaternionRotationYawPitchRoll(&deltaRotation, cameraRotationSpeed * -mouseDeltaX * deltaTime, 0.0f, 0.0f);
		D3DXQuaternionMultiply(&cameraRotation, &cameraRotation, &deltaRotation);
	}

}


D3DXMATRIXA16 FreeCamera::GetView() const
{
	D3DXMATRIXA16 mtxCameraRotation;
	D3DXMatrixRotationQuaternion(&mtxCameraRotation, &cameraRotation);

	D3DXMATRIXA16 mtxCameraTranslation;
	D3DXMatrixTranslation(&mtxCameraTranslation, cameraPosition.x, cameraPosition.y, cameraPosition.z);

	D3DXMATRIXA16 mtxCamera;
	D3DXMatrixMultiply(&mtxCamera, &mtxCameraRotation, &mtxCameraTranslation);

	D3DXMATRIXA16 mtxView;
	D3DXMatrixInverse(&mtxView, NULL, &mtxCamera);

	return mtxView;
}

D3DXVECTOR3 FreeCamera::GetForward() const
{
	D3DXMATRIX mtxCamRotation;
	D3DXMatrixRotationQuaternion(&mtxCamRotation, &cameraRotation);
	D3DXVECTOR3 forward(mtxCamRotation.m[2][0], mtxCamRotation.m[2][1], mtxCamRotation.m[2][2]);

	return forward;
}


D3DXVECTOR3 FreeCamera::GetStrafe() const
{
	D3DXMATRIX mtxCamRotation;
	D3DXMatrixRotationQuaternion(&mtxCamRotation, &cameraRotation);
	D3DXVECTOR3 strafe(mtxCamRotation.m[0][0], mtxCamRotation.m[0][1], mtxCamRotation.m[0][2]);
	return strafe;
}
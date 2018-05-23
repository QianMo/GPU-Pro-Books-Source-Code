//Copyright(c) 2009 - 2011, yakiimo02
//	All rights reserved.
//
//Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met :
//
//*Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
//
//	* Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and / or other materials provided with the distribution.
//
//	* Neither the name of Yakiimo3D nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
//
//	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// Original per-pixel linked list implementation code from Yakiimo02 was altered by Joao Raza and Gustavo Nunes for GPU Pro 5 'Screen Space Deformable Meshes via CSG with Per-pixel Linked Lists'

#include "Camera.h"

using namespace DirectX;

Camera::Camera(float height, float width)  
{
	m_leftRightRotation = 0.0f;
	m_upDownRotation = 0.0f;
	m_lastMousePos = XMVectorSet(0.0f, 0.0f, 0.0f, 0.0f);
	m_Position = XMVectorSet(0.0f, 0.0f, -10.0f, 0.0f);
	m_Proj = XMMatrixPerspectiveFovLH(XM_PIDIV4, width / (float) height, 0.5f, 10000.0f);
	this->UpdateViewMatrix();
}

void Camera::SetValues(const FXMVECTOR& pos, const FXMVECTOR forward, const FXMVECTOR& up, const float udrot, const float lrrot)
{
	m_Position = pos;
	m_Forward  = forward;
	m_UpVector = up;
	
	m_upDownRotation    = udrot;
	m_leftRightRotation = lrrot;

	this->UpdateViewMatrix();
}

XMVECTOR Camera::GetPosition() const
{
	return m_Position;
}

XMVECTOR Camera::GetForward() const
{
	return m_Forward;
}

XMVECTOR Camera::GetUpVector() const
{
	return m_UpVector;
}

XMMATRIX Camera::GetViewMatrix() const
{
	return m_View;
}

XMMATRIX Camera::GetProjectionMatrix() const
{
	return m_Proj;
}

void Camera::UpdateViewMatrix()
{
	XMMATRIX rot = XMMatrixRotationX(m_upDownRotation) * XMMatrixRotationY(m_leftRightRotation);
	XMVECTOR originalTarget = XMVectorSet(0.0f, 0.0f, 1.0f, 0.0f);
	XMVECTOR originalUP = XMVectorSet(0.0f, 1.0f, 0.0f, 0.0f);
	XMVECTOR rotatedTarget = XMVector3Transform(originalTarget, rot);

	m_Forward = m_Position + rotatedTarget;
	m_UpVector = XMVector3Transform(originalUP, rot);
	m_View = XMMatrixLookAtLH(m_Position, m_Forward, m_UpVector);
}

void Camera::MoveCamera(const XMVECTOR& directionToMove)
{
	float moveSpeed = 0.3f;
	XMMATRIX rot = XMMatrixRotationX(m_upDownRotation) * XMMatrixRotationY(m_leftRightRotation);
	XMVECTOR rotatedVector = XMVector3Transform(directionToMove, rot);
	m_Position = m_Position + moveSpeed * rotatedVector;
	this->UpdateViewMatrix();
}

void Camera::Update(float delta)
{
	(void) delta; //! removes warning for delta not being used. included as parameter for future reference. 

	if (GetAsyncKeyState('W') & 0x8000)
	{
		this->MoveCamera(XMVectorSet(0, 0, 1, 0));
	}
	if (GetAsyncKeyState('A') & 0x8000)
	{
		this->MoveCamera(XMVectorSet(-1, 0, 0, 0));
	}
	if (GetAsyncKeyState('S') & 0x8000)
	{
		this->MoveCamera(XMVectorSet(0, 0, -1, 0));
	}
	if (GetAsyncKeyState('D') & 0x8000)
	{
		this->MoveCamera(XMVectorSet(1, 0, 0, 0));
	}
}

void Camera::RotateCamera(WPARAM buttonState, int x, int y)
{
	float rotationSpeed = 0.01f;
	
	if ((buttonState & MK_LBUTTON) != 0)
	{
		float previousX = XMVectorGetX(m_lastMousePos);
		float previousY = XMVectorGetY(m_lastMousePos);
		float deltaX = x - previousX;
		float deltaY = y - previousY;

		m_leftRightRotation = m_leftRightRotation + (rotationSpeed * deltaX);
		m_upDownRotation = m_upDownRotation + (rotationSpeed * deltaY);
		this->UpdateViewMatrix();
	}

	m_lastMousePos = XMVectorSet(float(x), float(y), 0, 0);
}




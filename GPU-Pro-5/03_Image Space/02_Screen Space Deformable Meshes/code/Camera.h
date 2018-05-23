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

#include <directxmath.h>
#include <windows.h>

using namespace DirectX;

class Camera
{
public:
	XMVECTOR GetPosition() const;
	XMVECTOR GetForward() const;
	XMVECTOR GetUpVector() const;
	XMMATRIX GetViewMatrix() const;
	XMMATRIX GetProjectionMatrix() const;
	Camera(float height, float width);
	void UpdateViewMatrix();
	void Update(float delta);
	void MoveCamera(const XMVECTOR& directionToMove);
	void RotateCamera(WPARAM btnState, int x, int y);
	
	void SetValues(const FXMVECTOR& pos, const FXMVECTOR forward, const FXMVECTOR& up, const float udrot, const float lrrot);
private:
	XMVECTOR m_Position;
	XMVECTOR m_Forward;
	XMVECTOR m_UpVector;
	XMMATRIX m_View;
	XMMATRIX m_Proj;
	XMVECTOR m_lastMousePos;
	float m_leftRightRotation;
	float m_upDownRotation;


};
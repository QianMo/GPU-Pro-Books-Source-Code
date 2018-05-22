// ================================================================================ //
// Copyright (c) 2011 Arturo Garcia, Francisco Avila, Sergio Murguia and Leo Reyes	//
//																					//
// Permission is hereby granted, free of charge, to any person obtaining a copy of	//
// this software and associated documentation files (the "Software"), to deal in	//
// the Software without restriction, including without limitation the rights to		//
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies	//
// of the Software, and to permit persons to whom the Software is furnished to do	//
// so, subject to the following conditions:											//
//																					//
// The above copyright notice and this permission notice shall be included in all	//
// copies or substantial portions of the Software.									//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR		//
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,			//
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE		//
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER			//
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,	//
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE	//
// SOFTWARE.																		//
// ================================================================================ //

// ------------------------------------------------
// Camera.h
// ------------------------------------------------
// Manipulates the camera on the scene.

#ifndef CAMERA_H
#define CAMERA_H

#include "Common.h"
#include "Geometry.h"
#include "Matrix4.h"

class Camera
{
private:
	Matrix4	m_mInverseMatrix;
	float	m_fSpeed;		// Movemente speed
	
	// Controls the movement of the camera.
	bool	m_bFront;		// W									
	bool	m_bBack;		// S						
	bool	m_bLeft;		// A							
	bool	m_bRight;		// D							
	bool	m_bUp;			// Q						
	bool	m_bDown;		// E							
	bool	m_bPause;		// P
	bool	m_bOrbit;
public:
	Camera() {}
	Camera(Point& pos, float speed);
	~Camera(void);

	// Functions
	bool	Move(float fTime);											// Moves the Camera using the keyboard
	void	Update();													// Updates the position of the camera every frame
	void	Rotate(int x, int y);										// Rotates the camera using the mouse
	void	IncreaseSpeed(void);
	void	DecreaseSpeed(void);
	void	SetPosition(Point &a_Pos);
	void	Turn( int axis, float angle );
	void	SetCamera( unsigned int a_iOption ) ;
	void	Orbit( float a_fTimer);
	void	LookAt(const Vector3& dir, const Vector3& up, Matrix4& m); 

	void	ChangeOrbitingState() { m_bOrbit = !m_bOrbit; printf("Orbiting is %s\n", m_bOrbit?"ON":"OFF"); }
	void	ChangePausingState() { m_bPause = !m_bPause; printf("Pause is %s\n", m_bPause?"ON":"OFF"); }

	// Getters
	Vector3 GetPosition() { return Vector3(m_mInverseMatrix.xt, m_mInverseMatrix.yt, m_mInverseMatrix.zt);}
	Matrix4	GetInverseMatrix() { return m_mInverseMatrix; }
	float	GetSpeed() { return m_fSpeed; }
	bool	IsFront() { return m_bFront; }
	bool	IsBack() { return m_bBack; }
	bool	IsLeft() { return m_bLeft; }
	bool	IsRight() { return m_bRight; }
	bool	IsUp() { return m_bUp; }
	bool	IsDown() { return m_bDown; }
	bool	IsPaused() { return m_bPause; }
	bool	IsOrbiting() { return m_bOrbit; }

	// Setters
	void	SetFront( bool bValue ) { m_bFront = bValue; }
	void	SetBack( bool bValue ) { m_bBack = bValue; }
	void	SetLeft( bool bValue ) { m_bLeft = bValue; }
	void	SetRight( bool bValue ) { m_bRight = bValue; }
	void	SetUp( bool bValue ) { m_bUp = bValue; }
	void	SetDown( bool bValue ) { m_bDown = bValue; }
	void	SetPause( bool bValue ) { m_bPause = bValue; }
	void	SetOrbit( bool bValue ) { m_bOrbit = bValue; }
};

#endif
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

#include "Camera.h"

//------------------------------------------------
// Constructur
//------------------------------------------------
Camera::Camera(Point &position, float fSpeed)
{
	m_fSpeed = fSpeed;

	m_bFront = false;									
	m_bBack = false;									
	m_bLeft = false;									
	m_bRight = false;									
	m_bUp = false;									
	m_bDown = false;									
	m_bPause = false;	
	m_bOrbit = false;

	SetPosition(position);
}

//------------------------------------------------
// Destructor
//------------------------------------------------
Camera::~Camera(void)
{
}

//------------------------------------------------
// Increases the movement speed
//------------------------------------------------
void Camera::IncreaseSpeed(void) 
{ 
	if(m_fSpeed < 2.f) 
	{ 
		m_fSpeed += 0.1; 
		printf("Increase speed. "); 
	} 
	else
	{
		printf("Speed cannot increase.\n");
	}
	printf("New speed = %f\n", m_fSpeed); 
}	

//------------------------------------------------
// Decreases the movement speed
//------------------------------------------------
void Camera::DecreaseSpeed(void) 
{ 
	if(m_fSpeed > 0.1f) 
	{ 
		m_fSpeed -= 0.1; 
		printf("Decrease speed. "); 
	}
	else
	{
		printf("Speed cannot decrease.\n");
	}
	printf("New speed = %f\n", m_fSpeed); 
}

//------------------------------------------------
// Manipulate the Camera (keyboard)
//------------------------------------------------
bool Camera::Move(float fTime)
{
	// If is paused = do nothing
	if (m_bPause) return false;

	//Move as necessary
	Vector3 traslation(0.0f,0.0f,0.0f);
	if (m_bFront) traslation.z -= 1.0;
	if (m_bBack) traslation.z += 1.0;
	if (m_bLeft) traslation.x += 1.0;
	if (m_bRight) traslation.x -= 1.0;
	if (m_bUp) traslation.y -= 1.0;
	if (m_bDown) traslation.y += 1.0;

	float NormSQ;
	Dot(NormSQ, traslation,traslation);
	if(NormSQ > 0.1f)
	{
		traslation = (-fTime*m_fSpeed/sqrt(NormSQ)) * traslation;
		Matrix4 trans, tmpTrans = m_mInverseMatrix;
		M4_Translation(trans,traslation.x, traslation.y, traslation.z);
		M4_Compose(m_mInverseMatrix, trans, tmpTrans);
		return true;
	}

	Orbit(fTime);

	return false;
}

void Camera::Update(void)
{
}

//------------------------------------------------
// Rotate camera (mouse).
//------------------------------------------------
void Camera::Rotate(int x, int y)
{
	Matrix4 tmpTransform3 ,transRotation;
	float norma = float(x*x+y*y);
	if (norma>0.1f)
	{
		norma = sqrt(norma);
		Vector3 Axle(float(y),float(x),0.0f);
		Axle = (1.0f/norma)*Axle;
		M4_Rotation(transRotation,Axle.x,Axle.y,Axle.z,-norma/300.0f);
		tmpTransform3 = m_mInverseMatrix;
		M4_Compose(m_mInverseMatrix,transRotation,tmpTransform3);
		M4_MakeOrthoNormal(m_mInverseMatrix);
	}
}

//------------------------------------------------
// Rotate camera on a given angle (keyboard)
//------------------------------------------------
void Camera::Turn( int axis, float angle )
{
	printf("Turn camera %f degrees on %d-axis\n", angle,axis);
	float ax[3] = { static_cast<float>(axis == 0), static_cast<float>(axis == 1), static_cast<float>(axis == 2) };
	float radians = (angle*3.14159265)/180.f;
	float rot[4] = {ax[0], ax[1], ax[2], radians};
	Matrix4 rotation, tmp, tmp2;
	M4_Rotation(rotation, rot[0],rot[1],rot[2], rot[3]);
	M4_Identity(tmp);
	M4_Compose(tmp2, tmp, rotation);

	M4_Compose(tmp, tmp2, m_mInverseMatrix);
	m_mInverseMatrix = tmp;
}

//------------------------------------------------
// Set a predetermined camera (keyboard)
//------------------------------------------------
void Camera::SetCamera( unsigned int uiOption ) 
{
	printf("Set camera %d\n", uiOption);

	Point p;
	switch(uiOption)
	{
	case 0:
		p = Point(0.f,0.f,0.f);
		SetPosition(p);
		break;
	case 1:
		p = Point(0.f,0.f,-1.5f);
		SetPosition(p);
		break;
	case 2:
		p = Point(0.208097, -0.100240, 0.000600);
		SetPosition(p);	
		break;
	case 3:
		p = Point(0.212407,-0.095989,-0.000000);
		SetPosition(p);
		Turn( 1, 90 );
		break;
	case 4:
		p = Point(0.233747,-0.005260,-0.027424);
		SetPosition(p);
		Turn( 1, -90 );
		break;
	case 5:
		p = Point(0.233747,-0.005260,-0.027424);
		SetPosition(Point(-0.002284,-0.100781,-0.035037));
		Turn( 1, -67.5 );
		break;
	case 6:
		p = Point(-0.252590,0.309072,0.001339);
		SetPosition(p);
		Turn( 1, 90 );
		Turn( 0, 45 );
		break;
	case 7:
		p = Point(-0.122386,0.104292,-0.030843);
		SetPosition(p);
		Turn( 0, 67.5 );
		break;
	case 8:
		p = Point(-0.270368,0.000899,0.097372);
		SetPosition(p);
		Turn( 1 , 90 );
		break;
	case 9:
		break;
	default:
		p = Point(0.f,0.f,0.0f);
		SetPosition(p);
		break;
	}
}

//------------------------------------------------
// Orbit camera (keyboard)
//------------------------------------------------
void Camera::Orbit( float a_fTimer )
{
	if(m_bOrbit)
	{
		Vector3 traslation(1.0f,0.0f,0.0f);
		float NormSQ;
		Dot(NormSQ, traslation,traslation);

		float angle = a_fTimer*4*m_fSpeed/sqrt(NormSQ);
		float ax[3] = { 0.f,1.f,0.f };
		float radians = (angle*3.14159265)/180.f;
		float rot[4] = {ax[0], ax[1], ax[2], radians};
		Matrix4 rotation, tmp, tmp2;
		M4_Rotation(rotation, rot[0],rot[1],rot[2], rot[3]);
		M4_Identity(tmp);
		M4_Compose(tmp2, tmp, rotation);

		M4_Compose(tmp, m_mInverseMatrix, tmp2);
		m_mInverseMatrix = tmp;
	}
}

//------------------------------------------------
// Change to a camera to a specific position
//------------------------------------------------
void Camera::SetPosition(Point &a_Pos) 
{ 	
	printf("Camera position: %f,%f,%f\n", a_Pos.x, a_Pos.y,a_Pos.z);
	m_mInverseMatrix = Matrix4();
	M4_Translation( m_mInverseMatrix,a_Pos.x,a_Pos.y,a_Pos.z ); 
}

//------------------------------------------------
// Look at a specific point
//------------------------------------------------
void Camera::LookAt(const Vector3& dir, const Vector3& up, Matrix4& m) 
{ 
    Vector3 z(dir); 
    Normalize(z);
	Vector3 x;
    Cross(x,up,z); // x = up cross z 
    Normalize(x);
	Vector3 y;
	Cross(y,z,x); // x = up cross z 
} 
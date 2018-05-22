#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <windows.h>
#include <GL/gl.h>

#include <stdio.h>

#include <iostream>
#include <fstream>
#include <cstring>
#include <sstream>

#include "Camera.h"
#include "../Input/InputManager.h"
#include "../Util/Matrix4.h"
#include "../Util/Vector3.h"
#include "../Util/Math.h"

#include <GL/glut.h>

// -----------------------------------------------------------------------------
// ----------------------- Camera::Camera --------------------------------------
// -----------------------------------------------------------------------------
Camera::Camera(void):
		distance(0.0),
		rotationX(0.0),
		rotationY(0.0),
		updateFixedMatrix(true),
		useFreeFly(true)
{
}

// -----------------------------------------------------------------------------
// ----------------------- Camera::Camera --------------------------------------
// -----------------------------------------------------------------------------
Camera::~Camera(void)
{

}

// -----------------------------------------------------------------------------
// ----------------------- Camera::Init ----------------------------------------
// -----------------------------------------------------------------------------
bool Camera::Init(void)
{
	freeFlyPosition = Vector3(0.0f, 50.0f, 300.0f);
	freeFlyDirection = Vector3(0.0f, 0.0f, 1.0f);
	freeFlyRotationX = 0.0f;
	freeFlyRotationY = 0.0f;

	distance = 250.0f;
	rotationX = 30.0f;
	rotationY = 280.0f;
	return true;
}

// -----------------------------------------------------------------------------
// -------------------------------- Camera::Exit -------------------------------
// -----------------------------------------------------------------------------
void Camera::Exit(void)
{

}

// -----------------------------------------------------------------------------
// ----------------------- Camera::Update --------------------------------------
// -----------------------------------------------------------------------------
bool Camera::Update(float deltaTime)
{
	POINT mouseMovement = InputManager::Instance()->GetMousePosition();

	if (useFreeFly)
	{
		if (InputManager::Instance()->IsMouseKeyPressed(MOUSE_LEFT))
		{
			freeFlyRotationX -= (float) 2.5f * mouseMovement.y * deltaTime;
			freeFlyRotationY -= (float) 5.0f * mouseMovement.x * deltaTime;

			if (freeFlyRotationX < -85.0f)
				freeFlyRotationX = -85.0f;
			if (freeFlyRotationX >= 85.0f)
				freeFlyRotationX = 85.0f;
		}

		Matrix4 xRoation = Matrix4::Matrix4RotationX(freeFlyRotationX*Math::DEG_TO_RAD);
		Matrix4 yRoation = Matrix4::Matrix4RotationY(freeFlyRotationY*Math::DEG_TO_RAD);

		freeFlyDirection = xRoation*yRoation*Vector3(0.0f, 0.0f, 1.0f);
		freeFlyDirection.Normalize();

		Vector3 rightVector = freeFlyDirection.CrossProduct(Vector3(0.0f, 1.0f, 0.0f));
		Vector3 realUpVector = freeFlyDirection.CrossProduct(rightVector);
		
		if (InputManager::Instance()->IsKeyPressed('w'))
		{
			if (InputManager::Instance()->IsKeyPressed(KEY_SPACE))
				freeFlyPosition -= realUpVector*200.0f*deltaTime;
			else
				freeFlyPosition -= freeFlyDirection*200.0f*deltaTime;
		}
		else if (InputManager::Instance()->IsKeyPressed('s'))
		{
			if (InputManager::Instance()->IsKeyPressed(KEY_SPACE))
				freeFlyPosition += realUpVector*200.0f*deltaTime;
			else
				freeFlyPosition += freeFlyDirection*200.0f*deltaTime;
		}
		else if (InputManager::Instance()->IsKeyPressed('i'))
		{
			if (InputManager::Instance()->IsKeyPressed(KEY_SPACE))
				freeFlyPosition -= realUpVector*20.0f*deltaTime;
			else
				freeFlyPosition -= freeFlyDirection*20.0f*deltaTime;
		}
		else if (InputManager::Instance()->IsKeyPressed('k'))
		{
			if (InputManager::Instance()->IsKeyPressed(KEY_SPACE))
				freeFlyPosition += realUpVector*20.0f*deltaTime;
			else
				freeFlyPosition += freeFlyDirection*20.0f*deltaTime;
		}

		if (InputManager::Instance()->IsKeyPressed('a'))
			freeFlyPosition += rightVector*200.0f*deltaTime;
		else if (InputManager::Instance()->IsKeyPressed('d'))
			freeFlyPosition -= rightVector*200.0f*deltaTime;
		else if (InputManager::Instance()->IsKeyPressed('j'))
			freeFlyPosition += rightVector*20.0f*deltaTime;
		else if (InputManager::Instance()->IsKeyPressed('l'))
			freeFlyPosition -= rightVector*20.0f*deltaTime;
	}
	else
	{
		if (InputManager::Instance()->IsMouseKeyPressed(MOUSE_RIGHT))
		{
			distance += (float) 10.0f * mouseMovement.y * deltaTime;

			if (distance < 0.0f)
				distance = 0.0f;
			if (distance > 2250.0f)
				distance = 2250.0f;
		}
		else if (InputManager::Instance()->IsMouseKeyPressed(MOUSE_LEFT))
		{
			static bool firstTime = true;
			if (firstTime)
			{
				firstTime = false;
			}
			else
			{
				rotationX += (float) 5.0f * mouseMovement.y * deltaTime;
				rotationY += (float) 5.0f * mouseMovement.x * deltaTime;
			}

			if (rotationX < -85.0f)
				rotationX = -85.0f;
			if (rotationX >= 85.0f)
				rotationX = 85.0f;
			
			if (rotationY < 0.0f)
				rotationY = 360.0f;
			if (rotationY > 360.0f)
				rotationY = 0.0f;
		}
	}

	if (InputManager::Instance()->IsKeyPressedAndReset(KEY_BACKSPACE))
		updateFixedMatrix = !updateFixedMatrix;

	if (useFreeFly)
		viewMatrix = Matrix4::Matrix4LookDir(freeFlyPosition, freeFlyDirection, Vector3(0.0f, 1.0f, 0.0f));
	else
		viewMatrix = Matrix4::Matrix4RotationY(rotationY*Math::DEG_TO_RAD)*Matrix4::Matrix4RotationX(rotationX*Math::DEG_TO_RAD)*Matrix4::Matrix4Translation(Vector3(0.0f, 0.0f, -distance));

	if (updateFixedMatrix)
	{
		fixedMatrix = viewMatrix;
	}

	return true;
}

// -----------------------------------------------------------------------------
// ----------------------- Camera::SetViewMatrix -------------------------------
// -----------------------------------------------------------------------------
bool Camera::SetViewMatrix(void) 
{
	glLoadIdentity();
	glMultMatrixf(viewMatrix.entry);

	return true;
}

// -----------------------------------------------------------------------------
// ----------------------- Camera::SetViewMatrixCentered -----------------------
// -----------------------------------------------------------------------------
bool Camera::SetViewMatrixCentered(void)
{
	glLoadIdentity();

	if (useFreeFly)
	{
		gluLookAt(0, 0, 0,
			freeFlyDirection.x, -freeFlyDirection.y, freeFlyDirection.z,
			0, 1, 0);
	}
	else
	{
		glRotatef(rotationX, 1, 0, 0);
		glRotatef(rotationY+180.0f, 0, 1, 0);
	}

	return true;
}

// -----------------------------------------------------------------------------
// ----------------------- Camera::GetCameraMatrix -----------------------------
// -----------------------------------------------------------------------------
const Matrix4 Camera::GetCameraMatrix(void) const 
{
	return viewMatrix;
}

// -----------------------------------------------------------------------------
// -------------------- Camera::GetFixedCameraMatrix ---------------------------
// -----------------------------------------------------------------------------
const Matrix4 Camera::GetFixedCameraMatrix(void) const 
{
	return fixedMatrix;
}

// -----------------------------------------------------------------------------
// ----------------------- Camera::GetCameraPosition ---------------------------
// -----------------------------------------------------------------------------
const Vector3 Camera::GetCameraPosition(void) const 
{
	if (useFreeFly)
		return freeFlyPosition;
	else
	{
		Matrix4 invViewMatrix;
		invViewMatrix = fixedMatrix.Inverse();
		return invViewMatrix.GetTranslation();
	}
}

// -----------------------------------------------------------------------------
// ----------------------- Camera::GetCameraPosition ---------------------------
// -----------------------------------------------------------------------------
void Camera::SetCameraPosition(const Vector3& position)
{
	freeFlyPosition = position;
}

// -----------------------------------------------------------------------------
// ----------------------- Camera::GetCameraPosition ---------------------------
// -----------------------------------------------------------------------------
void Camera::SetCameraDirection(const Vector3& direction)
{
	freeFlyDirection = direction;

	float p, h;
	Math::DirectionToPitchHeading(freeFlyDirection, p, h);
	freeFlyRotationX = p*Math::RAD_TO_DEG;
	freeFlyRotationY = h*Math::RAD_TO_DEG;
}
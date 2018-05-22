#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <windows.h>
#include <GL/gl.h>

#include <stdio.h>

#include "../Level/Light.h"

#include "../Util/Math.h"

#include <GL/glut.h>

// -----------------------------------------------------------------------------
// ----------------------- Light::Light ----------------------------------------
// -----------------------------------------------------------------------------
Light::Light(void) :
	lightPosition(0.0f, 0.0f, 0.0f),
	lightLookAt(0.0f, 0.0f, 0.0f),
	nearPlane(1.0f),
	farPlane(1000.0f),
	fieldOfView(90.0f),
	lightRadiusWorld(0.5f),
	lightDiff(1.0f, 1.0f, 1.0f, 1.0f)
{
}

// -----------------------------------------------------------------------------
// ----------------------- Light::~Light ---------------------------------------
// -----------------------------------------------------------------------------
Light::~Light(void)
{

}

// -----------------------------------------------------------------------------
// ----------------------- Light::Init -----------------------------------------
// -----------------------------------------------------------------------------
void Light::Init(const Vector3 position, const Vector3 lookAt)
{
	lightPosition = position;
	lightLookAt = lookAt;

	lightProjection = Matrix4::Matrix4Perspective(fieldOfView*Math::DEG_TO_RAD, 1.0f, nearPlane, farPlane, false);
	lightView = Matrix4::Matrix4LookAt(lightPosition, lightLookAt, Vector3(0.0f, 1.0f, 0.0f));
}

// -----------------------------------------------------------------------------
// --------------------------- Light::Update -----------------------------------
// -----------------------------------------------------------------------------
void Light::Update(const Vector3 position, const Vector3 lookAt, float _fieldOfView, float aspectRation, float _nearPlane, float _farPlane)
{
	lightPosition = position;
	lightLookAt = lookAt;
	nearPlane = _nearPlane;
	farPlane = _farPlane;

	fieldOfView = _fieldOfView;

	lightProjection = Matrix4::Matrix4Perspective(fieldOfView*Math::DEG_TO_RAD, aspectRation, nearPlane, farPlane, false);
	lightView = Matrix4::Matrix4LookAt(position, lookAt, Vector3(0.0f, 1.0f, 0.0f));
}

// -----------------------------------------------------------------------------
// -------------------- Light::SetLightRadiusWorld -----------------------------
// -----------------------------------------------------------------------------
void Light::SetLightRadiusWorld(float size)
{
	lightRadiusWorld = size;
}

// -----------------------------------------------------------------------------
// ------------------------ Light::DebugRender ---------------------------------
// -----------------------------------------------------------------------------
void Light::DebugRender(void)
{
	float nearOffset = Math::Tan(fieldOfView*Math::DEG_TO_RAD*0.5f)*nearPlane;
	float farOffset = Math::Tan(fieldOfView*Math::DEG_TO_RAD*0.5f)*farPlane;

	Vector3 origin(0.0f, 0.0f, 0.0f);

	Matrix4 tmp;
	tmp = lightView.Inverse();

	Matrix4 rot;
	rot.BuildRotationX(Math::HALF_PI);

	rot *= tmp;

	glPushMatrix();
	glMultMatrixf(rot.entry);

	glColor4f(1.0f, 1.0f, 0.0f, 1.0f);

	glBegin(GL_LINES);
	{
		// top
		glVertex3f(origin.x+lightRadiusWorld, origin.y, origin.z+lightRadiusWorld);
		glVertex3f(origin.x-lightRadiusWorld, origin.y, origin.z+lightRadiusWorld);

		glVertex3f(origin.x-lightRadiusWorld, origin.y, origin.z+lightRadiusWorld);
		glVertex3f(origin.x-lightRadiusWorld, origin.y, origin.z-lightRadiusWorld);

		glVertex3f(origin.x-lightRadiusWorld, origin.y, origin.z-lightRadiusWorld);
		glVertex3f(origin.x+lightRadiusWorld, origin.y, origin.z-lightRadiusWorld);

		glVertex3f(origin.x+lightRadiusWorld, origin.y, origin.z-lightRadiusWorld);
		glVertex3f(origin.x+lightRadiusWorld, origin.y, origin.z+lightRadiusWorld);

		// near
		glVertex3f(origin.x+nearOffset, origin.y-nearPlane, origin.z+nearOffset);
		glVertex3f(origin.x-nearOffset, origin.y-nearPlane, origin.z+nearOffset);

		glVertex3f(origin.x-nearOffset, origin.y-nearPlane, origin.z+nearOffset);
		glVertex3f(origin.x-nearOffset, origin.y-nearPlane, origin.z-nearOffset);

		glVertex3f(origin.x-nearOffset, origin.y-nearPlane, origin.z-nearOffset);
		glVertex3f(origin.x+nearOffset, origin.y-nearPlane, origin.z-nearOffset);

		glVertex3f(origin.x+nearOffset, origin.y-nearPlane, origin.z-nearOffset);
		glVertex3f(origin.x+nearOffset, origin.y-nearPlane, origin.z+nearOffset);

		//////////////////////////////////////////////////////////////////////////

		// far
		glVertex3f(origin.x+farOffset, origin.y-farPlane, origin.z+farOffset);
		glVertex3f(origin.x-farOffset, origin.y-farPlane, origin.z+farOffset);

		glVertex3f(origin.x-farOffset, origin.y-farPlane, origin.z+farOffset);
		glVertex3f(origin.x-farOffset, origin.y-farPlane, origin.z-farOffset);

		glVertex3f(origin.x-farOffset, origin.y-farPlane, origin.z-farOffset);
		glVertex3f(origin.x+farOffset, origin.y-farPlane, origin.z-farOffset);

		glVertex3f(origin.x+farOffset, origin.y-farPlane, origin.z-farOffset);
		glVertex3f(origin.x+farOffset, origin.y-farPlane, origin.z+farOffset);

		// connection
		glVertex3f(origin.x, origin.y, origin.z);
		glVertex3f(origin.x+farOffset, origin.y-farPlane, origin.z+farOffset);

		glVertex3f(origin.x, origin.y, origin.z);
		glVertex3f(origin.x-farOffset, origin.y-farPlane, origin.z+farOffset);

		glVertex3f(origin.x, origin.y, origin.z);
		glVertex3f(origin.x-farOffset, origin.y-farPlane, origin.z-farOffset);

		glVertex3f(origin.x, origin.y, origin.z);
		glVertex3f(origin.x+farOffset, origin.y-farPlane, origin.z-farOffset);
	}
	glEnd();

	glPopMatrix();
}
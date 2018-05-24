#include "Camera.h"

Camera::Camera()
	: up(Vector3(0, 1, 0)), right(Vector3(0, 0, -1)), facing(Vector3(1, 0, 0)), position(Vector3(-1307.005f, 262.526398f, -38.6900518f)), camSpeed(205.0f)
{}

Camera::~Camera()
{

}

void Camera::Update()
{
	camData.viewMat = Matrix::CreateLookAt(position, position + facing, up);
	camData.viewProjMat = camData.viewMat * camData.projMat;
}

void Camera::SetLens(float fov, float nearPlane, float farPlane, int width, int height)
{
	camData.projMat = Matrix::CreatePerspectiveFieldOfView(fov, (float)width/(float)height, nearPlane, farPlane);
}

void Camera::Pitch(float angle)
{
	Quaternion rot = Quaternion::CreateFromAxisAngle(right, angle);
	facing = Vector3::Transform(facing, rot);
	up = Vector3::Transform(up, rot);
}

void Camera::Yaw(float angle)
{
	Quaternion rot = Quaternion::CreateFromAxisAngle(Vector3(0, 1, 0), angle);
	facing = Vector3::Transform(facing, rot);
	up = Vector3::Transform(up, rot);
	right = Vector3::Transform(right, rot);
}

void Camera::MoveForward(float dt, float speed)
{
	position += camSpeed * dt * facing * speed;
}

void Camera::MoveBackward(float dt, float speed)
{
	position -= camSpeed * dt * facing * speed;
}

void Camera::MoveUp(float dt, float speed)
{
	position += camSpeed * dt * up * speed;
}

void Camera::MoveDown(float dt, float speed)
{
	position -= camSpeed * dt * up * speed;
}

void Camera::StrafeLeft(float dt, float speed)
{
	position += camSpeed * dt * right * speed;
}

void Camera::StrafeRight(float dt, float speed)
{
	position -= camSpeed * dt * right * speed;
}
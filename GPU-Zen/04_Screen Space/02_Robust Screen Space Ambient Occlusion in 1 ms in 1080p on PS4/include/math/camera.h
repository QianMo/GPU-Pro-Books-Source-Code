#pragma once


#include "vector.h"
#include "matrix.h"


namespace NMath
{
	class Camera
	{
	public:
		Camera()
		{
			horizontalAngle = 0.0f;
			verticalAngle = 0.0f;
			distanceFromEyeToAt = 1.0f;
		}

		void UpdateFixed(const Vector3& eye, const Vector3& at, const Vector3& up = VectorCustom(0.0f, 1.0f, 0.0f));
		void UpdateFree(const Vector3& eye, const Vector3& up = VectorCustom(0.0f, 1.0f, 0.0f));
		void UpdateFocused(const Vector3& at, const Vector3& up = VectorCustom(0.0f, 1.0f, 0.0f));

	public:
		float horizontalAngle, verticalAngle;
		float distanceFromEyeToAt;

	public: // readonly
		Vector3 eye, at, up;
		Vector3 forwardVector, rightVector, upVector;
	};
}


inline void NMath::Camera::UpdateFixed(const Vector3& eye, const Vector3& at, const Vector3& up)
{
	forwardVector = Normalize(at - eye);

	this->eye = eye;
	this->at = at;
	this->up = up;
}


inline void NMath::Camera::UpdateFree(const Vector3& eye, const Vector3& up)
{
	Matrix transformMatrix = MatrixRotateX(verticalAngle) * MatrixRotateY(horizontalAngle);

	forwardVector = -Vector3EZ * transformMatrix;
	rightVector = Vector3EX * transformMatrix;
	upVector = Cross(rightVector, forwardVector);

	this->eye = eye;
	this->at = eye + forwardVector;
	this->up = up;
}


inline void NMath::Camera::UpdateFocused(const Vector3& at, const Vector3& up)
{
	Matrix transformMatrix = MatrixRotateX(verticalAngle) * MatrixRotateY(horizontalAngle);

	forwardVector = -Vector3EZ * transformMatrix;
	rightVector = Vector3EX * transformMatrix;
	upVector = Cross(rightVector, forwardVector);

	this->eye = at - forwardVector*distanceFromEyeToAt;
	this->at = at;
	this->up = up;
}

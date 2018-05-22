#ifndef __CAMERA__H__
#define __CAMERA__H__

#include "../Util/Matrix4.h"

#include <vector>

class Vector3;


// -----------------------------------------------------------------------------
/// Camera
// -----------------------------------------------------------------------------
/// \ingroup 
/// 
/// 
// -----------------------------------------------------------------------------
class Camera
{
public:
	Camera(void);
	~Camera(void);

	/// Inits the camera
	bool Init(void);

	/// Exit the camera
	void Exit(void);

	/// Updates the camera
	bool Update(float deltaTime);

	/// Sets the current view matrix into the render pipeline
	bool SetViewMatrix(void);

	/// Sets the current view matrix into the render pipeline (camera centered in origin)
	bool SetViewMatrixCentered(void);

	/// Returns the camera state as a matrix
	const Matrix4 GetCameraMatrix(void) const;

	/// This camera matrix can be fixed (no update) for testing
	const Matrix4 GetFixedCameraMatrix(void) const;

	/// Returns the position of the camera
	const Vector3 GetCameraPosition(void) const;

	void SetCameraPosition(const Vector3& position);
	void SetCameraDirection(const Vector3& direction);

	/// Returns the current x rotation of the camera
	float GetRotationX(void) { return rotationX; }
	/// Returns the current y rotation of the camera
	float GetRotationY(void) { return rotationY; }

private:
	Vector3 freeFlyPosition;
	Vector3 freeFlyDirection;
	float freeFlyRotationX;
	float freeFlyRotationY;

	Matrix4 viewMatrix;

	bool useFreeFly;

	/// The distance of the camera from the center of the world
	float	distance;
	/// X rotation
	float	rotationX;
	/// Y rotation
	float	rotationY;

	bool updateFixedMatrix;
	Matrix4 fixedMatrix;
};

#endif
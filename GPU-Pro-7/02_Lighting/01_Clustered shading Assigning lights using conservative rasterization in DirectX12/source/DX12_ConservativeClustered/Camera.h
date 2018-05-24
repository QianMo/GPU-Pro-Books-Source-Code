#pragma once

#include <SimpleMath.h>

using namespace DirectX::SimpleMath;

struct CameraData
{
	Matrix viewProjMat;
	Matrix viewMat;
	Matrix projMat;
};

class Camera
{
public:

	Camera();
	~Camera();

	void Update();
	void SetLens(float fov, float nearPlane, float farPlane, int width, int height);
	void Pitch(float angle);
	void Yaw(float angle);
	void MoveForward(float dt, float speed = 1.0f);
	void MoveBackward(float dt, float speed = 1.0f);
	void MoveUp(float dt, float speed = 1.0f);
	void MoveDown(float dt, float speed = 1.0f);
	void StrafeLeft(float dt, float speed = 1.0f);
	void StrafeRight(float dt, float speed = 1.0f);

	const CameraData& GetCamData() { return camData; }

	Vector3 up;
	Vector3 right;
	Vector3 facing;
	Vector3 position;
	float camSpeed;

private:

	CameraData camData;
};
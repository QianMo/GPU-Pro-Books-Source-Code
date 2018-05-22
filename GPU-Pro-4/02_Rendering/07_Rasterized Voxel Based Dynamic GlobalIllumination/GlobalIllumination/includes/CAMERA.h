#ifndef CAMERA_H
#define CAMERA_H

class DX11_UNIFORM_BUFFER;

// CAMERA
//  Simple first person camera.
class CAMERA
{
public:
  CAMERA()
	{
		nearClipDistance = 0.0f;
		farClipDistance = 0.0f;
		nearFarClipDistance = 0.0f;
		float fovy = 0.0f;
		halfFarWidth = 0.0f;
		halfFarHeight = 0.0f;
		aspectRatio = 0.0f;
		uniformBuffer = NULL;
	}

	bool Init(float fovy,float nearClipDistance,float farClipDistance);

	void Update(const VECTOR3D &position,const VECTOR3D &rotation);

	DX11_UNIFORM_BUFFER* GetUniformBuffer() const
	{
		return uniformBuffer;
	}

	MATRIX4X4 GetViewMatrix() const
	{
		return viewMatrix;
	}

  MATRIX4X4 GetInvTransposeViewMatrix() const
	{
		return invTransposeViewMatrix;
	}

 	MATRIX4X4 GetProjMatrix() const
	{
		return projMatrix;
	}

	MATRIX4X4 GetInvProjMatrix() const
	{
		return invProjMatrix;
	}

	MATRIX4X4 GetViewProjMatrix() const
	{
		return viewProjMatrix;
	}

	VECTOR3D GetPosition() const
	{
		return position;
	}
	
	VECTOR3D GetRotation() const
	{
		return rotation;
	}

  VECTOR3D GetDirection() const
	{
		return direction;
	}

	float GetFovy() const
	{
		return fovy;
	}

	float GetAspectRatio() const
	{
		return aspectRatio;
	}

	float GetNearClipDistance() const
	{
		return nearClipDistance;
	}

  float GetFarClipDistance() const
	{
		return farClipDistance;
	}

	float GetNearFarClipDistance() const
	{
		return nearFarClipDistance;
	}

private:
  void UpdateUniformBuffer();

	// data for camera uniform-buffer
	MATRIX4X4 viewMatrix;
	MATRIX4X4 invTransposeViewMatrix;
	MATRIX4X4 projMatrix;
	MATRIX4X4 viewProjMatrix;
	VECTOR4D frustumRays[4];
	VECTOR3D position;
	float nearClipDistance;
	float farClipDistance;
	float nearFarClipDistance;
	
	VECTOR3D rotation;
	VECTOR3D direction;
	float fovy;
	float halfFarWidth,halfFarHeight;
	float aspectRatio;
	MATRIX4X4 invProjMatrix;
	DX11_UNIFORM_BUFFER *uniformBuffer;

};

#endif

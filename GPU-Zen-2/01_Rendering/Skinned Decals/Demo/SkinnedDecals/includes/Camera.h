#ifndef CAMERA_H
#define CAMERA_H

#include <Frustum.h>

class DX12_Buffer;

// Camera
//
class Camera
{
public:
  struct CameraConstData
  {
    CameraConstData():
      aspectRatio(0.0f),
      nearClipDistance(0.0f),
      farClipDistance(0.0f),
      nearFarClipDistance(0.0f)
    {
    }

    Matrix4 viewMatrix;
    Matrix4 projMatrix;
    Matrix4 viewProjMatrix;
    Matrix4 invViewProjMatrix;
    Vector3 position;
    float aspectRatio;
    float nearClipDistance;
    float farClipDistance;
    float nearFarClipDistance;
  };

  Camera():
    fovy(0.0f),
    cameraCB(nullptr)
  {
  }

  bool Init(float fovy, float aspectRatio, float nearClipDistance, float farClipDistance);

  void Update(const Vector3 &position, const Vector3 &rotation);

  DX12_Buffer* GetConstantBuffer() const
  {
    return cameraCB;
  }

  Matrix4 GetViewMatrix() const
  {
    return cameraConstData.viewMatrix;
  }

  Matrix4 GetProjMatrix() const
  {
    return cameraConstData.projMatrix;
  }

  Matrix4 GetInvProjMatrix() const
  {
    return invProjMatrix;
  }

  Matrix4 GetViewProjMatrix() const
  {
    return cameraConstData.viewProjMatrix;
  }

  Matrix4 GetInvViewProjMatrix() const
  {
    return cameraConstData.invViewProjMatrix;
  }

  Vector3 GetPosition() const
  {
    return cameraConstData.position;
  }
  
  Vector3 GetRotation() const
  {
    return rotation;
  }
  
  Vector3 GetDirection() const
  {
    return direction;
  }

  Vector3 GetUpVector() const
	{
		return up;
	}

	Vector3 GetRightVector() const
	{
		return right;
	}

  float GetFovy() const
  {
    return fovy;
  }

  float GetAspectRatio() const
  {
    return cameraConstData.aspectRatio;
  }

  float GetNearClipDistance() const
  {
    return cameraConstData.nearClipDistance;
  }

  float GetFarClipDistance() const
  {
    return cameraConstData.farClipDistance;
  }

  float GetNearFarClipDistance() const
  {
    return cameraConstData.nearFarClipDistance;
  }

  const Frustum& GetFrustum() const
  {
    return frustum;
  }

private:
  void UpdateConstantBuffer();

  CameraConstData cameraConstData;
  
  Vector3 rotation;
  Vector3 direction, up, right;
  float fovy;
  Matrix4 invProjMatrix;
  Frustum frustum;
  DX12_Buffer *cameraCB;

};

#endif

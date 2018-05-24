#ifndef CAMERA_H
#define CAMERA_H

#include <Frustum.h>

class OGL_UniformBuffer;

// Camera
//
// Simple camera.
class Camera
{
public:
  struct BufferData
  {
    BufferData():
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
    uniformBuffer(NULL)
  {
  }

  bool Init(float fovy, float aspectRatio, float nearClipDistance, float farClipDistance);

  void Update(const Vector3 &position, const Vector3 &rotation);

  OGL_UniformBuffer* GetUniformBuffer() const
  {
    return uniformBuffer;
  }

  Matrix4 GetViewMatrix() const
  {
    return bufferData.viewMatrix;
  }

  Matrix4 GetProjMatrix() const
  {
    return bufferData.projMatrix;
  }

  Matrix4 GetInvProjMatrix() const
  {
    return invProjMatrix;
  }

  Matrix4 GetViewProjMatrix() const
  {
    return bufferData.viewProjMatrix;
  }


  Matrix4 GetInvViewProjMatrix() const
  {
    return bufferData.invViewProjMatrix;
  }

  Vector3 GetPosition() const
  {
    return bufferData.position;
  }
  
  Vector3 GetRotation() const
  {
    return rotation;
  }

  float GetFovy() const
  {
    return fovy;
  }

  float GetAspectRatio() const
  {
    return bufferData.aspectRatio;
  }

  float GetNearClipDistance() const
  {
    return bufferData.nearClipDistance;
  }

  float GetFarClipDistance() const
  {
    return bufferData.farClipDistance;
  }

  float GetNearFarClipDistance() const
  {
    return bufferData.nearFarClipDistance;
  }

  const Frustum& GetFrustum() const
  {
    return frustum;
  }

private:
  void UpdateUniformBuffer();

  // data for camera uniform-buffer
  BufferData bufferData;
  
  Vector3 rotation;
  float fovy;
  Matrix4 invProjMatrix;
  Frustum frustum;
  OGL_UniformBuffer *uniformBuffer;

};

#endif

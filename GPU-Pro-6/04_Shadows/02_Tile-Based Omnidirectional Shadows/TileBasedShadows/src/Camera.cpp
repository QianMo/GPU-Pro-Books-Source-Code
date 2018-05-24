#include <stdafx.h>
#include <Demo.h>
#include <Camera.h>

bool Camera::Init(float fovy, float aspectRatio, float nearClipDistance, float farClipDistance)
{
  this->fovy = fovy;	
  bufferData.aspectRatio = aspectRatio;
  if((nearClipDistance <= 0.0f) || (farClipDistance <= 0.0f))
    return false;
  bufferData.nearClipDistance = nearClipDistance;
  bufferData.farClipDistance = farClipDistance;
  bufferData.nearFarClipDistance = farClipDistance-nearClipDistance;
  bufferData.projMatrix.SetPerspective(fovy, aspectRatio, nearClipDistance, farClipDistance);	
  invProjMatrix = bufferData.projMatrix.GetInverse();	

  uniformBuffer = Demo::renderer->CreateUniformBuffer(sizeof(BufferData));
  if(!uniformBuffer)
    return false;
  
  UpdateUniformBuffer();
  
  return true;
}

void Camera::UpdateUniformBuffer()
{
  uniformBuffer->Update(&bufferData);
}

void Camera::Update(const Vector3 &position, const Vector3 &rotation)
{
  bufferData.position = position;
  this->rotation = rotation;
  Matrix4 xRotMatrix, yRotMatrix, zRotMatrix, transMatrix, rotMatrix;
  xRotMatrix.SetRotationY(-rotation.x);
  yRotMatrix.SetRotationX(rotation.y);
  zRotMatrix.SetRotationZ(rotation.z);
  transMatrix.SetTranslation(-position);
  rotMatrix = zRotMatrix*yRotMatrix*xRotMatrix;
  bufferData.viewMatrix = rotMatrix*transMatrix;
  bufferData.viewProjMatrix = bufferData.projMatrix*bufferData.viewMatrix;
  bufferData.invViewProjMatrix = bufferData.viewProjMatrix.GetInverse();

  frustum.Update(bufferData.viewProjMatrix);

  UpdateUniformBuffer();
}






#include <stdafx.h>
#include <Demo.h>
#include <Camera.h>

bool Camera::Init(float fovy, float aspectRatio, float nearClipDistance, float farClipDistance)
{
  this->fovy = fovy;	
  cameraConstData.aspectRatio = aspectRatio;
  if((nearClipDistance <= 0.0f) || (farClipDistance <= 0.0f))
    return false;
  cameraConstData.nearClipDistance = nearClipDistance;
  cameraConstData.farClipDistance = farClipDistance;
  cameraConstData.nearFarClipDistance = farClipDistance-nearClipDistance;
  cameraConstData.projMatrix.SetPerspective(fovy, aspectRatio, nearClipDistance, farClipDistance);	
  invProjMatrix = cameraConstData.projMatrix.GetInverse();	

  BufferDesc bufferDesc;
  bufferDesc.bufferType = CONSTANT_BUFFER;
  bufferDesc.elementSize = sizeof(CameraConstData);
  bufferDesc.numElements = 1;
  bufferDesc.flags = CPU_WRITE_BUFFER_FLAG | DYNAMIC_BUFFER_FLAG;
  cameraCB = Demo::renderer->CreateBuffer(bufferDesc, "Camera");
  if(!cameraCB)
    return false;
  
  UpdateConstantBuffer();
  
  return true;
}

void Camera::UpdateConstantBuffer()
{
  cameraCB->Update(&cameraConstData, 1);
}

void Camera::Update(const Vector3 &position, const Vector3 &rotation)
{
  cameraConstData.position = position;
  this->rotation = rotation;
  Matrix4 xRotMatrix, yRotMatrix, zRotMatrix, transMatrix, rotMatrix;
  xRotMatrix.SetRotationY(-rotation.x);
  yRotMatrix.SetRotationX(rotation.y);
  zRotMatrix.SetRotationZ(rotation.z);
  transMatrix.SetTranslation(-position);
  rotMatrix = zRotMatrix * yRotMatrix * xRotMatrix;
  cameraConstData.viewMatrix = rotMatrix * transMatrix;
  cameraConstData.viewProjMatrix = cameraConstData.projMatrix * cameraConstData.viewMatrix;
  cameraConstData.invViewProjMatrix = cameraConstData.viewProjMatrix.GetInverse();

  direction.Set(-cameraConstData.viewMatrix.entries[2], -cameraConstData.viewMatrix.entries[6], -cameraConstData.viewMatrix.entries[10]);
  direction.Normalize();
  up.Set(cameraConstData.viewMatrix.entries[1], cameraConstData.viewMatrix.entries[5], cameraConstData.viewMatrix.entries[9]);
	up.Normalize();
	right.Set(cameraConstData.viewMatrix.entries[0], cameraConstData.viewMatrix.entries[4], cameraConstData.viewMatrix.entries[8]);
	right.Normalize();

  frustum.Update(cameraConstData.viewProjMatrix);

  UpdateConstantBuffer();
}






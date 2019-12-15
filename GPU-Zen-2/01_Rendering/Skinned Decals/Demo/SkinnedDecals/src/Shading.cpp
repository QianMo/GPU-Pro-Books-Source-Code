#include <stdafx.h>
#include <Demo.h>
#include <Shading.h>

Shading::~Shading()
{
  SAFE_DELETE(directionalLight);
}

bool Shading::Create()
{	  
  {
    // create constant buffer for lighting related info
    BufferDesc bufferDesc;
    bufferDesc.bufferType = CONSTANT_BUFFER;
    bufferDesc.elementSize = sizeof(LightingConstData);
    bufferDesc.numElements = 1;
    bufferDesc.flags = CPU_WRITE_BUFFER_FLAG | DYNAMIC_BUFFER_FLAG;
    lightingCB = Demo::renderer->CreateBuffer(bufferDesc, "Lighting info");
    if(!lightingCB)
      return false;
  }

  return true;
}

DirectionalLight* Shading::CreateDirectionalLight(const Vector3 &direction, const Color &color, float intensity)
{
  assert(directionalLight == nullptr); // only one directional light supported

  directionalLight = new DirectionalLight;
  if(!directionalLight)
    return nullptr;
  if(!directionalLight->Create(direction, color, intensity))
  {
    SAFE_DELETE(directionalLight);
    return nullptr;
  }
  return directionalLight;
}

void Shading::UpdateLights()
{
  if(directionalLight)
  {
    memcpy(&lightingConstData.dirLightDirection, &directionalLight->lightBufferData, sizeof(DirectionalLight::LightBufferData));

    // Update constant buffer that stores lighting related information.
    lightingCB->Update(&lightingConstData, 1);
  }
}

void Shading::Execute()
{
  if(!active)
    return;
  
  UpdateLights();
}


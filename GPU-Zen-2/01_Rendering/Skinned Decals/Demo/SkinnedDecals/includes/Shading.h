#ifndef SHADING_H
#define SHADING_H

#include <IPostProcessor.h>
#include <DirectionalLight.h>

class DX12_PipelineState;
class DX12_RenderTarget;
class DX12_Buffer;

// Shading
//
class Shading: public IPostProcessor
{
public: 
  struct LightingConstData
  {
    Vector4 dirLightDirection;                                      
    Color dirLightColor;
  };

  Shading():
    directionalLight(nullptr),
    lightingCB(nullptr)
  {
    strcpy(name, "Shading");
  }

  virtual ~Shading();

  virtual bool Create() override;

  virtual DX12_RenderTarget* GetOutputRT() const override
  {
    return nullptr;
  }

  virtual void Execute() override;

  DirectionalLight* CreateDirectionalLight(const Vector3 &direction, const Color &color, float intensity);

  DirectionalLight* GetDirectionalLight() const
  {
    return directionalLight;
  }

  DX12_Buffer* GetLightingCB() const
  {
    return lightingCB;
  }

private:
  void UpdateLights();

  DirectionalLight *directionalLight;
  LightingConstData lightingConstData;

  DX12_Buffer *lightingCB;

};

#endif
#ifndef __RENDERABLE_H
#define __RENDERABLE_H

class DeviceContext11;

class Renderable
{
public:
  virtual void DrawPrePass(DeviceContext11&) = 0;
  virtual void DrawShadowMap(DeviceContext11&) = 0;
  virtual void DrawCubeShadowMap(DeviceContext11&) = 0;
  virtual void DrawCubeShadowMapArray(DeviceContext11&) = 0;
  virtual void DrawParabolicShadowMap(DeviceContext11&) = 0;
  virtual void DrawASMLayerShadowMap(DeviceContext11&) = 0;
};

#endif //#ifndef __RENDERABLE_H

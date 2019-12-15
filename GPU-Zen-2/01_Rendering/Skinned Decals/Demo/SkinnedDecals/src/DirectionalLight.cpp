#include <stdafx.h>
#include <Demo.h>
#include <Shading.h>
#include <DirectionalLight.h>

bool DirectionalLight::Create(const Vector3 &direction, const Color &color, float intensity)
{
  this->color = color;
  this->intensity = intensity;

  lightBufferData.direction = direction.GetNormalized();
  Color linearColor = color;
  linearColor.ConvertSrgbToLinear();
  lightBufferData.color.Set(linearColor.r * intensity, linearColor.g * intensity, linearColor.b * intensity);

  return true;
}
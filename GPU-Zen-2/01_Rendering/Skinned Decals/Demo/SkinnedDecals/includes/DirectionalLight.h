#ifndef DIRECTIONAL_LIGHT_H
#define DIRECTIONAL_LIGHT_H

// DirectionalLight
//
class DirectionalLight
{
public:
  friend class Shading;

  struct LightBufferData
  {
    Vector4 direction;
    Color color;
  };

  DirectionalLight():
    intensity(1.0f)
	{
	}

	bool Create(const Vector3 &direction, const Color &color, float intensity);

	void SetDirection(const Vector3 &direction)
	{
		lightBufferData.direction = direction.GetNormalized();
	}

	Vector3 GetDirection() const
	{
		return Vector3(lightBufferData.direction.x, lightBufferData.direction.y, lightBufferData.direction.z);
	}

  void SetColor(const Color &color)
  {
    if(this->color != color)
    {
      Color linearColor = color;
      linearColor.ConvertSrgbToLinear();
      lightBufferData.color.Set(linearColor.r * intensity, linearColor.g * intensity, linearColor.b * intensity);
      this->color = color;
    }
  }

  Color GetColor() const
  {
    return color;
  }

  void SetIntensity(float intensity)
  {
    this->intensity = intensity;
  }

  float GetIntensity() const
  {
    return intensity;
  }

private:
  LightBufferData lightBufferData;

  Color color;
  float intensity;
	
};

#endif

#ifndef PLANE_H
#define PLANE_H

enum planeSides
{ 
  ON_PLANE=0,
  IN_FRONT_OF_PLANE,
  BEHIND_PLANE
};

// Plane
//
// Plane in 3D-space.
class Plane
{
public:
  Plane():
    intercept(0.0f)
  {
  }

  Plane(const Vector3 &normal, float intercept)
  {
    this->normal = normal;
    this->intercept = intercept;
  }

  void SetFromPoints(const Vector3 &p0, const Vector3 &p1, const Vector3 &p2)
  {
    normal = (p1 - p0).CrossProduct(p2 - p0);
    normal.Normalize();
    CalculateIntercept(p0);
  }

  void Normalize()
  {
    const float length = normal.GetLength();
    if(length == 0.0f)
      return;
    const float inv = 1.0f / length;
    normal = normal * inv;
    intercept = intercept * inv;
  }

  void CalculateIntercept(const Vector3 &pointOnPlane)
  { 
    intercept = -normal.DotProduct(pointOnPlane); 
  }

  float GetDistance(const Vector3 &point) const
  {
    return (normal.DotProduct(point) + intercept);
  }

  planeSides ClassifyPoint(const Vector3 &point) const
  {
    const float distance = GetDistance(point);
    if(distance > EPSILON)
      return IN_FRONT_OF_PLANE;
    else if(distance < -EPSILON)
      return BEHIND_PLANE;
    else
      return ON_PLANE;
  }

  bool IsFacing(const Vector3 &point) const
  {
    return (GetDistance(point) > EPSILON);
  }

  Vector3 normal;
  float intercept;

};

#endif
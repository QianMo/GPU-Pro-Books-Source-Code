#ifndef AABB_H
#define AABB_H

// Aabb
//
// Axis-aligned bounding-box.
class Aabb
{
public:
  // sets mins to FLT_MAX and maxes to -FLT_MAX
  void Clear()
  {
    mins.SetMax();
    maxes.SetMin();
  }

  bool IsValid() const
  {
    return ((maxes.x >= mins.x) && (maxes.y >= mins.y) && (maxes.z >= mins.z));
  }

  void Inflate(const Vector3 &point);

  void Inflate(const Aabb &box);

  void Expand(const Vector3 &value)
  {
    mins -= value;
    maxes += value;
  }

  Aabb Lerp(const Aabb &box, float factor) const;

  void GetTransformedAabb(Aabb &transformedBox, const Matrix4 &transformMatrix) const;

  bool IntersectRay(const Vector3 &rayOrigin, const Vector3 &rayDir, float &minDistance, float &maxDistance) const;

  void GetCorners(Vector3 *corners) const;

  Vector3 GetCenter() const
  {
    return ((maxes + mins) * 0.5f);
  }

  Vector3 GetExtents() const
  {
    return (maxes - mins);
  }

  Vector3 mins, maxes; 
  
};

#endif 

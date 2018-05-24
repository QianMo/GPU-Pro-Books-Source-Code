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

  void Inflate(const Vector3 &point);

  Vector3 mins, maxes; 
  
};

#endif 

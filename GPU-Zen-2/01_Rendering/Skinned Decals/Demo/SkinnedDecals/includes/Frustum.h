#ifndef FRUSTUM_H
#define FRUSTUM_H

#include <Aabb.h>

// 6 plane-sides of frustum
enum frustumPlanes
{
  LEFT_FRUSTUM_PLANE=0,
  RIGHT_FRUSTUM_PLANE,
  TOP_FRUSTUM_PLANE,
  BOTTOM_FRUSTUM_PLANE,
  NEAR_FRUSTUM_PLANE,
  FAR_FRUSTUM_PLANE
};

// Frustum
//
// 3D view-frustum. 
class Frustum
{
public:
  // updates frustum with view-projection matrix
  void Update(const Matrix4 &viewProj);  

  // checks, if specified axis-aligned bounding box is inside frustum
  bool IsAabbInside(const Aabb &box) const;

  // checks, if specified sphere is inside frustum
  bool IsSphereInside(const Vector3 &position, float radius) const;

private:
  Plane planes[6]; // 6 planes of frustum

};

#endif


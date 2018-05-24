#ifndef FRUSTUM_H
#define FRUSTUM_H

#include <Aabb.h>

// 6 plane-sides of frustum
enum frustumPlanes
{
  FRUSTUM_LEFT_PLANE=0,
  FRUSTUM_RIGHT_PLANE,
  FRUSTUM_TOP_PLANE,
  FRUSTUM_BOTTOM_PLANE,
  FRUSTUM_NEAR_PLANE,
  FRUSTUM_FAR_PLANE
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


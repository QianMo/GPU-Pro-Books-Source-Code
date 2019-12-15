#include <stdafx.h>
#include <Frustum.h>  

void Frustum::Update(const Matrix4 &viewProj)
{
  planes[RIGHT_FRUSTUM_PLANE].normal.x = viewProj.entries[3] - viewProj.entries[0];
  planes[RIGHT_FRUSTUM_PLANE].normal.y = viewProj.entries[7] - viewProj.entries[4];
  planes[RIGHT_FRUSTUM_PLANE].normal.z = viewProj.entries[11] - viewProj.entries[8];
  planes[RIGHT_FRUSTUM_PLANE].intercept = viewProj.entries[15] - viewProj.entries[12];
  planes[LEFT_FRUSTUM_PLANE].normal.x = viewProj.entries[3] + viewProj.entries[0];
  planes[LEFT_FRUSTUM_PLANE].normal.y = viewProj.entries[7] + viewProj.entries[4];
  planes[LEFT_FRUSTUM_PLANE].normal.z = viewProj.entries[11] + viewProj.entries[8];
  planes[LEFT_FRUSTUM_PLANE].intercept = viewProj.entries[15] + viewProj.entries[12];
  planes[BOTTOM_FRUSTUM_PLANE].normal.x = viewProj.entries[3] + viewProj.entries[1];
  planes[BOTTOM_FRUSTUM_PLANE].normal.y = viewProj.entries[7] + viewProj.entries[5];
  planes[BOTTOM_FRUSTUM_PLANE].normal.z = viewProj.entries[11] + viewProj.entries[9];
  planes[BOTTOM_FRUSTUM_PLANE].intercept = viewProj.entries[15] + viewProj.entries[13];
  planes[TOP_FRUSTUM_PLANE].normal.x = viewProj.entries[3] - viewProj.entries[1];
  planes[TOP_FRUSTUM_PLANE].normal.y = viewProj.entries[7] - viewProj.entries[5];
  planes[TOP_FRUSTUM_PLANE].normal.z = viewProj.entries[11] - viewProj.entries[9];
  planes[TOP_FRUSTUM_PLANE].intercept = viewProj.entries[15] - viewProj.entries[13];
  planes[FAR_FRUSTUM_PLANE].normal.x = viewProj.entries[3] - viewProj.entries[2];
  planes[FAR_FRUSTUM_PLANE].normal.y = viewProj.entries[7] - viewProj.entries[6];
  planes[FAR_FRUSTUM_PLANE].normal.z = viewProj.entries[11] - viewProj.entries[10];
  planes[FAR_FRUSTUM_PLANE].intercept = viewProj.entries[15] - viewProj.entries[14];
  planes[NEAR_FRUSTUM_PLANE].normal.x = viewProj.entries[3] + viewProj.entries[2];
  planes[NEAR_FRUSTUM_PLANE].normal.y = viewProj.entries[7] + viewProj.entries[6];
  planes[NEAR_FRUSTUM_PLANE].normal.z = viewProj.entries[11] + viewProj.entries[10];
  planes[NEAR_FRUSTUM_PLANE].intercept = viewProj.entries[15] + viewProj.entries[14];
  for(UINT i=0; i<6; i++)
    planes[i].Normalize();
}

bool Frustum::IsAabbInside(const Aabb &box) const
{
  for(UINT i=0; i<6; i++)
  {
    Vector3 closestCorner;
    closestCorner.x = (planes[i].normal.x > 0.0f) ? box.maxes.x : box.mins.x;
    closestCorner.y = (planes[i].normal.y > 0.0f) ? box.maxes.y : box.mins.y;
    closestCorner.z = (planes[i].normal.z > 0.0f) ? box.maxes.z : box.mins.z;
    if(planes[i].ClassifyPoint(closestCorner) == BEHIND_PLANE)
      return false;
  }
  return true;
}

bool Frustum::IsSphereInside(const Vector3 &position, float radius) const
{
  for(UINT i=0; i<6; i++)
  {
    if(planes[i].GetDistance(position) < -radius)
      return false;
  }
  return true;
}


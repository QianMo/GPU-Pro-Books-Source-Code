#include <stdafx.h>
#include <Frustum.h>  

void Frustum::Update(const Matrix4 &viewProj)
{
  planes[FRUSTUM_RIGHT_PLANE].normal.x = viewProj.entries[3]-viewProj.entries[0];
  planes[FRUSTUM_RIGHT_PLANE].normal.y = viewProj.entries[7]-viewProj.entries[4];
  planes[FRUSTUM_RIGHT_PLANE].normal.z = viewProj.entries[11]-viewProj.entries[8];
  planes[FRUSTUM_RIGHT_PLANE].intercept = viewProj.entries[15]-viewProj.entries[12];
  planes[FRUSTUM_LEFT_PLANE].normal.x = viewProj.entries[3]+viewProj.entries[0];
  planes[FRUSTUM_LEFT_PLANE].normal.y = viewProj.entries[7]+viewProj.entries[4];
  planes[FRUSTUM_LEFT_PLANE].normal.z = viewProj.entries[11]+viewProj.entries[8];
  planes[FRUSTUM_LEFT_PLANE].intercept = viewProj.entries[15]+viewProj.entries[12];
  planes[FRUSTUM_BOTTOM_PLANE].normal.x = viewProj.entries[3]+viewProj.entries[1];
  planes[FRUSTUM_BOTTOM_PLANE].normal.y = viewProj.entries[7]+viewProj.entries[5];
  planes[FRUSTUM_BOTTOM_PLANE].normal.z = viewProj.entries[11]+viewProj.entries[9];
  planes[FRUSTUM_BOTTOM_PLANE].intercept = viewProj.entries[15]+viewProj.entries[13];
  planes[FRUSTUM_TOP_PLANE].normal.x = viewProj.entries[3]-viewProj.entries[1];
  planes[FRUSTUM_TOP_PLANE].normal.y = viewProj.entries[7]-viewProj.entries[5];
  planes[FRUSTUM_TOP_PLANE].normal.z = viewProj.entries[11]-viewProj.entries[9];
  planes[FRUSTUM_TOP_PLANE].intercept = viewProj.entries[15]-viewProj.entries[13];
  planes[FRUSTUM_FAR_PLANE].normal.x = viewProj.entries[3]-viewProj.entries[2];
  planes[FRUSTUM_FAR_PLANE].normal.y = viewProj.entries[7]-viewProj.entries[6];
  planes[FRUSTUM_FAR_PLANE].normal.z = viewProj.entries[11]-viewProj.entries[10];
  planes[FRUSTUM_FAR_PLANE].intercept = viewProj.entries[15]-viewProj.entries[14];
  planes[FRUSTUM_NEAR_PLANE].normal.x = viewProj.entries[3]+viewProj.entries[2];
  planes[FRUSTUM_NEAR_PLANE].normal.y = viewProj.entries[7]+viewProj.entries[6];
  planes[FRUSTUM_NEAR_PLANE].normal.z = viewProj.entries[11]+viewProj.entries[10];
  planes[FRUSTUM_NEAR_PLANE].intercept = viewProj.entries[15]+viewProj.entries[14];
  for(unsigned int i=0; i<6; i++)
    planes[i].Normalize();
}

bool Frustum::IsAabbInside(const Aabb &box) const
{
  for(unsigned int i=0; i<6; i++)
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
  for(unsigned int i=0; i<6; i++)
  {
    if(planes[i].GetDistance(position) < -radius)
      return false;
  }
  return true;
}


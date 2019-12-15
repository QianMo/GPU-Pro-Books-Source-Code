#include <stdafx.h>
#include <Aabb.h>

void Aabb::Inflate(const Vector3 &point)
{
  if(point.x < mins.x)
    mins.x = point.x;
  if(point.y < mins.y)
    mins.y = point.y;
  if(point.z < mins.z)
    mins.z = point.z;
  if(point.x > maxes.x)
    maxes.x = point.x;
  if(point.y > maxes.y)
    maxes.y = point.y;
  if(point.z > maxes.z)
    maxes.z = point.z;
}

void Aabb::Inflate(const Aabb &box)
{
  if(box.mins.x < mins.x)
    mins.x = box.mins.x;
  if(box.mins.y < mins.y)
    mins.y = box.mins.y;
  if(box.mins.z < mins.z)
    mins.z = box.mins.z;
  if(box.maxes.x > maxes.x)
    maxes.x = box.maxes.x;
  if(box.maxes.y > maxes.y)
    maxes.y = box.maxes.y;
  if(box.maxes.z > maxes.z)
    maxes.z = box.maxes.z;
}

Aabb Aabb::Lerp(const Aabb &box, float factor) const
{
	Aabb result;
	result.mins = mins.Lerp(box.mins, factor);
	result.maxes = maxes.Lerp(box.maxes, factor);
  return result;
}

void Aabb::GetTransformedAabb(Aabb &transformedBox, const Matrix4 &transformMatrix) const
{
  Vector3 axis[3];
  axis[0].Set(transformMatrix.entries[0], transformMatrix.entries[1], transformMatrix.entries[2]);
  axis[0].Normalize();
  axis[1].Set(transformMatrix.entries[4], transformMatrix.entries[5], transformMatrix.entries[6]);
  axis[1].Normalize();
  axis[2].Set(transformMatrix.entries[8], transformMatrix.entries[9], transformMatrix.entries[10]);
  axis[2].Normalize();

  Vector3 halfExtents = GetExtents() * 0.5f;
  Vector3 rotHalfExtents;
  for(UINT i=0; i<3; i++)
    rotHalfExtents[i] = fabs(halfExtents[0] * axis[0][i]) + fabs(halfExtents[1] * axis[1][i]) + fabs(halfExtents[2] * axis[2][i]);
  Vector3 transCenter = (transformMatrix*GetCenter());

  transformedBox.mins = transCenter - rotHalfExtents;
  transformedBox.maxes = transCenter + rotHalfExtents;
}

bool Aabb::IntersectRay(const Vector3 &rayOrigin, const Vector3 &rayDir, float &minDistance, float &maxDistance) const
{
  float tMin = -FLT_MAX;
  float tMax = FLT_MAX;

  for(UINT i=0; i<3; i++) 
  {
    if(rayDir[i] != 0.0f) 
    {
      float invRayDir = 1.0f / rayDir[i];
      float t1 = (mins[i] - rayOrigin[i]) * invRayDir;
      float t2 = (maxes[i] - rayOrigin[i]) * invRayDir;
      tMin = max(tMin, min(t1, t2));
      tMax = min(tMax, max(t1, t2));
    } 
    else if((rayOrigin[i] <= mins[i]) || (rayOrigin[i] >= maxes[i])) 
    {
      return false;
    }
  }

  minDistance = tMin;
  maxDistance = tMax;

  return ((tMax > tMin) && (tMax > 0.0f));
}

void Aabb::GetCorners(Vector3 *corners) const
{
  assert(corners != nullptr);

  corners[0].Set(mins.x, mins.y, mins.z);
  corners[1].Set(mins.x, mins.y, maxes.z);
  corners[2].Set(mins.x, maxes.y, mins.z);
  corners[3].Set(mins.x, maxes.y, maxes.z);
  corners[4].Set(maxes.x, mins.y, mins.z);
  corners[5].Set(maxes.x, mins.y, maxes.z);
  corners[6].Set(maxes.x, maxes.y, mins.z);
  corners[7].Set(maxes.x, maxes.y, maxes.z);
}
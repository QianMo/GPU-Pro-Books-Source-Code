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

#ifndef PATH_POINT_LIGHT_H
#define PATH_POINT_LIGHT_H

#include <PointLight.h>
#include <Aabb.h>

// PathPointLight
//
// Point light that follows a random path.
class PathPointLight
{
public:
  PathPointLight():
    pointLight(NULL),
    paused(false)
  {
  }

  bool Init();

  void Update();

  void SetActive(bool active)
  {
    pointLight->SetActive(active);
  }

  void SetPaused(bool paused)
  {
    this->paused = paused;
  }

  static void SetBounds(const Vector3 &mins, const Vector3 &maxes)
  {
    boundingBox.mins = mins;
    boundingBox.maxes = maxes;
  }

private:  
  PointLight *pointLight;
  Vector3 direction;
  bool paused;
  static Aabb boundingBox;

};

#endif
#ifndef PATH_POINT_LIGHT_H
#define PATH_POINT_LIGHT_H

#include <POINT_LIGHT.h>

#define PATH_POINTLIGHT_MOVE_SPEED 0.1f
 
// PATH_POINT_LIGHT
//  Point-light that follows a simple rectangular path on the XZ-plane.
class PATH_POINT_LIGHT
{
public:
  PATH_POINT_LIGHT()
	{
		pointLight = NULL;
		paused = false;
	}

	bool Init(const VECTOR3D &position,float radius,const COLOR &color,float multiplier,const VECTOR3D &direction);

	void Update();

	void SetActive(bool active)
	{
		pointLight->SetActive(active);
	}

	void SetPaused(bool paused)
	{
		this->paused = paused;
	}

	static void SetControlPoints(float minX,float maxX,float minZ,float maxZ)
	{
    controlPoints[0] = minX;
		controlPoints[1] = maxX;
		controlPoints[2] = minZ;
    controlPoints[3] = maxZ;
	}

private:  
	POINT_LIGHT *pointLight;
	VECTOR3D direction;
	bool paused;
	static float controlPoints[4];

};

#endif
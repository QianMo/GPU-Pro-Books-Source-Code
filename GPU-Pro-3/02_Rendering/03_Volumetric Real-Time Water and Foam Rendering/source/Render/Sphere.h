#ifndef __SPHERE__H__
#define __SPHERE__H__

#include "../Render/RenderObject.h"

class Vector3;

class Sphere : public RenderObject
{
public:
	Sphere(void);
	~Sphere(void);

	// inits the sphere
	void Init(const int& idNum, const float& r, const int& p, const Vector3& pos, const int& matId=0, const bool& addToPhysicWorld=true, const bool& isLevelElement=false);

private:
	// creats the sphere geometry
	void CreateSphereGeometry(const float& cx, const float& cy, const float& cz, float& r, int& p);

	// set a vertex
	void SetVertexData(const int& index,
		const float& tu, const float& tv, 
		const float& tanx, const float& tany, const float& tanz,
		const float& binx, const float& biny, const float& binz, 
		const float& nx, const float& ny, const float& nz, 
		const float& vx, const float& vy, const float& vz);

	// radius of the sphere
	float radius;

	// precision of the sphere
	int precision;
};

#endif


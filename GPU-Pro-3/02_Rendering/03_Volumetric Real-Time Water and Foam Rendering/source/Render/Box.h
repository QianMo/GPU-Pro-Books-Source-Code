#ifndef __BOX__H__
#define __BOX__H__

#include "../Render/RenderObject.h"

class Vector3;

class Box : public RenderObject
{
public:
	Box(void);
	~Box(void);

	// inits the box
	void Init(const int& idNum, const Vector3& pos, const float& w, const float& h, const float& d,
		      const Vector3& su, const Vector3& sv, const int& matId=0, const bool& addToPhysicWorld=true,
			  const bool& isLevelElement=false);

private:
	// sets a vertex
	void SetVertexData(const int& index,
					   const float& tu, const float& tv,
					   const float& tanx, const float& tany, const float& tanz,
					   const float& binx, const float& biny, const float& binz, 
					   const float& nx, const float& ny, const float& nz, 
					   const float& vx, const float& vy, const float& vz);

	// width of the box
	float width;

	// height of the box
	float height;

	// depth of the box
	float depth;
};

#endif


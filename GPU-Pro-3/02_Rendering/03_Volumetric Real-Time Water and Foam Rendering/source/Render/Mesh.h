#ifndef __MESH__H__
#define __MESH__H__

#include "../Render/RenderObject.h"

class Vector3;

class Mesh : public RenderObject
{
public:
	Mesh(void);
	~Mesh(void);

	// inits the mesh
	void Init(const int& idNum, const char* fileName, const int& matId=0, const bool& addToPhysicWorld=true, const bool& isLevelElement=false);
	
	void InitCollisionMesh(const char* fileName);

private:

	// set a vertex
	void SetVertexData(const int& index,
		const float& tu, const float& tv, 
		const float& nx, const float& ny, const float& nz, 
		const float& vx, const float& vy, const float& vz);
};

#endif


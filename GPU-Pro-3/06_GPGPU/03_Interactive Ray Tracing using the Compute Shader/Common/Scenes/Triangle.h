#ifndef __TRIANGLE__H_
#define __TRIANGLE__H_

#include "Primitive.h"
#include "Geometry.h"

class Triangle : public Primitive
{
public:
	Triangle(Vertex* a_V1, Vertex* a_V2);
	~Triangle(void);

	int getType() { return TRIANGLE; }
	int intersectP(Ray &a_Ray, float &a_T);
	Normal getNormal( Point &a_Pos );
	//Color getColor( void ) { return m_Material.getColor(); }
};

#endif
#ifndef TRIANGLE_H
#define TRIANGLE_H

#include "Vector.h"


class Triangle 
{

    public:
   
	Triangle(Vector p0, Vector p1, Vector p2); 
	void draw(bool drawFaceNormal); 
	Vector getNormal() { return mNormal; };

    private:
   
        Vector mVertices[3]; 
        Vector mNormals[3]; 
        Vector mColor; 
        Vector mNormal; 
};


#endif   // !TRIANGLE_H


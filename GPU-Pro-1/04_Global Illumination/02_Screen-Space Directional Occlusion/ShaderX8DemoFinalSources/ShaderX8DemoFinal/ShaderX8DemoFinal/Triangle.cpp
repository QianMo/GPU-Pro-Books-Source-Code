
#include "Triangle.h"

#include <stdio.h>
#include <glut.h>

Triangle::Triangle(Vector p0, Vector p1, Vector p2)
{
    mVertices[0] = p0;
    mVertices[1] = p1;
    mVertices[2] = p2;

	Vector u = mVertices[1] - mVertices[0];
	Vector v = mVertices[2] - mVertices[0];
	mNormal = cross(u, v);
	mNormal.normalize();
}



void Triangle::draw(bool drawFaceNormal)
{
	// glColor3f(mColor[0], mColor[1], mColor[2]);
    // glBegin(GL_TRIANGLES);
	if (drawFaceNormal)
		glNormal3f(mNormal[0],mNormal[1], mNormal[2]);
    glVertex3f(mVertices[0][0], mVertices[0][1], mVertices[0][2]);
	if (drawFaceNormal)
		glNormal3f(mNormal[0],mNormal[1], mNormal[2]);
    glVertex3f(mVertices[1][0], mVertices[1][1], mVertices[1][2]);
	if (drawFaceNormal)
		glNormal3f(mNormal[0],mNormal[1], mNormal[2]);
    glVertex3f(mVertices[2][0], mVertices[2][1], mVertices[2][2]);
    // glEnd();


}




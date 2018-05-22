#pragma once

#include "Mesh3D.h"

enum _shapeType
{
    SHAPE_NONE = 0,
    SHAPE_PLANE,
    SHAPE_SPHERE,
    SHAPE_CYLINDER,
    SHAPE_TORUS,
    SHAPE_SUPER_TOROID,
    SHAPE_SUPER_SHAPE,

    SHAPE_ELLIPSOID,
    SHAPE_HYPERBOLOID_OF_ONE_SHEET,
    SHAPE_SUPER_ELLIPSOID,
    SHAPE_SUPER_HYPERBOLOID_OF_ONE_SHEET
};

class Shape3D : public Mesh3D
{
public:
	Shape3D(_shapeType type, float radius, int depth);
	
    // unused really since the geometry is constructed
    void readFormat (char *filename) {};
    void writeFormat (char *filename) {};

	void sphere (float radius, int depth);
	void torus (float R0, float R1, float N1, float N2,
        float radius, int depth);
	void superShape (
        float M1, float A1, float B1, float N11, float N12, float N13,
        float M2, float A2, float B2, float N21, float N22, float N23,
        float radius, int depth);
    void ellipsoid (float N1, float N2, float radius, int depth);
    void hyperboloid_one_sheet (float N1, float N2, float radius, int depth);

    void createGeometry (int numfaces, int thetaSize, int phiSize,
        int phiTrisNorth, int phiTrisSouth,
        Vector3D * coords, Vector3D * norms, Vector2D * textr);

protected:
    _shapeType type;
    float radius;
    int depth;
};


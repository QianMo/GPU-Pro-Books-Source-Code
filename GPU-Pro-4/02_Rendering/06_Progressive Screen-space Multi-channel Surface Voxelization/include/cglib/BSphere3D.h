#pragma once

#include <math.h>

#include "cglibdefines.h" 
#include "Vector3D.h" 

class BSphere3D 
{
public:
	void makeEmpty () { center = Vector3D (0.0f, 0.0f, 0.0f); radius = -1.0f; }
	void set (Vector3D c, DBL r) { center = c; radius = r; }
	void setRadius (DBL r) { radius = r; }
	inline DBL getRadius (void) const { return radius; }
	inline DBL getRadius2 (void) const { return radius * radius; }

	void setCenter (Vector3D c) { center = c; }
	inline Vector3D getCenter (void) const { return center; }

    // s = s * m
    BSphere3D operator* (class Matrix4D m);
    BSphere3D & operator*= (class Matrix4D m);
    void xform (class Matrix4D m);
    void xform (BSphere3D s, class Matrix4D m);

    inline bool isValid (void) const { return radius >= 0.0f; }

    // Expands the sphere to encompass the given point. Repositions the
    // sphere center to minimize the radius increase. If the sphere is
    // uninitialized, set its center to v and radius to zero. */
    void expandBy (const Vector3D & v);

    // Expands the sphere to encompass the given point. Does not
    // reposition the sphere center. If the sphere is
    // uninitialized, set its center to v and radius to zero. */
    void expandRadiusBy (const Vector3D & v);

    // Expands the sphere to encompass the given sphere. Repositions the
    // sphere center to minimize the radius increase. If the sphere is
    // uninitialized, set its center and radius to match sphere. */
    void expandBy (const BSphere3D & bsphere);

    // Expands the sphere to encompass the given sphere. Does not
    // repositions the sphere center. If the sphere is
    // uninitialized, set its center and radius to match sphere. */
    void expandRadiusBy (const BSphere3D & bsphere);

    // Expands the sphere to encompass the given box. Repositions the
    // sphere center to minimize the radius increase. */
    void expandBy (const class BBox3D & bbox);

    // Expands the sphere to encompass the given box. Does not
    // repositions the sphere center. */
    void expandRadiusBy (const class BBox3D & bbox);

    // Returns true if v is within the sphere. */
    inline bool contains (const Vector3D & v) const
    {
        return isValid() && (v - center).length2 () <= getRadius2 ();
    }

    void dump (void)
    {
        EAZD_PRINT ("Bounding Sphere [center, radius]: [" <<
                    center.x << " " << center.y << " " << center.z << " --- " << radius << "]");
    }

    void draw (void);

private:
	DBL radius;
	Vector3D center;
};


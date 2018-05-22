#pragma once

#include <math.h>

#include "cglibdefines.h"
#include "Vector3D.h"
#include "Ray3D.h"

class BBox3D
{
public:
    void makeEmpty ()
	{
		min = max = center = size = Vector3D (0.0f, 0.0f, 0.0f);
	}

	void set (Vector3D corner_min, Vector3D corner_max)
	{
		min = corner_min; max = corner_max;
		size = max - min; center = (min + max) * 0.5;
	}

	void setSymmetrical (Vector3D center_point, Vector3D sz)
	{
		size = sz; center = center_point;
		min = center - size * 0.5;
		max = center + size * 0.5;
	}

	inline Vector3D getSize     (void) const { return size; }
	       DBL      getMaxSide  (void);

	inline Vector3D getCenter   (void) const { return center; }
	inline Vector3D getMin      (void) const { return min; }
	inline Vector3D getMax      (void) const { return max; }

	inline bool isValid (void) const
	{
		return max.x >= min.x && max.y >= min.y && max.z >= min.z;
	}

	// Calculates and returns the bounding box radius.
	inline DBL getRadius (void) const { return sqrt (getRadius2 ()); }

    // Calculates and returns the squared length of the bounding box radius.
    // Note, getRadius2 () is faster to calculate than radius ().
    inline DBL getRadius2() const { return 0.25f * (size.length2 ()); }

    // Returns a specific corner of the bounding box.
    // pos specifies the corner as a number between 0 and 7.
    // Each bit selects an axis, X, Y, or Z from least- to
    // most-significant. Unset bits select the minimum value
    // for that axis, and set bits select the maximum.
    inline const Vector3D getCorner (unsigned int pos) const
    {
        return Vector3D (pos & 1 ? max.x : min.x,
                         pos & 2 ? max.y : min.y,
                         pos & 4 ? max.z : min.z);
    }

    // Expands the bounding box to include the given coordinate.
    // If the box is uninitialized, set its min and max extents to v. */
    void expandBy (const Vector3D & v);

    // Expands the bounding box to include the given coordinate.
    // If the box is uninitialized, set its min and max extents to Vec3(x,y,z).
    void expandBy (DBL x, DBL y, DBL z);

    // Expands this bounding box to include the given bounding box.
    // If this box is uninitialized, set it equal to box.
    void expandBy (const BBox3D & bbox);

    // Expands this bounding box to include the given sphere.
    // If this box is uninitialized, set it to include the sphere.
    void expandBy (const class BSphere3D & bsphere);

    // Returns true if this bounding box contains the specified coordinate.
    inline bool contains (const Vector3D & v) const
    {
        return isValid () &&
               (v[0] >= min.x && v[0] <= max.x) &&
               (v[1] >= min.y && v[1] <= max.y) &&
               (v[2] >= min.z && v[2] <= max.z);
    }

    void dump (void)
    {
        EAZD_PRINT ("Bounding Box [min, max]: [" <<
                    min.x << " " << min.y << " " << min.z << " --- " <<
                    max.x << " " << max.y << " " << max.z << "]" << std::endl <<
                    "             [center]: [" <<
                    center.x << " " << center.y << " " << center.z << "]" << std::endl <<
                    "             [diagonal diameter]: " <<
                    size.length ());
    }

    void draw (void);

	bool intersect(Ray3D &ray);

private:
	Vector3D size;
	Vector3D min;
	Vector3D max;
	Vector3D center;
};


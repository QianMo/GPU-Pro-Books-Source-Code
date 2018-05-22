
#ifdef WIN32
	#define WIN32_LEAN_AND_MEAN
	#include <windows.h>
#endif

#include <math.h>
#include <float.h>
#include "Vector3D.h"
#include "BBox3D.h"
#include "BSphere3D.h"

#include <GL/gl.h>

float BBox3D::getMaxSide (void)
{
    float   maxside;

    if (size.x > size.y)
    {
        if (size.x > size.z)
            maxside = size.x;
        else
            maxside = size.z;
    }
    else
    {
        if (size.y > size.z)
            maxside = size.y;
        else
            maxside = size.z;
    }

    return maxside;
}

// Expands the bounding box to include the given coordinate.
// If the box is uninitialized, set its min and max extents to v. */
void BBox3D::expandBy (const Vector3D & v)
{
    if (v[0] < min.x) min.x = v[0];
    if (v[0] > max.x) max.x = v[0];

    if (v[1] < min.y) min.y = v[1];
    if (v[1] > max.y) max.y = v[1];

    if (v[2] < min.z) min.z = v[2];
    if (v[2] > max.z) max.z = v[2];

    size = max - min; center = (min + max) * 0.5;
}

// Expands the bounding box to include the given coordinate.
// If the box is uninitialized, set its min and max extents to Vec3(x,y,z).
void BBox3D::expandBy (DBL x, DBL y, DBL z)
{
    if (x < min.x) min.x = x;
    if (x > max.x) max.x = x;

    if (y < min.y) min.y = y;
    if (y > max.y) max.y = y;

    if (z < min.z) min.z = z;
    if (z > max.z) max.z = z;

    size = max - min; center = (min + max) * 0.5;
}

// Expands this bounding box to include the given bounding box.
// If this box is uninitialized, set it equal to box.
void BBox3D::expandBy (const BBox3D & bbox)
{
    if (! bbox.isValid ()) return;

    if (bbox.getMin ().X () < min.x) min.x = bbox.getMin ().X ();
    if (bbox.getMax ().X () > max.x) max.x = bbox.getMax ().X ();

    if (bbox.getMin ().Y () < min.y) min.y = bbox.getMin ().Y ();
    if (bbox.getMax ().Y () > max.y) max.y = bbox.getMax ().Y ();

    if (bbox.getMin ().Z () < min.z) min.z = bbox.getMin ().Z ();
    if (bbox.getMax ().Z () > max.z) max.z = bbox.getMax ().Z ();

    size = max - min; center = (min + max) * 0.5;
}

// Expands this bounding box to include the given sphere.
// If this box is uninitialized, set it to include the sphere.
void BBox3D::expandBy (const BSphere3D & bsphere)
{
    if (! bsphere.isValid ()) return;

    if (bsphere.getCenter ().X () - bsphere.getRadius () < min.x)
        min.x = bsphere.getCenter ().X () - bsphere.getRadius ();
    if (bsphere.getCenter ().X () + bsphere.getRadius () > max.x)
        max.x = bsphere.getCenter ().X () + bsphere.getRadius ();

    if (bsphere.getCenter ().Y () - bsphere.getRadius () < min.y)
        min.y = bsphere.getCenter ().Y () - bsphere.getRadius ();
    if (bsphere.getCenter ().Y () + bsphere.getRadius () > max.y)
        max.y = bsphere.getCenter ().Y () + bsphere.getRadius ();

    if (bsphere.getCenter ().Z () - bsphere.getRadius () < min.z)
        min.z = bsphere.getCenter ().Z () - bsphere.getRadius ();
    if (bsphere.getCenter ().Z () + bsphere.getRadius () > max.z)
        max.z = bsphere.getCenter ().Z () + bsphere.getRadius ();

    size = max - min; center = (min + max) * 0.5;
}

void BBox3D::draw (void)
{
    Vector3D verts[8];

    verts[0] = Vector3D (min.x, min.y, min.z);
    verts[1] = Vector3D (max.x, min.y, min.z);
    verts[2] = Vector3D (max.x, max.y, min.z);
    verts[3] = Vector3D (min.x, max.y, min.z);
    verts[4] = Vector3D (min.x, min.y, max.z);
    verts[5] = Vector3D (max.x, min.y, max.z);
    verts[6] = Vector3D (max.x, max.y, max.z);
    verts[7] = Vector3D (min.x, max.y, max.z);

    glColor3f (1.0, 1.0, 0.0);

    glBegin (GL_LINE_STRIP);
    glVertex3f (verts[0].x, verts[0].y, verts[0].z);
    glVertex3f (verts[1].x, verts[1].y, verts[1].z);
    glVertex3f (verts[2].x, verts[2].y, verts[2].z);
    glVertex3f (verts[3].x, verts[3].y, verts[3].z);
    glVertex3f (verts[0].x, verts[0].y, verts[0].z);
    glVertex3f (verts[4].x, verts[4].y, verts[4].z);
    glVertex3f (verts[5].x, verts[5].y, verts[5].z);
    glVertex3f (verts[6].x, verts[6].y, verts[6].z);
    glVertex3f (verts[7].x, verts[7].y, verts[7].z);
    glVertex3f (verts[4].x, verts[4].y, verts[4].z);
    glEnd ();

    glBegin (GL_LINE_STRIP);
    glVertex3f (verts[1].x, verts[1].y, verts[1].z);
    glVertex3f (verts[5].x, verts[5].y, verts[5].z);
    glEnd ();

    glBegin (GL_LINE_STRIP);
    glVertex3f (verts[2].x, verts[2].y, verts[2].z);
    glVertex3f (verts[6].x, verts[6].y, verts[6].z);
    glEnd ();

    glBegin (GL_LINE_STRIP);
    glVertex3f (verts[3].x, verts[3].y, verts[3].z);
    glVertex3f (verts[7].x, verts[7].y, verts[7].z);
    glEnd ();
}

bool BBox3D::intersect(Ray3D &ray)
{
	// From M. Pharr, G. Humphreys, Physically Based Rendering, MK.
	
	DBL t0=-FLT_MAX, t1=FLT_MAX;
	for (int i=0; i<3; i++)
	{
		DBL invRayDir = 1.0/ray.dir[i];
		DBL tNear = (min[i] - ray.origin[i])*invRayDir;
		DBL tFar = (max[i] - ray.origin[i])*invRayDir;
		if (tNear > tFar)
		{
			DBL tmp = tNear;
			tNear = tFar;
			tFar = tmp;
		}
		t0 = tNear > t0 ? tNear : t0;
		t1 = tFar < t1 ?  tFar  : t1;
		if (t0 >= t1+ZERO_TOLERANCE_7)
		{
			ray.hit = false;
			return false;
		}
	}
	if (t1>ZERO_TOLERANCE_7)
	{
		if (t0>0)
		{
			ray.t = t0;
			ray.inside = false;
		}
		else
		{
			ray.t = t1;
			ray.inside = true;
		}
	}
	else
	{
		ray.hit = false;
		return false;
	}
	
	ray.hit = true;
	
	ray.p_isect = ray.origin + ray.dir*ray.t;

	return ray.hit;
}
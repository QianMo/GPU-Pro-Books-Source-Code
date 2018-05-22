
#include <math.h>

#include "Vector3D.h"
#include "Matrix4D.h"
#include "BBox3D.h"
#include "BSphere3D.h"

#include <GL/glut.h>

// take any point on the sphere that we will transform
// as well and use later, to determine the new radius;
// we simply choose center + (radius, 0, 0)
BSphere3D BSphere3D::operator* (Matrix4D m)
{
    Vector3D sphPnt = center + Vector3D (radius, 0.0f, 0.0f);

    // transform the center (as point) & radius
    BSphere3D sph;
    sph.center = center * m;
    sph.radius = (sphPnt * m - sph.center).length ();
    return sph;
}

BSphere3D & BSphere3D::operator*= (Matrix4D m)
{
    Vector3D sphPnt = center + Vector3D (radius, 0.0f, 0.0f);

    // transform the center (as point) & radius
    BSphere3D sph;
    sph.center = center * m;
    sph.radius = (sphPnt * m - sph.center).length ();

    center = sph.center;
    radius = sph.radius;
    return *this;
}

void BSphere3D::xform (Matrix4D m)
{
    Vector3D sphPnt = center + Vector3D (radius, 0.0f, 0.0f);

    // transform the center (as point) & radius
    BSphere3D sph;
    sph.center = center * m;
    sph.radius = (sphPnt * m - sph.center).length ();

    center = sph.center;
    radius = sph.radius;
}

void BSphere3D::xform (BSphere3D s, Matrix4D m)
{
    Vector3D sphPnt = s.getCenter () + Vector3D (s.getRadius (), 0.0f, 0.0f);

    // transform the center (as point) & radius
    BSphere3D sph;
    sph.center = s.getCenter () * m;
    sph.radius = (sphPnt * m - sph.center).length ();

    center = sph.center;
    radius = sph.radius;
}

// Expands the sphere to encompass the given point. Repositions the
// sphere center to minimize the radius increase. If the sphere is
// uninitialized, set its center to v and radius to zero. */
void BSphere3D::expandBy (const Vector3D & v)
{
    if (isValid ())
    {
        Vector3D dv = v - center;
        DBL r = dv.length ();
        if (r > radius)
        {
            DBL dr = (r - radius) * 0.5f;
            center += dv * (dr/r);
            radius += dr;
        } // else do nothing as vertex is within sphere.
    }
    else
    {
        center = v;
        radius = 0.0;
    }
}

// Expands the sphere to encompass the given point. Does not
// reposition the sphere center. If the sphere is
// uninitialized, set its center to v and radius to zero. */
void BSphere3D::expandRadiusBy (const Vector3D & v)
{
    if (isValid ())
    {
        DBL r = (v - center).length ();
        if (r > radius) radius = r;
        // else do nothing as vertex is within sphere.
    }
    else
    {
        center = v;
        radius = 0.0;
    }
}

// Expands the sphere to encompass the given sphere. Repositions the
// sphere center to minimize the radius increase. If the sphere is
// uninitialized, set its center and radius to match sphere. */
void BSphere3D::expandBy (const BSphere3D & bsphere)
{
    // ignore operation if incomming BoundingSphere is invalid.
    if (! bsphere.isValid ()) return;

    // This sphere is not set so use the inbound sphere
    if (! isValid ())
    {
        center = bsphere.getCenter ();
        radius = bsphere.getRadius ();

        return;
    }

    // Calculate d == The distance between the sphere centers
    DBL d = (center - bsphere.getCenter ()).length ();

    // New sphere is already inside this one
    if (d + bsphere.getRadius () <= radius) return;

    //  New sphere completely contains this one
    if (d + radius <= bsphere.getRadius ())
    {
        center = bsphere.center;
        radius = bsphere.radius;

        return;
    }

    // Build a new sphere that completely contains the other two:
    //
    // The center point lies halfway along the line between the furthest
    // points on the edges of the two spheres.
    //
    // Computing those two points is ugly - so we'll use similar triangles
    DBL new_radius = (radius + d + bsphere.getRadius ()) * 0.5f;
    DBL ratio = (new_radius - radius) / d;

    center += (bsphere.getCenter () - center) * ratio;
    radius = new_radius;
}

// Expands the sphere to encompass the given sphere. Does not
// repositions the sphere center. If the sphere is
// uninitialized, set its center and radius to match sphere. */
void BSphere3D::expandRadiusBy (const BSphere3D & bsphere)
{
    if (bsphere.isValid ())
    {
        if (isValid ())
        {
            DBL r = (bsphere.getCenter () - center).length () + bsphere.getRadius ();
            if (r > radius) radius = r;
            // else do nothing as vertex is within sphere.
        }
        else
        {
            center = bsphere.getCenter ();
            radius = bsphere.getRadius ();
        }
    }
}

// Expands the sphere to encompass the given box. Repositions the
// sphere center to minimize the radius increase. */
void BSphere3D::expandBy (const BBox3D & bbox)
{
    if (bbox.isValid ())
    {
        if (isValid ())
        {
            BBox3D newbb (bbox);

            for (unsigned int c = 0; c < 8; ++c)
            {
                Vector3D v = bbox.getCorner (c) - center; // get the direction vector from corner
                v.normalize (); // normalise it.
                v *= -radius; // move the vector in the opposite direction distance radius.
                v += center; // move to absolute position.
                newbb.expandBy (v); // add it into the new bounding box.
            }

            center = newbb.getCenter ();
            radius = newbb.getRadius ();
        }
        else
        {
            center = bbox.getCenter ();
            radius = bbox.getRadius ();
        }
    }
}

// Expands the sphere to encompass the given box. Does not
// repositions the sphere center. */
void BSphere3D::expandRadiusBy (const BBox3D & bbox)
{
    if (bbox.isValid ())
    {
        if (isValid ())
        {
            for (unsigned int c = 0; c < 8; ++c)
                expandRadiusBy (bbox.getCorner (c));
        }
        else
        {
            center = bbox.getCenter ();
            radius = bbox.getRadius ();
        }
    }
}

void BSphere3D::draw (void)
{
    int   matrixMode;

    // save current matrix mode
    glGetIntegerv (GL_MATRIX_MODE, &matrixMode);
    glMatrixMode (GL_MODELVIEW);

    glColor3f (1.0, 1.0, 0.0);

    glPushMatrix ();
    // set the location of the sphere
    glTranslatef (center.x, center.y, center.z);
    glutWireSphere (radius, 10, 6);
    glPopMatrix ();

    // restore user's matrix mode
    glMatrixMode (matrixMode);
}

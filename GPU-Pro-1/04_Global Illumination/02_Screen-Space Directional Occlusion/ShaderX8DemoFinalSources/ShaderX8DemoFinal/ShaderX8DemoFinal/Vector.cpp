
#include "Vector.h"


//------------------------------------------------------------------------------

/**
 * Leerer Konstruktor.
 */
Vector::Vector()
{
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

/**
 * Konstruktor fuer direkte Wertzuweisung.
 * \param x X-Koordinate
 * \param y Y-Koordinate
 * \param z Z-Koordinate
 */
Vector::Vector(float x, float y, float z)
{
  setValues(x, y, z);
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

/**
 * Vektor fuer Konvertierung Punkt->Vektor.
 * \param p Zeilpunkt
 */
Vector::Vector(const Vector &p)
{
   mElement[X] = p[X];
   mElement[Y] = p[Y];
   mElement[Z] = p[Z];
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

#if 0
Vector Vector::worldToCam()
{
    Vector worldCoord = Vector(mElement[X], mElement[Y], mElement[Z]);
    Vector camCoord = Vector(0.0f, 0.0f, 0.0f);
    
    Vector camZ = Vector(Scpm::getInstance()->getScene()->getCamera()->getLookAt() - Scpm::getInstance()->getScene()->getCamera()->getPosition());    
    camZ.normalize();
    
    Vector camX = Vector(cross(Scpm::getInstance()->getScene()->getCamera()->getUpVector(), camZ));
    camX.normalize();
	    
    Vector camY = Vector(cross(camZ, camX));
    
    Vector camPosition = Scpm::getInstance()->getScene()->getCamera()->getPosition();
    
    worldCoord = -camPosition + worldCoord;
    
    camCoord[0] = dot(worldCoord, camX);
    camCoord[1] = dot(worldCoord, camY);
    camCoord[2] = dot(worldCoord, camZ);
    
    return camCoord;
    
}
#endif

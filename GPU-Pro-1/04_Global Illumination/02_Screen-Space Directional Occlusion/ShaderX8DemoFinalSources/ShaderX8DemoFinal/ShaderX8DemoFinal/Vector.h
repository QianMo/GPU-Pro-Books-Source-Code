#ifndef VECTOR_H
#define VECTOR_H

//#include <cassert>
#include <cmath>

#define SQR(x)   ((x) * (x))

const int X = 0;
const int Y = 1;
const int Z = 2;

/**
 * Klasse zur Beschreibung eines Richtungsvektors im R3.
 * Der Vektor hat keinen Stuetzpunkt, nur eine Richtung, ausgehend vom Nullpunkt.
 */
class Vector
{
   public:
   
      Vector(); // Leerer Konstruktor
      Vector(float x, float y, float z); // Konstruktor ueber Werte
      Vector(const Vector &p); // Copy Konstruktor 
      
      // Vector worldToCam();
      
	inline void setValues(float x, float y, float z); // Setzt die Tupel mit eigenen Werten

      inline float length()    const; // Laenge des Vektors
      inline float lengthSqr() const; // Quadrat der Laenge
      inline void  normalize();
      inline void  normalizeCheck();
      
      inline Vector operator -() const; // Ueberladener Operator fuer Invertierung

      inline Vector operator +(const Vector& v) const; // Ueberladener Operator fuer Addition
      inline Vector operator -(const Vector& v) const; // Ueberladener Operator fuer Subtraktion

      inline Vector operator *(float s) const; // Ueberladener Operator fuer Multiplikation
      inline Vector operator /(float s) const; // Ueberladener Operator fuer Division
      inline Vector operator *(const Vector &v) const;
      inline Vector operator /(const Vector &v) const;

      inline float& operator [](int index); // Ueberladener Operator fuer schreibzugriff
      inline const float& operator [](int index) const; // Ueberladener Operator fuer Lesezugriff

	  inline Vector calcDifVector() const; //Funktion setzt die kleinste Komponente des Vektors auf 1
		
     
  private:
   
        float mElement[3]; //! Werte
};

inline float  dot  (const Vector& v1, const Vector& v2); // Skalarprodukt
inline Vector cross(const Vector& v1, const Vector& v2); // Kreuzprodukt

inline Vector normalize(const Vector& v); // Normalisiert einen Vektor
// inline Vector normalizeCheck(const Vector& v); // Normalisiert einen Vektor

inline Vector operator *(float s, const Vector& v); // Operator fuer Skalarmultiplikation

/**
* Vektor mit Werte fuellen
 */
inline void Vector::setValues(float x, float y, float z)
{
	mElement[X] = x;
	mElement[Y] = y;
	mElement[Z] = z;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

/**
* Laenge des Vektors.
 * \return Laenge des Vektors
 */
float Vector::length() const
{
	return (float)sqrt(SQR(mElement[X]) + SQR(mElement[Y]) + SQR(mElement[Z]));
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

/**
* Laenge des Vektors zum Quadrat.
 * \return Laenge des Vektors ohne die Wurzel zu ziehen
 */
float Vector::lengthSqr() const
{
	return SQR(mElement[X]) + SQR(mElement[Y]) + SQR(mElement[Z]);
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

/**
* Normalisiert den Vektor auf die L�ge 1.
 */
void Vector::normalize()
{
	float len = length();
	
	//assert(len != 0.0);
	if (len > 0.0001) {	                   // *** remove this check !!!
		mElement[X] /= len;
		mElement[Y] /= len;
		mElement[Z] /= len;
	}
	else {
		mElement[X] = mElement[Y] = mElement[Z] = 0.0;
	}
}

#if 0
void Vector::normalizeCheck()
{
	float len = length();
	
	if (len != 0.0) {
	
		mElement[X] /= len;
		mElement[Y] /= len;
		mElement[Z] /= len;
	}
	else {
		mElement[X] = mElement[Y] = mElement[Z] = 0.0;
	}
	// return this;
}
#endif

/**
* Ueberladener negierungs-Operator.
 */
Vector Vector::operator -() const
{
	return Vector(-mElement[X], -mElement[Y], -mElement[Z]);
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

/**
* Ueberladener Operator fuer die Addtion.
 */
Vector Vector::operator +(const Vector& v) const
{
	return Vector(mElement[X] + v[X], mElement[Y] + v[Y], mElement[Z] + v[Z]);
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

/**
* Ueberladener Operator fuer die Subtraktion.
 */
Vector Vector::operator -(const Vector& v) const
{
	return Vector(mElement[X] - v[X], mElement[Y] - v[Y], mElement[Z] - v[Z]);
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

/** 
* Ueberladener Operator fuer die Multiplikation mit einem Skalar.
*/
Vector Vector::operator *(float s) const
{
	return Vector(s * mElement[X], s * mElement[Y], s * mElement[Z]);
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

/** 
* Komponentenweise Multiplikation 
*/
Vector Vector::operator *(const Vector &v) const
{
	return Vector(v[X] * mElement[X], v[Y] * mElement[Y], v[Z] * mElement[Z]);
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

/** 
* Komponentenweise Division 
*/
Vector Vector::operator /(const Vector &v) const
{
	return Vector(v[X] / mElement[X], v[Y] / mElement[Y], v[Z] / mElement[Z]);
}


// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

/**
* Ueberladener Operator fuer die Division durch einen Skalar.
 */
Vector Vector::operator /(float s) const
{
	//assert(s != 0);
	
	return Vector(mElement[X] / s, mElement[Y] / s, mElement[Z] / s);
}


//------------------------------------------------------------------------------

/**
* Liefert das Skalar-(Vektor-)Produkt zweier Vektoren.
 * \param v1 Erster Vektor
 * \param v2 Zweiter Vektor
 * \return Skalarprodukt (Winkel)
 */
float dot(const Vector& v1, const Vector& v2)
{
	return (v1[X] * v2[X] + v1[Y] * v2[Y] + v1[Z] * v2[Z]);
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

/**
* Liefert das Kreuzprodukt zweier Vektoren.
 * \param v1 Erster Vektor
 * \param v2 Zweiter Vektor
 * \return Kreuzprodukt (Normalenvektor, nicht normalisiert)
 */
Vector cross(const Vector& v1, const Vector& v2)
{
	return Vector(v1[Y] * v2[Z] - v1[Z] * v2[Y],
				  v1[Z] * v2[X] - v1[X] * v2[Z],
				  v1[X] * v2[Y] - v1[Y] * v2[X]);
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

/**
* Liefert den Normalisierten Vektor zu dem uebergebenen.
 * Alternative zu der Membervariante, da hier der urspruengliche Vektor nicht veraendert wird.
 * \param v Zu normalisierender Vektor
 * \return Normalisierter Vektor
 */
Vector normalize(const Vector& v)
{
	float len = v.length();
	
	// //assert(len != 0);
	if (len > 0)
		return v / len;
	else 
		return Vector(0,0,0);
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

/**
* Ueberladener Operator zur Multiplikation mit einem Skalar.
 * Notwendig fuer die Kommutativitaet der Multiplikation
 */
Vector operator *(float s, const Vector& v)
{
	return v * s;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

/**
* Ueberladener Operator fuer den schreibenden Parameterzguriff.
 */
float& Vector::operator [](int index)
{
	//assert(index >= X && index <= Z);
	
	return mElement[index];
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

/**
* Ueberladener Operator fuer den lesenden Parameterzugriff.
 */
const float& Vector::operator [](int index) const
{
	//assert(index >= X && index <= Z);
	
	return mElement[index];
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

/**
* Setzt die kleinste Komponente des bergebenen Vektors auf 1
 * \param v Zu veraendernder Vektor
 * \return Vektor mit anderer Richtung
 */

Vector Vector::calcDifVector() const
{
	if (( fabsf(mElement[X]) <= fabsf(mElement[Y])) && ( fabsf(mElement[X]) <= fabsf(mElement[Z])))  
	{
		return Vector(1.0f, mElement[Y], mElement[Z]);
	}
	else if (( fabsf(mElement[Y]) <= fabsf(mElement[X])) && ( fabsf(mElement[Y]) <= fabsf(mElement[Z]))) 
	{
		return Vector(mElement[X], 1.0f, mElement[Z]);
	}
	else 
		return Vector(mElement[X], mElement[Y], 1.0f);
	
}


#endif   // !VECTOR_H


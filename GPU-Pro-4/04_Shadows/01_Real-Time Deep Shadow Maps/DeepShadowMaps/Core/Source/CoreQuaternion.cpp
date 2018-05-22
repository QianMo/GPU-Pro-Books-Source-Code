#include "CoreQuaternion.h"


// Adds 2 Quaternions
CoreQuaternion CoreQuaternionAdd(CoreQuaternion &q1, CoreQuaternion &q2)
{
	return CoreQuaternion(q1.x + q2.x, q1.y + q2.y, q1.z + q2.z, q1.w + q2.w);
}

// Subs 2 Quaternions
CoreQuaternion CoreQuaternionSub(CoreQuaternion &q1, CoreQuaternion &q2)
{
	return CoreQuaternion(q1.x - q2.x, q1.y - q2.y, q1.z - q2.z, q1.w - q2.w);
}

// Muls 2 Quaternions
CoreQuaternion CoreQuaternionMul(CoreQuaternion &q1, CoreQuaternion &q2)
{
	return CoreQuaternion(q1.w * q2.x + q1.x * q2.w + q1.y * q2.w + q1.z * q2.y,
						 q1.w * q2.y + q1.y * q2.w + q1.z * q2.x - q1.x * q2.z,
						 q1.w * q2.z + q1.z * q2.w + q1.x * q2.y - q1.y * q2.x,
						 q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z);
}

// Normalizes a Quaternion
CoreQuaternion CoreQuaternionNormalize(CoreQuaternion &qIn)
{
	return qIn / sqrtf(qIn.w * qIn.w + qIn.x * qIn.x + qIn.y * qIn.y + qIn.z * qIn.z);
}

// Create a Quaternion from an Axis/Angle
CoreQuaternion::CoreQuaternion(CoreVector3 &vAxis, float fAlpha)
{
	v = vAxis;
	w = cos(fAlpha / 2);
}


// Spherical Linear Interpolation
CoreQuaternion CoreQuaternionSlerp(CoreQuaternion &q1, CoreQuaternion &q2, float fAlpha)
{
	CoreQuaternion result;

	// check for out-of range parameter and return edge points if so 
	if(fAlpha <= 0.0)
		return q2;

	if(fAlpha >= 1.0)
		return q1;


	// compute dot product 
	float cosOmega;
	cosOmega = q1.x * q2.x + q1.y * q2.y + q1.z * q2.z + q1.w * q2.w;

	/* if negative dot, use -q2.  two quaternions q and -q
	represent the same rotation, but may produce
	different slerp.  we chose q or -q to rotate using
	the acute angle. */
	result = q2;

	if(cosOmega < 0.0f)
	{
		result.x = -result.x;
		result.y = -result.y;
		result.z = -result.z;
		result.w = -result.w;
		cosOmega = -cosOmega;
	}


	// compute interpolation fraction, checking for quaternions almost exactly the same 
	float k0, k1;

	if (cosOmega > 0.9999f )
	{
		// very close - just use linear interpolation

		k1 = fAlpha;
		k0 = 1.0f - fAlpha;
	}
	else
	{
		// compute the sin of the angle using the trig identity sin^2(omega) + cos^2(omega) = 1 
		float sinOmega = sqrt (1.0f - (cosOmega * cosOmega));

		// compute the angle from its sin and cosine
		float omega = atan2 (sinOmega, cosOmega);

		// compute inverse of denominator, so we only have to divide once
		float invSinOmega = 1.0f / sinOmega;

		// Compute interpolation parameters
		k1 = sin (fAlpha * omega) * invSinOmega;
		k0 = sin ((1.0f - fAlpha) * omega) * invSinOmega;
	}

	// interpolate and return new quaternion
	
	result = (q1 * k0) + (result * k1);
	return result;
}

// Create a Quaternion from a Matrix
CoreQuaternion::CoreQuaternion(CoreMatrix4x4 &mIn)
{
	float fTrace = mIn.Trace();

	if( fTrace > 0.0f ) 
	{
		float s = 0.5f / sqrtf( fTrace );
		
		this->w = 0.25f / s;
		this->x = ( mIn._23 - mIn._32 ) * s;
		this->y = ( mIn._31 - mIn._13 ) * s;
		this->z = ( mIn._12 - mIn._21 ) * s;
	} 
	else 
	{
		if( mIn._11 > mIn._22 && mIn._11 > mIn._33 ) 
		{
			float s = 2.0f * sqrtf( 1.0f + mIn._11 - mIn._22 - mIn._33 );
			
			this->w = ( mIn._32 - mIn._23 ) / s;
			this->x = 0.25f * s;
			this->y = ( mIn._21 + mIn._12 ) / s;
			this->z = ( mIn._31 + mIn._13 ) / s;
		} 
		else 
			if( mIn._22 > mIn._33 ) 
			{
				float s = 2.0f * sqrtf( 1.0f + mIn._22 - mIn._11 - mIn._33 );
				
				this->w = ( mIn._31 - mIn._13 ) / s;
				this->x = ( mIn._21 + mIn._12 ) / s;
				this->y = 0.25f * s;
				this->z = ( mIn._32 + mIn._23 ) / s;    
			} 
			else 
			{
				float s = 2.0f * sqrtf( 1.0f + mIn._33 - mIn._11 - mIn._22 );
				
				this->w = ( mIn._21 - mIn._12 ) / s;
				this->x = ( mIn._31 + mIn._13 ) / s;
				this->y = ( mIn._32 + mIn._23 ) / s;
				this->z = 0.25f * s;
			}
	}
}
#include "Math.h"
#include "Matrix3.h"
#include "Ray.h"

#include <assert.h>
#include <limits>

const float Math::MAX_FLOAT = std::numeric_limits<float>::max();
const float Math::EPSILON_FLOAT = std::numeric_limits<float>::epsilon();
const float Math::PI = (float)(4.0*atan(1.0));
const float Math::HALF_PI = (float)(2.0*atan(1.0));
const float Math::DOUBLE_PI = (float)(8.0*atan(1.0));
const float Math::DEG_TO_RAD = Math::PI/180.0f;
const float Math::RAD_TO_DEG = 180.0f/Math::PI;


// -----------------------------------------------------------------------------
// -------------------------- Math::NextPowerOf2Value --------------------------
// -----------------------------------------------------------------------------
int Math::NextPowerOf2Value(int val)
{
	val--;
	int bCnt=0;

	while(val>0)
	{
		bCnt++;
		val= val>>1;
	}

	return 1<<bCnt;
}


// -----------------------------------------------------------------------------
// ------------------------------ Math::IsPowerOf2 -----------------------------
// -----------------------------------------------------------------------------
bool Math::IsPowerOf2(int val)
{
	return NextPowerOf2Value(val) == val;
}


// -----------------------------------------------------------------------------
// ------------------------------ Math::GetAngleY ------------------------------
// -----------------------------------------------------------------------------
float Math::GetAngleY(const Vector3& direction)
{
	float angleY = Math::PI * 2.0f - (atan2f(direction.z, direction.x) - atan2f(1.0f, 0.0f));

	while (angleY < 0.0f)
		angleY += 2.0f*PI;
	while (angleY > 2.0f*PI)
		angleY -= 2.0f*PI;

	return angleY;
}

// -----------------------------------------------------------------------------
// ------------------------ Math::GetDirectionFromAngleY -----------------------
// -----------------------------------------------------------------------------
Vector3 Math::GetDirectionFromAngleY(float angle)
{
	float sinAngle = Math::Sin(angle);
	float cosAngle = Math::Cos(angle);
	return Vector3(sinAngle, 0.0f, cosAngle);
}


// -----------------------------------------------------------------------------
// ------------------------------ Math::TruncFloat -----------------------------
// -----------------------------------------------------------------------------
float Math::Trunc(float val)
{
    return (float)((int)val);
}

// -----------------------------------------------------------------------------
// --------------------------------- Math::Frac --------------------------------
// -----------------------------------------------------------------------------
float Math::Frac(float val)
{
    return val - (int)val;
}

// -----------------------------------------------------------------------------
// -------------------------------- Math::Clamp --------------------------------
// -----------------------------------------------------------------------------
float Math::Clamp(float val, float min, float max)
{
	assert(min <= max);
	if (val < min)
		val = min;
	if (val > max)
		val = max;
	return val;
}

// -----------------------------------------------------------------------------
// -------------------------------- Math::Clamp --------------------------------
// -----------------------------------------------------------------------------
int Math::Clamp(int val, int min, int max)
{
	assert(min <= max);
	if (val < min)
		val = min;
	if (val > max)
		val = max;
	return val;
}


// -----------------------------------------------------------------------------
// ------------------------------- Math::InvSqrt -------------------------------
// -----------------------------------------------------------------------------
float Math::InvSqrt(float value)
{
	float xhalf = 0.5f*value;
	int i = *(int*)&value;						// get bits for floating value
	i = 0x5f375a86- (i>>1);						// gives initial guess y0
	value = *(float*)&i;						// convert bits back to float
	value = value*(1.5f-xhalf*value*value);		// Newton step, repeating increases accuracy
	return value;
}


// -----------------------------------------------------------------------------
// ----------------------- Math::DirectionToPitchHeading -----------------------
// -----------------------------------------------------------------------------
void Math::DirectionToPitchHeading(const Vector3& dir, float& pitch, float& heading)
{
	Vector3 dirNorm = dir;
	dirNorm.Normalize();

	Vector3 dirXZ = dirNorm;
	dirXZ.y = 0;
	dirXZ.Normalize();

	heading = atan2(dirNorm.x, dirNorm.z);

	while (heading < -Math::HALF_PI)
		heading += 2 * Math::PI;
	while (heading > Math::HALF_PI)
		heading -= 2 * Math::PI;

	float dot = dirNorm.DotProduct(dirXZ);
	pitch = fabs(acosf(dot));

	if (dir.y > 0)
		pitch = -pitch;
}

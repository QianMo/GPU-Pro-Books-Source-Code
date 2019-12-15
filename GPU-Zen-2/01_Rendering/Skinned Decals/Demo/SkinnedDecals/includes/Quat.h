#ifndef QUAT_H
#define QUAT_H

// Quat
//
// Unit quaternion.
class Quat
{
public:
  Quat()
  {
    x = y = z = 0.0f;
    w = 1.0f;
  }

  Quat(float x, float y, float z, float w)
  {
    this->x = x;
    this->y = y;
    this->z = z;
    this->w = w;
  }

  Quat(float x, float y, float z)
  {
    this->x = x;
    this->y = y;
    this->z = z;
    w = -sqrt(fabs(1.0f - (x * x) - (y * y) - (z * z)));
  }

  Quat(const Vector3 &V, float angle)
  {
    const float radians = DEG2RAD(angle);
    const float sinThetaDiv2 = sin(radians * 0.5f);
    x = V.x * sinThetaDiv2;
    y = V.y * sinThetaDiv2;
    z = V.z * sinThetaDiv2;
    w = cos(radians * 0.5f);
  }

  Quat operator* (const Quat &Q) const
  {
    Quat result;
    result.x = (w * Q.x) + (x * Q.w) + (y * Q.z) - (z * Q.y);
    result.y = (w * Q.y) + (y * Q.w) + (z * Q.x) - (x * Q.z);
    result.z = (w * Q.z) + (z * Q.w) + (x * Q.y) - (y * Q.x);
    result.w = (w * Q.w) - (x * Q.x) - (y * Q.y) - (z * Q.z);
    return result;
  }

  Vector3 operator* (const Vector3 &V) const
  {
    Quat q(V.x, V.y, V.z, 0.0f);
    Quat result = (*this) * q * (*this).GetConjugate();
    return Vector3(result.x, result.y, result.z);
  }

  void Conjugate()
  {
    x = -x;
    y = -y;
    z = -z;
  }

  Quat GetConjugate() const
  {
    return Quat(-x, -y, -z, w);
  }

  float GetLength() const
  {
    return (sqrt((x * x) + (y * y) + (z * z) + (w * w)));
  }

  void Normalize()
  {
    const float length = GetLength();
    if(length == 0.0f)
      return;
    const float inv = 1.0f / length;
    x *= inv;
    y *= inv;
    z *= inv;
    w *= inv;
  }

  void CreateMatrix(Matrix4 &matrix) const
  {
    matrix.entries[0] = 1.0f - (2.0f * (y * y + z * z));
    matrix.entries[1] = 2.0f * (x * y - w * z);
    matrix.entries[2] = 2.0f * (x * z + w * y);
    matrix.entries[3] = 0.0f;
    matrix.entries[4] = 2.0f * (x * y + w * z);
    matrix.entries[5] = 1.0f - (2.0f * (x * x + z * z));
    matrix.entries[6] = 2.0f * (y * z - w * x);
    matrix.entries[7] = 0.0f;
    matrix.entries[8] = 2.0f * (x * z - w * y);
    matrix.entries[9] = 2.0f * (y * z + w * x);
    matrix.entries[10] = 1.0f - (2.0f * (x * x + y * y));
    matrix.entries[11] = 0.0f;
    matrix.entries[12] = 0.0f;
    matrix.entries[13] = 0.0f;
    matrix.entries[14] = 0.0f;
    matrix.entries[15] = 1.0f;
  }

  void CreateFromMatrix(const Matrix4 &matrix)
  {
    const float diagonal = matrix.entries[0] + matrix.entries[5] + matrix.entries[10] + 1.0f;
    float scale = 0.0f;
    if(diagonal > 0.00000001f)
    {
      scale = sqrt(diagonal) * 2.0f;
      const float inv = 1.0f / scale;
      x = (matrix.entries[9] - matrix.entries[6]) * inv;
      y = (matrix.entries[2] - matrix.entries[8]) * inv;
      z = (matrix.entries[4] - matrix.entries[1]) * inv;
      w = 0.25f * scale;
    }
    else
    {
      if((matrix.entries[0] > matrix.entries[5]) && (matrix.entries[0] > matrix.entries[10]))
      {
        scale = sqrt(1.0f + matrix.entries[0] - matrix.entries[5] - matrix.entries[10]) * 2.0f;
        x = 0.25f * scale;
        const float inv = 1.0f / scale;
        y = (matrix.entries[4] + matrix.entries[1]) * inv;
        z = (matrix.entries[2] + matrix.entries[8]) * inv;
        w = (matrix.entries[9] - matrix.entries[6]) * inv;
      }
      else if(matrix.entries[5] > matrix.entries[10])
      {
        scale = sqrt(1.0f + matrix.entries[5] - matrix.entries[0] - matrix.entries[10]) * 2.0f;
        const float inv = 1.0f / scale;
        x = (matrix.entries[4] + matrix.entries[1]) * inv;
        y = 0.25f * scale;
        z = (matrix.entries[9] + matrix.entries[6]) * inv;
        w = (matrix.entries[2] - matrix.entries[8]) * inv;
      }
      else
      {
        scale = sqrt(1.0f + matrix.entries[10] - matrix.entries[0] - matrix.entries[5]) * 2.0f;
        const float inv = 1.0f / scale; 
        x = (matrix.entries[2] + matrix.entries[8]) * inv;
        y = (matrix.entries[9] + matrix.entries[6]) * inv;
        z = 0.25f * scale;
        w = (matrix.entries[4] - matrix.entries[1]) * inv;
      }
    }
  }

  Quat Slerp(const Quat &Q, float t) const
  {
    Quat interpolated;
    Quat quat(Q);
    if((x == quat.x) && (y == quat.y) && (z == quat.z) && (w == quat.w))
      return (*this);
    float result = (x * quat.x) + (y * quat.y) + (z * quat.z) + (w * quat.w);
    if(result < 0.0f)
    {
      quat = Quat(-quat.x, -quat.y, -quat.z, -quat.w);
      result = -result;
    }
    float scale0 = 1.0f - t;
    float scale1 = t;
    if((1.0f - result) > 0.1f)
    {
      const float theta = acos(result);
      const float invSinTheta = 1.0f / sin(theta);
      scale0 = sin((1.0f - t) * theta) * invSinTheta;
      scale1 = sin((t * theta)) * invSinTheta;
    }
    interpolated.x = (scale0 * x) + (scale1 * quat.x);
    interpolated.y = (scale0 * y) + (scale1 * quat.y);
    interpolated.z = (scale0 * z) + (scale1 * quat.z);
    interpolated.w = (scale0 * w) + (scale1 * quat.w);
    return interpolated;
  }

  float x, y, z, w;

};

#endif
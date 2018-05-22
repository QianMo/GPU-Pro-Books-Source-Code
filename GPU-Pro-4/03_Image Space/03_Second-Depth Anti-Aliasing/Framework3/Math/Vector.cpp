
/* * * * * * * * * * * * * Author's note * * * * * * * * * * * *\
*   _       _   _       _   _       _   _       _     _ _ _ _   *
*  |_|     |_| |_|     |_| |_|_   _|_| |_|     |_|  _|_|_|_|_|  *
*  |_|_ _ _|_| |_|     |_| |_|_|_|_|_| |_|     |_| |_|_ _ _     *
*  |_|_|_|_|_| |_|     |_| |_| |_| |_| |_|     |_|   |_|_|_|_   *
*  |_|     |_| |_|_ _ _|_| |_|     |_| |_|_ _ _|_|  _ _ _ _|_|  *
*  |_|     |_|   |_|_|_|   |_|     |_|   |_|_|_|   |_|_|_|_|    *
*                                                               *
*                     http://www.humus.name                     *
*                                                                *
* This file is a part of the work done by Humus. You are free to   *
* use the code in any way you like, modified, unmodified or copied   *
* into your own work. However, I expect you to respect these points:  *
*  - If you use this file and its contents unmodified, or use a major *
*    part of this file, please credit the author and leave this note. *
*  - For use in anything commercial, please request my approval.     *
*  - Share your work and ideas too as much as you can.             *
*                                                                *
\* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include "Vector.h"

half::half(const float x){
	union {
		float floatI;
		unsigned int i;
	};
	floatI = x;

//	unsigned int i = *((unsigned int *) &x);
	int e = ((i >> 23) & 0xFF) - 112;
	int m =  i & 0x007FFFFF;

	sh = (i >> 16) & 0x8000;
	if (e <= 0){
		// Denorm
		m = ((m | 0x00800000) >> (1 - e)) + 0x1000;
		sh |= (m >> 13);
	} else if (e == 143){
		sh |= 0x7C00;
		if (m != 0){
			// NAN
			m >>= 13;
			sh |= m | (m == 0);
		}
	} else {
		m += 0x1000;
		if (m & 0x00800000){
			// Mantissa overflow
			m = 0;
			e++;
		}
		if (e >= 31){
			// Exponent overflow
			sh |= 0x7C00;
		} else {
			sh |= (e << 10) | (m >> 13);
		}
	}
}

half::operator float () const {
	union {
		unsigned int s;
		float result;
	};

	s = (sh & 0x8000) << 16;
	unsigned int e = (sh >> 10) & 0x1F;
	unsigned int m = sh & 0x03FF;

	if (e == 0){
		// +/- 0
		if (m == 0) return result;

		// Denorm
		while ((m & 0x0400) == 0){
			m += m;
			e--;
		}
		e++;
		m &= ~0x0400;
	} else if (e == 31){
		// INF / NAN
		s |= 0x7F800000 | (m << 13);
		return result;
	}

	s |= ((e + 112) << 23) | (m << 13);

	return result;
}

/* --------------------------------------------------------------------------------- */

void vec2::operator += (const float s){
	x += s;
	y += s;
}

void vec2::operator += (const vec2 &v){
	x += v.x;
	y += v.y;
}

void vec2::operator -= (const float s){
	x -= s;
	y -= s;
}

void vec2::operator -= (const vec2 &v){
	x -= v.x;
	y -= v.y;
}

void vec2::operator *= (const float s){
	x *= s;
	y *= s;
}

void vec2::operator *= (const vec2 &v){
	x *= v.x;
	y *= v.y;
}

void vec2::operator /= (const float s){
	x /= s;
	y /= s;
}

void vec2::operator /= (const vec2 &v){
	x /= v.x;
	y /= v.y;
}

vec2 operator + (const vec2 &u, const vec2 &v){
	return vec2(u.x + v.x, u.y + v.y);
}

vec2 operator + (const vec2 &v, const float s){
	return vec2(v.x + s, v.y + s);
}

vec2 operator + (const float s, const vec2 &v){
	return vec2(v.x + s, v.y + s);
}

vec2 operator - (const vec2 &u, const vec2 &v){
	return vec2(u.x - v.x, u.y - v.y);
}

vec2 operator - (const vec2 &v, const float s){
	return vec2(v.x - s, v.y - s);
}

vec2 operator - (const float s, const vec2 &v){
	return vec2(s - v.x, s - v.y);
}

vec2 operator - (const vec2 &v){
	return vec2(-v.x, -v.y);
}

vec2 operator * (const vec2 &u, const vec2 &v){
	return vec2(u.x * v.x, u.y * v.y);
}

vec2 operator * (const float s, const vec2 &v){
	return vec2(v.x * s, v.y * s);
}

vec2 operator * (const vec2 &v, const float s){
	return vec2(v.x * s, v.y * s);
}

vec2 operator / (const vec2 &u, const vec2 &v){
	return vec2(u.x / v.x, u.y / v.y);
}

vec2 operator / (const vec2 &v, const float s){
	return vec2(v.x / s, v.y / s);
}

vec2 operator / (const float s, const vec2 &v){
	return vec2(s / v.x, s / v.y);
}

bool operator == (const vec2 &u, const vec2 &v){
	return (u.x == v.x && u.y == v.y);
}

/* --------------------------------------------------------------------------------- */

void vec3::operator += (const float s){
	x += s;
	y += s;
	z += s;
}

void vec3::operator += (const vec3 &v){
	x += v.x;
	y += v.y;
	z += v.z;
}

void vec3::operator -= (const float s){
	x -= s;
	y -= s;
	z -= s;
}

void vec3::operator -= (const vec3 &v){
	x -= v.x;
	y -= v.y;
	z -= v.z;
}

void vec3::operator *= (const float s){
	x *= s;
	y *= s;
	z *= s;
}

void vec3::operator *= (const vec3 &v){
	x *= v.x;
	y *= v.y;
	z *= v.z;
}

void vec3::operator /= (const float s){
	x /= s;
	y /= s;
	z /= s;
}

void vec3::operator /= (const vec3 &v){
	x /= v.x;
	y /= v.y;
	z /= v.z;
}

vec3 operator + (const vec3 &u, const vec3 &v){
	return vec3(u.x + v.x, u.y + v.y, u.z + v.z);
}

vec3 operator + (const vec3 &v, const float s){
	return vec3(v.x + s, v.y + s, v.z + s);
}

vec3 operator + (const float s, const vec3 &v){
	return vec3(v.x + s, v.y + s, v.z + s);
}

vec3 operator - (const vec3 &u, const vec3 &v){
	return vec3(u.x - v.x, u.y - v.y, u.z - v.z);
}

vec3 operator - (const vec3 &v, const float s){
	return vec3(v.x - s, v.y - s, v.z - s);
}

vec3 operator - (const float s, const vec3 &v){
	return vec3(s - v.x, s - v.y, s - v.z);
}

vec3 operator - (const vec3 &v){
	return vec3(-v.x, -v.y, -v.z);
}

vec3 operator * (const vec3 &u, const vec3 &v){
	return vec3(u.x * v.x, u.y * v.y, u.z * v.z);
}

vec3 operator * (const float s, const vec3 &v){
	return vec3(v.x * s, v.y * s, v.z * s);
}

vec3 operator * (const vec3 &v, const float s){
	return vec3(v.x * s, v.y * s, v.z * s);
}

vec3 operator / (const vec3 &u, const vec3 &v){
	return vec3(u.x / v.x, u.y / v.y, u.z / v.z);
}

vec3 operator / (const vec3 &v, const float s){
	return vec3(v.x / s, v.y / s, v.z / s);
}

vec3 operator / (const float s, const vec3 &v){
	return vec3(s / v.x, s / v.y, s / v.z);
}

bool operator == (const vec3 &u, const vec3 &v){
	return (u.x == v.x && u.y == v.y && u.z == v.z);
}

/* --------------------------------------------------------------------------------- */

void vec4::operator += (const float s){
	x += s;
	y += s;
	z += s;
	w += s;
}

void vec4::operator += (const vec4 &v){
	x += v.x;
	y += v.y;
	z += v.z;
	w += v.w;
}

void vec4::operator -= (const float s){
	x -= s;
	y -= s;
	z -= s;
	w -= s;
}

void vec4::operator -= (const vec4 &v){
	x -= v.x;
	y -= v.y;
	z -= v.z;
	w -= v.w;
}

void vec4::operator *= (const float s){
	x *= s;
	y *= s;
	z *= s;
	w *= s;
}

void vec4::operator *= (const vec4 &v){
	x *= v.x;
	y *= v.y;
	z *= v.z;
	w *= v.w;
}

void vec4::operator /= (const float s){
	x /= s;
	y /= s;
	z /= s;
	w /= s;
}

void vec4::operator /= (const vec4 &v){
	x /= v.x;
	y /= v.y;
	z /= v.z;
	w /= v.w;
}

vec4 operator + (const vec4 &u, const vec4 &v){
	return vec4(u.x + v.x, u.y + v.y, u.z + v.z, u.w + v.w);
}

vec4 operator + (const vec4 &v, const float s){
	return vec4(v.x + s, v.y + s, v.z + s, v.w + s);
}

vec4 operator + (const float s, const vec4 &v){
	return vec4(v.x + s, v.y + s, v.z + s, v.w + s);
}

vec4 operator - (const vec4 &u, const vec4 &v){
	return vec4(u.x - v.x, u.y - v.y, u.z - v.z, u.w - v.w);
}

vec4 operator - (const vec4 &v, const float s){
	return vec4(v.x - s, v.y - s, v.z - s, v.w - s);
}

vec4 operator - (const float s, const vec4 &v){
	return vec4(s - v.x, s - v.y, s - v.z, s - v.w);
}

vec4 operator - (const vec4 &v){
	return vec4(-v.x, -v.y, -v.z, -v.w);
}

vec4 operator * (const vec4 &u, const vec4 &v){
	return vec4(u.x * v.x, u.y * v.y, u.z * v.z, u.w * v.w);
}

vec4 operator * (const float s, const vec4 &v){
	return vec4(v.x * s, v.y * s, v.z * s, v.w * s);
}

vec4 operator * (const vec4 &v, const float s){
	return vec4(v.x * s, v.y * s, v.z * s, v.w * s);
}

vec4 operator / (const vec4 &u, const vec4 &v){
	return vec4(u.x / v.x, u.y / v.y, u.z / v.z, u.w / v.w);
}

vec4 operator / (const vec4 &v, const float s){
	return vec4(v.x / s, v.y / s, v.z / s, v.w / s);
}

vec4 operator / (const float s, const vec4 &v){
	return vec4(s / v.x, s / v.y, s / v.z, s / v.w);
}

bool operator == (const vec4 &u, const vec4 &v){
	return (u.x == v.x && u.y == v.y && u.z == v.z && u.w && v.w);
}

/* --------------------------------------------------------------------------------- */

float dot(const vec2 &u, const vec2 &v){
	return u.x * v.x + u.y * v.y;
}

float dot(const vec3 &u, const vec3 &v){
	return u.x * v.x + u.y * v.y + u.z * v.z;
}

float dot(const vec4 &u, const vec4 &v){
	return u.x * v.x + u.y * v.y + u.z * v.z + u.w * v.w;
}

float lerp(const float u, const float v, const float x){
	return u + x * (v - u);
}

vec2 lerp(const vec2 &u, const vec2 &v, const float x){
	return u + x * (v - u);
}

vec3 lerp(const vec3 &u, const vec3 &v, const float x){
	return u + x * (v - u);
}

vec4 lerp(const vec4 &u, const vec4 &v, const float x){
	return u + x * (v - u);
}

vec2 lerp(const vec2 &u, const vec2 &v, const vec2 &x){
	return u + x * (v - u);
}

vec3 lerp(const vec3 &u, const vec3 &v, const vec3 &x){
	return u + x * (v - u);
}

vec4 lerp(const vec4 &u, const vec4 &v, const vec4 &x){
	return u + x * (v - u);
}

float cerp(const float u0, const float u1, const float u2, const float u3, float x){
	float p = (u3 - u2) - (u0 - u1);
	float q = (u0 - u1) - p;
	float r = u2 - u0;
	return x * (x * (x * p + q) + r) + u1;
}

vec2 cerp(const vec2 &u0, const vec2 &u1, const vec2 &u2, const vec2 &u3, float x){
	vec2 p = (u3 - u2) - (u0 - u1);
	vec2 q = (u0 - u1) - p;
	vec2 r = u2 - u0;
	return x * (x * (x * p + q) + r) + u1;
}

vec3 cerp(const vec3 &u0, const vec3 &u1, const vec3 &u2, const vec3 &u3, float x){
	vec3 p = (u3 - u2) - (u0 - u1);
	vec3 q = (u0 - u1) - p;
	vec3 r = u2 - u0;
	return x * (x * (x * p + q) + r) + u1;
}

vec4 cerp(const vec4 &u0, const vec4 &u1, const vec4 &u2, const vec4 &u3, float x){
	vec4 p = (u3 - u2) - (u0 - u1);
	vec4 q = (u0 - u1) - p;
	vec4 r = u2 - u0;
	return x * (x * (x * p + q) + r) + u1;
}

float sign(const float v){
	return (v > 0)? 1.0f : (v < 0)? -1.0f : 0.0f;
}

vec2 sign(const vec2 &v){
	return vec2(sign(v.x), sign(v.y));
}

vec3 sign(const vec3 &v){
	return vec3(sign(v.x), sign(v.y), sign(v.z));
}

vec4 sign(const vec4 &v){
	return vec4(sign(v.x), sign(v.y), sign(v.z), sign(v.w));
}

float clamp(const float v, const float c0, const float c1){
	return min(max(v, c0), c1);
}

vec2 clamp(const vec2 &v, const float c0, const float c1){
	return vec2(min(max(v.x, c0), c1), min(max(v.y, c0), c1));
}

vec2 clamp(const vec2 &v, const vec2 &c0, const vec2 &c1){
	return vec2(min(max(v.x, c0.x), c1.x), min(max(v.y, c0.y), c1.y));
}

vec3 clamp(const vec3 &v, const float c0, const float c1){
	return vec3(min(max(v.x, c0), c1), min(max(v.y, c0), c1), min(max(v.z, c0), c1));
}

vec3 clamp(const vec3 &v, const vec3 &c0, const vec3 &c1){
	return vec3(min(max(v.x, c0.x), c1.x), min(max(v.y, c0.y), c1.y), min(max(v.z, c0.z), c1.z));
}

vec4 clamp(const vec4 &v, const float c0, const float c1){
	return vec4(min(max(v.x, c0), c1), min(max(v.y, c0), c1), min(max(v.z, c0), c1), min(max(v.z, c0), c1));
}

vec4 clamp(const vec4 &v, const vec4 &c0, const vec4 &c1){
	return vec4(min(max(v.x, c0.x), c1.x), min(max(v.y, c0.y), c1.y), min(max(v.z, c0.z), c1.z), min(max(v.w, c0.w), c1.w));
}

vec2 normalize(const vec2 &v){
	float invLen = 1.0f / sqrtf(v.x * v.x + v.y * v.y);
	return v * invLen;
}

vec3 normalize(const vec3 &v){
	float invLen = 1.0f / sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
	return v * invLen;
}

vec4 normalize(const vec4 &v){
	float invLen = 1.0f / sqrtf(v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w);
	return v * invLen;
}

vec2 fastNormalize(const vec2 &v){
	float invLen = rsqrtf(v.x * v.x + v.y * v.y);
	return v * invLen;
}

vec3 fastNormalize(const vec3 &v){
	float invLen = rsqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
	return v * invLen;
}

vec4 fastNormalize(const vec4 &v){
	float invLen = rsqrtf(v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w);
	return v * invLen;
}

float length(const vec2 &v){
	return sqrtf(v.x * v.x + v.y * v.y);
}

float length(const vec3 &v){
	return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

float length(const vec4 &v){
	return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w);
}

vec3 reflect(const vec3 &v, const vec3 &normal){
	float n = dot(v, normal);
	return v - 2 * n * normal;
}

float distance(const vec2 &u, const vec2 &v){
    vec2 d = u - v;
	return dot(d, d);
}

float distance(const vec3 &u, const vec3 &v){
    vec3 d = u - v;
	return sqrtf(dot(d, d));
}

float distance(const vec4 &u, const vec4 &v){
    vec4 d = u - v;
	return sqrtf(dot(d, d));
}

float planeDistance(const vec3 &normal, const float offset, const vec3 &point){
    return point.x * normal.x + point.y * normal.y + point.z * normal.z + offset;
}

float planeDistance(const vec4 &plane, const vec3 &point){
    return point.x * plane.x + point.y * plane.y + point.z * plane.z + plane.w;
}

float sCurve(const float t){
	return t * t * (3 - 2 * t);
}

vec3 cross(const vec3 &u, const vec3 &v){
	return vec3(u.y * v.z - v.y * u.z, u.z * v.x - u.x * v.z, u.x * v.y - u.y * v.x);
}

float lineProjection(const vec3 &line0, const vec3 &line1, const vec3 &point){
	vec3 v = line1 - line0;
	return dot(v, point - line0) / dot(v, v);
}

unsigned int toRGBA(const vec4 &u){
	return (int(u.x * 255) | (int(u.y * 255) << 8) | (int(u.z * 255) << 16) | (int(u.w * 255) << 24));
}

unsigned int toBGRA(const vec4 &u){
	return (int(u.z * 255) | (int(u.y * 255) << 8) | (int(u.x * 255) << 16) | (int(u.w * 255) << 24));
}

vec3 rgbeToRGB(unsigned char *rgbe){
	if (rgbe[3]){
		return vec3(rgbe[0], rgbe[1], rgbe[2]) * ldexpf(1.0f, rgbe[3] - (int) (128 + 8));
	} else return vec3(0, 0, 0);
}

unsigned int rgbToRGBE8(const vec3 &rgb){
	float v = max(rgb.x, rgb.y);
	v = max(v, rgb.z);

	if (v < 1e-32f){
		return 0;
	} else {
		int ex;
		float m = frexpf(v, &ex) * 256.0f / v;

		unsigned int r = (unsigned int) (m * rgb.x);
		unsigned int g = (unsigned int) (m * rgb.y);
		unsigned int b = (unsigned int) (m * rgb.z);
		unsigned int e = (unsigned int) (ex + 128);

		return r | (g << 8) | (b << 16) | (e << 24);
	}
}

unsigned int rgbToRGB9E5(const vec3 &rgb){
	float v = max(rgb.x, rgb.y);
	v = max(v, rgb.z);

	if (v < 1.52587890625e-5f){
		return 0;
	} else if (v < 65536){
		int ex;
		float m = frexpf(v, &ex) * 512.0f / v;

		unsigned int r = (unsigned int) (m * rgb.x);
		unsigned int g = (unsigned int) (m * rgb.y);
		unsigned int b = (unsigned int) (m * rgb.z);
		unsigned int e = (unsigned int) (ex + 15);

		return r | (g << 9) | (b << 18) | (e << 27);
	} else {
		unsigned int r = (rgb.x < 65536)? (unsigned int) (rgb.x * (1.0f / 128.0f)) : 0x1FF;
		unsigned int g = (rgb.y < 65536)? (unsigned int) (rgb.y * (1.0f / 128.0f)) : 0x1FF;
		unsigned int b = (rgb.z < 65536)? (unsigned int) (rgb.z * (1.0f / 128.0f)) : 0x1FF;
		unsigned int e = 31;

		return r | (g << 9) | (b << 18) | (e << 27);
	}
}

/* --------------------------------------------------------------------------------- */

mat2 operator + (const mat2 &m, const mat2 &n){
	return mat2(m.rows[0] + n.rows[0], m.rows[1] + n.rows[1]);
}

mat2 operator - (const mat2 &m, const mat2 &n){
	return mat2(m.rows[0] - n.rows[0], m.rows[1] - n.rows[1]);
}

mat2 operator - (const mat2 &m){
	return mat2(-m.rows[0], -m.rows[1]);
}

#define rcDot2(r, c) (m.rows[r].x * n.rows[0][c] + m.rows[r].y * n.rows[1][c])

mat2 operator * (const mat2 &m, const mat2 &n){
	return mat2(rcDot2(0, 0), rcDot2(0, 1), rcDot2(1, 0), rcDot2(1, 1));
}

vec2 operator * (const mat2 &m, const vec2 &v){
	return vec2(dot(m.rows[0], v), dot(m.rows[1], v));
}

mat2 operator * (const mat2 &m, const float x){
	return mat2(m.rows[0] * x, m.rows[1] * x);
}

mat2 transpose(const mat2 &m){
	return mat2(
		m.rows[0].x, m.rows[1].x,
		m.rows[0].y, m.rows[1].y);
}

float det(const mat2 &m){
	return (m.rows[0].x * m.rows[1].y - m.rows[0].y * m.rows[1].x);
}

mat2 operator ! (const mat2 &m){
	float invDet = 1.0f / det(m);

	return mat2(
		 m.rows[1].y, -m.rows[0].y,
		-m.rows[1].x,  m.rows[0].x) * invDet;
}

/* --------------------------------------------------------------------------------- */

mat3 operator + (const mat3 &m, const mat3 &n){
	return mat3(m.rows[0] + n.rows[0], m.rows[1] + n.rows[1], m.rows[2] + n.rows[2]);
}

mat3 operator - (const mat3 &m, const mat3 &n){
	return mat3(m.rows[0] - n.rows[0], m.rows[1] - n.rows[1], m.rows[2] - n.rows[2]);
}

mat3 operator - (const mat3 &m){
	return mat3(-m.rows[0], -m.rows[1], -m.rows[2]);
}

#define rcDot3(r, c) (m.rows[r].x * n.rows[0][c] + m.rows[r].y * n.rows[1][c] + m.rows[r].z * n.rows[2][c])

mat3 operator * (const mat3 &m, const mat3 &n){
	return mat3(
		rcDot3(0, 0), rcDot3(0, 1), rcDot3(0, 2),
		rcDot3(1, 0), rcDot3(1, 1), rcDot3(1, 2),
		rcDot3(2, 0), rcDot3(2, 1), rcDot3(2, 2));
}

vec3 operator * (const mat3 &m, const vec3 &v){
	return vec3(dot(m.rows[0], v), dot(m.rows[1], v), dot(m.rows[2], v));
}

mat3 operator * (const mat3 &m, const float x){
	return mat3(m.rows[0] * x, m.rows[1] * x, m.rows[2] * x);
}

mat3 transpose(const mat3 &m){
	return mat3(
		m.rows[0].x, m.rows[1].x, m.rows[2].x,
		m.rows[0].y, m.rows[1].y, m.rows[2].y,
		m.rows[0].z, m.rows[1].z, m.rows[2].z);
}

float det(const mat3 &m){
	return 
		m.rows[0].x * (m.rows[1].y * m.rows[2].z - m.rows[2].y * m.rows[1].z) -
		m.rows[0].y * (m.rows[1].x * m.rows[2].z - m.rows[1].z * m.rows[2].x) +
		m.rows[0].z * (m.rows[1].x * m.rows[2].y - m.rows[1].y * m.rows[2].x);
}

mat3 operator ! (const mat3 &m){
	float invDet = 1.0f / det(m);

	return mat3(
		m.rows[1].y * m.rows[2].z - m.rows[1].z * m.rows[2].y, m.rows[2].y * m.rows[0].z - m.rows[0].y * m.rows[2].z, m.rows[0].y * m.rows[1].z - m.rows[1].y * m.rows[0].z,
		m.rows[1].z * m.rows[2].x - m.rows[1].x * m.rows[2].z, m.rows[0].x * m.rows[2].z - m.rows[2].x * m.rows[0].z, m.rows[1].x * m.rows[0].z - m.rows[0].x * m.rows[1].z,
		m.rows[1].x * m.rows[2].y - m.rows[2].x * m.rows[1].y, m.rows[2].x * m.rows[0].y - m.rows[0].x * m.rows[2].y, m.rows[0].x * m.rows[1].y - m.rows[0].y * m.rows[1].x) * invDet;
}

/* --------------------------------------------------------------------------------- */

void mat4::translate(const vec3 &v){
	rows[0].w += dot(rows[0].xyz(), v);
	rows[1].w += dot(rows[1].xyz(), v);
	rows[2].w += dot(rows[2].xyz(), v);
	rows[3].w += dot(rows[3].xyz(), v);
}

mat4 operator + (const mat4 &m, const mat4 &n){
	return mat4(m.rows[0] + n.rows[0], m.rows[1] + n.rows[1], m.rows[2] + n.rows[2], m.rows[3] + n.rows[3]);
}

mat4 operator - (const mat4 &m, const mat4 &n){
	return mat4(m.rows[0] - n.rows[0], m.rows[1] - n.rows[1], m.rows[2] - n.rows[2], m.rows[3] - n.rows[3]);
}

mat4 operator - (const mat4 &m){
	return mat4(-m.rows[0], -m.rows[1], -m.rows[2], -m.rows[3]);
}

#define rcDot4(r, c) (m.rows[r].x * n.rows[0][c] + m.rows[r].y * n.rows[1][c] + m.rows[r].z * n.rows[2][c] + m.rows[r].w * n.rows[3][c])

mat4 operator * (const mat4 &m, const mat4 &n){
	return mat4(
		rcDot4(0, 0), rcDot4(0, 1), rcDot4(0, 2), rcDot4(0, 3),
		rcDot4(1, 0), rcDot4(1, 1), rcDot4(1, 2), rcDot4(1, 3),
		rcDot4(2, 0), rcDot4(2, 1), rcDot4(2, 2), rcDot4(2, 3),
		rcDot4(3, 0), rcDot4(3, 1), rcDot4(3, 2), rcDot4(3, 3));
}

vec4 operator * (const mat4 &m, const vec4 &v){
	return vec4(dot(m.rows[0], v), dot(m.rows[1], v), dot(m.rows[2], v), dot(m.rows[3], v));
}

mat4 operator * (const mat4 &m, const float x){
	return mat4(m.rows[0] * x, m.rows[1] * x, m.rows[2] * x, m.rows[3] * x);
}

mat4 transpose(const mat4 &m){
	return mat4(
		m.rows[0].x, m.rows[1].x, m.rows[2].x, m.rows[3].x,
		m.rows[0].y, m.rows[1].y, m.rows[2].y, m.rows[3].y,
		m.rows[0].z, m.rows[1].z, m.rows[2].z, m.rows[3].z,
		m.rows[0].w, m.rows[1].w, m.rows[2].w, m.rows[3].w);
}

mat4 operator ! (const mat4 &m){
	mat4 mat;

	float p00 = m.rows[2][2] * m.rows[3][3];
	float p01 = m.rows[3][2] * m.rows[2][3];
	float p02 = m.rows[1][2] * m.rows[3][3];
	float p03 = m.rows[3][2] * m.rows[1][3];
	float p04 = m.rows[1][2] * m.rows[2][3];
	float p05 = m.rows[2][2] * m.rows[1][3];
	float p06 = m.rows[0][2] * m.rows[3][3];
	float p07 = m.rows[3][2] * m.rows[0][3];
	float p08 = m.rows[0][2] * m.rows[2][3];
	float p09 = m.rows[2][2] * m.rows[0][3];
	float p10 = m.rows[0][2] * m.rows[1][3];
	float p11 = m.rows[1][2] * m.rows[0][3];

	mat.rows[0][0] = (p00 * m.rows[1][1] + p03 * m.rows[2][1] + p04 * m.rows[3][1]) - (p01 * m.rows[1][1] + p02 * m.rows[2][1] + p05 * m.rows[3][1]);
	mat.rows[0][1] = (p01 * m.rows[0][1] + p06 * m.rows[2][1] + p09 * m.rows[3][1]) - (p00 * m.rows[0][1] + p07 * m.rows[2][1] + p08 * m.rows[3][1]);
	mat.rows[0][2] = (p02 * m.rows[0][1] + p07 * m.rows[1][1] + p10 * m.rows[3][1]) - (p03 * m.rows[0][1] + p06 * m.rows[1][1] + p11 * m.rows[3][1]);
	mat.rows[0][3] = (p05 * m.rows[0][1] + p08 * m.rows[1][1] + p11 * m.rows[2][1]) - (p04 * m.rows[0][1] + p09 * m.rows[1][1] + p10 * m.rows[2][1]);
	mat.rows[1][0] = (p01 * m.rows[1][0] + p02 * m.rows[2][0] + p05 * m.rows[3][0]) - (p00 * m.rows[1][0] + p03 * m.rows[2][0] + p04 * m.rows[3][0]);
	mat.rows[1][1] = (p00 * m.rows[0][0] + p07 * m.rows[2][0] + p08 * m.rows[3][0]) - (p01 * m.rows[0][0] + p06 * m.rows[2][0] + p09 * m.rows[3][0]);
	mat.rows[1][2] = (p03 * m.rows[0][0] + p06 * m.rows[1][0] + p11 * m.rows[3][0]) - (p02 * m.rows[0][0] + p07 * m.rows[1][0] + p10 * m.rows[3][0]);
	mat.rows[1][3] = (p04 * m.rows[0][0] + p09 * m.rows[1][0] + p10 * m.rows[2][0]) - (p05 * m.rows[0][0] + p08 * m.rows[1][0] + p11 * m.rows[2][0]);

	float q00 = m.rows[2][0] * m.rows[3][1];
	float q01 = m.rows[3][0] * m.rows[2][1];
	float q02 = m.rows[1][0] * m.rows[3][1];
	float q03 = m.rows[3][0] * m.rows[1][1];
	float q04 = m.rows[1][0] * m.rows[2][1];
	float q05 = m.rows[2][0] * m.rows[1][1];
	float q06 = m.rows[0][0] * m.rows[3][1];
	float q07 = m.rows[3][0] * m.rows[0][1];
	float q08 = m.rows[0][0] * m.rows[2][1];
	float q09 = m.rows[2][0] * m.rows[0][1];
	float q10 = m.rows[0][0] * m.rows[1][1];
	float q11 = m.rows[1][0] * m.rows[0][1];

	mat.rows[2][0] = (q00 * m.rows[1][3] + q03 * m.rows[2][3] + q04 * m.rows[3][3]) - (q01 * m.rows[1][3] + q02 * m.rows[2][3] + q05 * m.rows[3][3]);
	mat.rows[2][1] = (q01 * m.rows[0][3] + q06 * m.rows[2][3] + q09 * m.rows[3][3]) - (q00 * m.rows[0][3] + q07 * m.rows[2][3] + q08 * m.rows[3][3]);
	mat.rows[2][2] = (q02 * m.rows[0][3] + q07 * m.rows[1][3] + q10 * m.rows[3][3]) - (q03 * m.rows[0][3] + q06 * m.rows[1][3] + q11 * m.rows[3][3]);
	mat.rows[2][3] = (q05 * m.rows[0][3] + q08 * m.rows[1][3] + q11 * m.rows[2][3]) - (q04 * m.rows[0][3] + q09 * m.rows[1][3] + q10 * m.rows[2][3]);
	mat.rows[3][0] = (q02 * m.rows[2][2] + q05 * m.rows[3][2] + q01 * m.rows[1][2]) - (q04 * m.rows[3][2] + q00 * m.rows[1][2] + q03 * m.rows[2][2]);
	mat.rows[3][1] = (q08 * m.rows[3][2] + q00 * m.rows[0][2] + q07 * m.rows[2][2]) - (q06 * m.rows[2][2] + q09 * m.rows[3][2] + q01 * m.rows[0][2]);
	mat.rows[3][2] = (q06 * m.rows[1][2] + q11 * m.rows[3][2] + q03 * m.rows[0][2]) - (q10 * m.rows[3][2] + q02 * m.rows[0][2] + q07 * m.rows[1][2]);
	mat.rows[3][3] = (q10 * m.rows[2][2] + q04 * m.rows[0][2] + q09 * m.rows[1][2]) - (q08 * m.rows[1][2] + q11 * m.rows[2][2] + q05 * m.rows[0][2]);

	return mat * (1.0f / (m.rows[0][0] * mat.rows[0][0] + m.rows[1][0] * mat.rows[0][1] + m.rows[2][0] * mat.rows[0][2] + m.rows[3][0] * mat.rows[0][3]));
}


/* --------------------------------------------------------------------------------- */

mat2 rotate(const float angle){
	float cosA = cosf(angle), sinA = sinf(angle);

	return mat2(cosA, -sinA, sinA, cosA);
}

mat4 rotateX(const float angle){
	float cosA = cosf(angle), sinA = sinf(angle);
	
	return mat4(
		1, 0,     0,    0,
		0, cosA, -sinA, 0,
		0, sinA,  cosA, 0,
		0, 0,     0,    1);
}

mat4 rotateY(const float angle){
	float cosA = cosf(angle), sinA = sinf(angle);

	return mat4(
		cosA, 0, -sinA, 0,
		0,    1,  0,    0,
		sinA, 0,  cosA, 0,
		0,    0,  0,    1);
}

mat4 rotateZ(const float angle){
	float cosA = cosf(angle), sinA = sinf(angle);

	return mat4(
		cosA, -sinA, 0, 0,
		sinA,  cosA, 0, 0,
		0,     0,    1, 0,
		0,     0,    0, 1);
}

mat4 rotateXY(const float angleX, const float angleY){
	float cosX = cosf(angleX), sinX = sinf(angleX), 
		  cosY = cosf(angleY), sinY = sinf(angleY);

	return mat4(
		 cosY,        0,    -sinY,        0,
		-sinX * sinY, cosX, -sinX * cosY, 0,
		 cosX * sinY, sinX,  cosX * cosY, 0,
		 0,           0,     0,           1);
}

mat4 rotateYX(const float angleX, const float angleY){
	float cosX = cosf(angleX), sinX = sinf(angleX), 
		  cosY = cosf(angleY), sinY = sinf(angleY);

	return mat4(
		cosY, -sinX * sinY, -cosX * sinY, 0,
		0,     cosX,        -sinX,        0,
		sinY,  sinX * cosY,  cosX * cosY, 0,
		0,     0,            0,           1);
}

mat4 rotateZXY(const float angleX, const float angleY, const float angleZ){
	float cosX = cosf(angleX), sinX = sinf(angleX), 
		  cosY = cosf(angleY), sinY = sinf(angleY),
		  cosZ = cosf(angleZ), sinZ = sinf(angleZ);

	return mat4(
		cosY * cosZ + sinX * sinY * sinZ,   -cosX * sinZ,    sinX * cosY * sinZ - sinY * cosZ,  0,
		cosY * sinZ - sinX * sinY * cosZ,    cosX * cosZ,   -sinY * sinZ - sinX * cosY * cosZ,  0,
		cosX * sinY,                         sinX,           cosX * cosY,                       0,
		0,                                   0,              0,                                 1);
}

mat4 translate(const vec3 &v){
	return mat4(1,0,0,v.x, 0,1,0,v.y, 0,0,1,v.z, 0,0,0,1);
}

mat4 translate(const float x, const float y, const float z){
	return mat4(1,0,0,x, 0,1,0,y, 0,0,1,z, 0,0,0,1);
}

mat4 scale(const float x, const float y, const float z){
	return mat4(x,0,0,0, 0,y,0,0, 0,0,z,0, 0,0,0,1);
}

mat4 perspectiveMatrix(const float fov, const float zNear, const float zFar){
	float s = cosf(0.5f * fov) / sinf(0.5f * fov);

	return mat4(
		s, 0, 0, 0,
		0, s, 0, 0,
		0, 0, (zFar + zNear) / (zFar - zNear), -(2 * zFar * zNear) / (zFar - zNear),
		0, 0, 1, 0);
}

mat4 perspectiveMatrixX(const float fov, const int width, const int height, const float zNear, const float zFar){
	float w = cosf(0.5f * fov) / sinf(0.5f * fov);
	float h = (w * width) / height;

	return mat4(
		w, 0, 0, 0,
		0, h, 0, 0,
		0, 0, (zFar + zNear) / (zFar - zNear), -(2 * zFar * zNear) / (zFar - zNear),
		0, 0, 1, 0);
}

mat4 perspectiveMatrixY(const float fov, const int width, const int height, const float zNear, const float zFar){
	float h = cosf(0.5f * fov) / sinf(0.5f * fov);
	float w = (h * height) / width;

	return mat4(
		w, 0, 0, 0,
		0, h, 0, 0,
		0, 0, (zFar + zNear) / (zFar - zNear), -(2 * zFar * zNear) / (zFar - zNear),
		0, 0, 1, 0);
}

mat4 orthoMatrixX(const float left, const float right, const float top, const float bottom, const float zNear, const float zFar){
	float rl = right - left;
	float tb = top - bottom;
	float fn = zFar - zNear;

	return mat4(
		2.0f / rl, 0,         0,         -(right + left) / rl,
		0,         2.0f / tb, 0,         -(top + bottom) / tb,
		0,         0,        -2.0f / fn, -(zFar + zNear) / fn,
		0,         0,         0,         1);
}

mat4 toD3DProjection(const mat4 &m){
	mat4 mat;

	mat.rows[0] = m.rows[0];
	mat.rows[1] = m.rows[1];
	mat.rows[2] = 0.5f * (m.rows[2] + m.rows[3]);
	mat.rows[3] = m.rows[3];

	return mat;
}

mat4 toGLProjection(const mat4 &m){
	mat4 mat;

	mat.rows[0] = m.rows[0];
	mat.rows[1] = m.rows[1];
	mat.rows[2] = m.rows[2] * 2.0 - m.rows[3];
	mat.rows[3] = m.rows[3];

	return mat;
}

mat4 pegToFarPlane(const mat4 &m){
	mat4 mat;

	mat.rows[0] = m.rows[0];
	mat.rows[1] = m.rows[1];
	mat.rows[2] = m.rows[3];
	mat.rows[3] = m.rows[3];

	return mat;
}

mat4 cubeViewMatrix(const unsigned int side){
	switch(side){
	case POSITIVE_X:
		return mat4(
			0, 0, -1, 0,
			0, 1,  0, 0,
			1, 0,  0, 0,
			0, 0,  0, 1);
	case NEGATIVE_X:
		return mat4(
			 0, 0, 1, 0,
			 0, 1, 0, 0,
			-1, 0, 0, 0,
			 0, 0, 0, 1);
	case POSITIVE_Y:
		return mat4(
			1, 0,  0, 0,
			0, 0, -1, 0,
			0, 1,  0, 0,
			0, 0,  0, 1);
	case NEGATIVE_Y:
		return mat4(
			1,  0, 0, 0,
			0,  0, 1, 0,
			0, -1, 0, 0,
			0,  0, 0, 1);
	case POSITIVE_Z:
		return mat4(
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1);
	//case NEGATIVE_Z:
	default:
		return mat4(
			-1, 0,  0, 0,
			 0, 1,  0, 0,
			 0, 0, -1, 0,
			 0, 0,  0, 1);
	}
}

mat4 cubeProjectionMatrixGL(const float zNear, const float zFar){
	return mat4(
		1,  0, 0, 0,
		0, -1, 0, 0,
		0,  0, (zFar + zNear) / (zFar - zNear), -(2 * zFar * zNear) / (zFar - zNear),
		0,  0, 1, 0);
}

mat4 cubeProjectionMatrixD3D(const float zNear, const float zFar){
	return mat4(
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, zFar / (zFar - zNear), (zFar * zNear) / (zNear - zFar),
		0, 0, 1, 0);
}

mat2 identity2(){
	return mat2(1,0, 0,1);
}

mat3 identity3(){
	return mat3(1,0,0, 0,1,0, 0,0,1);
}

mat4 identity4(){
	return mat4(1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1);
}

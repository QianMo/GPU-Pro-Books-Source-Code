// parametrics.glsl - public domain GLSL parametric surface generator
// author: Jonathan Dupuy (jdupuy@liris.cnrs.fr)
/* 
resources:
http://jalape.no/math/mathgal.htm
http://www.econym.demon.co.uk/isotut/maths.htm
*/

#if (__VERSION__ < 410)
#	error Uncompatible GLSL version
#endif
#ifndef PARAMETRICS_GLSL
#define PARAMETRICS_GLSL

// ==================================================
// Internal
// ==================================================
// constants
#ifndef PI
#	define PS_PI 1
#	define PI 3.14159265
#endif
#ifndef TWOPI
#	define PS_TWOPI 1
#	define TWOPI 6.283185307
#endif
#ifndef INV_TWOPI
#	define PS_INV_TWOPI 1
#	define INV_TWOPI 0.159154943
#endif

// ==================================================
// plane
void ps_plane(in vec2 parametricCoords,
              out vec3 position,
              out vec3 normal,
              out vec3 tangent,
              out vec3 bitangent) {
	position = vec3(parametricCoords - 0.5, 0.0);
	normal = vec3(0,0,1);
	tangent = vec3(0,1,0);
	bitangent = vec3(1,0,0);
}

// ==================================================
// sphere
void ps_sphere(in vec2 parametricCoords,
               out vec3 position,
               out vec3 normal,
               out vec3 tangent,
               out vec3 bitangent) {
	float theta = fma(parametricCoords.x, 0.998, 0.002) * PI; // avoids singularities at poles
	float phi   = parametricCoords.y * TWOPI;
	float x = sin(theta) * cos(phi);
	float y = cos(theta);
	float z = sin(theta) * sin(phi);
	normal = normalize(vec3(x,-y,z));
	position = normal * 0.5;

	float tx =  cos(theta) * cos(phi);
	float ty =  sin(theta);
	float tz =  cos(theta) * sin(phi);
	tangent = normalize(vec3(tx,ty,tz));

	float bx = -sin(theta) * sin(phi);
	float by =  0.0;
	float bz =  sin(theta) * cos(phi);
	bitangent = normalize(vec3(bx,by,bz));
}

// ==================================================
// cylinder
void ps_cylinder(in vec2 parametricCoords,
                 out vec3 position,
                 out vec3 normal,
                 out vec3 tangent,
                 out vec3 bitangent) {
	float theta = parametricCoords.x * TWOPI;
	position.x = parametricCoords.y - 0.5;
	position.y = cos(theta) * 0.5;
	position.z = sin(theta) * 0.5;
	tangent = normalize(vec3(0, -sin(theta), cos(theta)));
	bitangent = vec3(1,0,0);
	normal = normalize(vec3(0, position.yz));
}

// ==================================================
// torus
void ps_torus(in vec2 parametricCoords,
              out vec3 position,
              out vec3 normal,
              out vec3 tangent,
              out vec3 bitangent) {
	const float R = 0.3; // outer radius
	const float r = 0.1045; // inner radius
	float theta = parametricCoords.x * TWOPI;
	float phi   = parametricCoords.y * TWOPI;

	float x = fma(r, cos(phi), R) * cos(theta);
	float y = fma(r, cos(phi), R) * sin(theta);
	float z = r * sin(phi);
	position = vec3(x,y,z);

	float tx = -fma(r, cos(phi), R) * sin(theta);
	float ty =  fma(r, cos(phi), R) * cos(theta);
	float tz =  0.0;
	tangent = normalize(vec3(tx,ty,tz));

	float bx = -r * sin(phi) * cos(theta);
	float by = -r * sin(phi) * sin(theta);
	float bz =  r * cos(phi);
	bitangent = normalize(vec3(bx,by,bz));

	normal = normalize(cross(-bitangent, tangent));
}

// ==================================================
// helix
void ps_helix(in vec2 parametricCoords,
              out vec3 position,
              out vec3 normal,
              out vec3 tangent,
              out vec3 bitangent) {
	const float r = 0.35; // radius
	const float l = 3.0; // loops
	float s = (1.0 - parametricCoords.x) * TWOPI;
	float t = parametricCoords.y * TWOPI;

	float x = (1.0 - r * cos(t)) * cos(s*l);
	float y = (1.0 - r * cos(t)) * sin(s*l);
	float z = s * l * INV_TWOPI + r * sin(t);
	position = vec3(x,y,z);

	float tx = -(1.0 - r * cos(t)) * sin(s*l);
	float ty =  (1.0 - r * cos(t)) * cos(s*l);
	float tz = INV_TWOPI * l;
	tangent = normalize(vec3(tx,ty,tz));

	float bx = r * sin(t) * cos(s*l);
	float by = r * sin(t) * sin(s*l);
	float bz = r * cos(t);
	bitangent = normalize(vec3(bx,by,bz));

	normal = normalize(cross(bitangent, tangent));
}

// ==================================================
// trefoil knot (https://en.wikipedia.org/wiki/Trefoil_knot)
// http://mathdl.maa.org/images/upload_library/23/stemkoski/knots/page7.html
// useful read: Calculation of Reference Frames along a Space Curve
void ps_trefoil_knot(in vec2 parametricCoords,
                     out vec3 position,
                     out vec3 normal,
                     out vec3 tangent,
                     out vec3 bitangent) {
	const float R = 0.1; // outer radius
	const float r = 0.1545; // inner radius
	float theta = (parametricCoords.x) * TWOPI; // cylinder
	float t = parametricCoords.y * TWOPI;

	// trefoil knot equation
	float xk = r * cos(2.0*t) * (2.0 + cos(3.0*t));
	float yk = r * sin(2.0*t) * (2.0 + cos(3.0*t));
	float zk = r * sin(3.0*t) * 2.0;
	vec3 k = vec3(xk,yk,zk);

	// trefoil first derivative
	float dkx = -2.0 * r * sin(2.0*t) * (2.0 + cos(3.0*t))
	          - 3.0 * r * cos(2.0*t) * sin(3.0*t);
	float dky =  2.0 * r * cos(2.0*t) * (2.0 + cos(3.0*t))
	          - 3.0 * r * sin(2.0*t) * sin(3.0*t);
	float dkz = 3.0 * r * cos(3.0*t) * 2.0;
	vec3 dk = vec3(dkx, dky, dkz);

	// trefoil second derivative
	float ddkx = -4.0 * r * cos(2.0*t) * (2.0 + cos(3.0*t))
	           + 6.0 * r * sin(2.0*t) * sin(3.0*t)
	           + 6.0 * r * sin(2.0*t) * sin(3.0*t)
	           - 9.0 * r * cos(2.0*t) * cos(3.0*t);
	float ddky = -4.0 * r * sin(2.0*t) * (2.0 + cos(3.0*t))
	           - 6.0 * r * cos(2.0*t) * sin(3.0*t)
	           - 6.0 * r * cos(2.0*t) * sin(3.0*t)
	           - 9.0 * r * sin(2.0*t) * cos(3.0*t);
	float ddkz = -9.0 * r * sin(3.0*t) * 2.0;
	vec3 ddk = vec3(ddkx, ddky, ddkz); // produces too much torsion ...
	ddk = cross(vec3(0,0,1), dk); // so take another vector

	// frenet frame
	vec3 ft = normalize(dk);
	vec3 fn = normalize(ddk);
	vec3 fb = normalize(cross(ft, fn));
	mat3 f = mat3(ft,fn,fb);

	// generate circle and map to frame
	vec3 c = vec3(0, R * cos(theta), R * sin(theta));

	// final position
	position = (f * c + k)*0.7;

	// tangent
	tangent = ft;

	// bitangent
	float bx = -2.0 * r * sin(2.0*t) * (2.0 + cos(3.0*t))
	         - r * cos(2.0*t) * sin(3.0*t);
	float by =  2.0 * r * cos(2.0*t) * (2.0 + cos(3.0*t))
	         - r * sin(2.0*t) * sin(3.0*t);
	float bz = 3.0 * r * cos(3.0*t);
	bitangent = normalize(vec3(bx,by,bz));

	normal = normalize(cross(bitangent, tangent));
	normal = f * c;
}

// ==================================================
// horn http://paulbourke.net/geometry/horn/
void ps_horn(in vec2 parametricCoords,
             out vec3 position,
             out vec3 normal,
             out vec3 tangent,
             out vec3 bitangent) {
	const float a = 0.52; // radius
	const float c = 0.62; // inner radius
	const float n = 0.90; // spiral count
	const float b = 0.00; // height
	float s = (1.0 - parametricCoords.x) * TWOPI;
	float t = parametricCoords.y * 0.99 * TWOPI;
	float x = a * (1.0 - t * INV_TWOPI) * cos(n*t)
	        * (1.0 + cos(s)) + c * cos(n*t);
	float y = a * (1.0 - t * INV_TWOPI) * sin(n*t)
	        * (1.0 + cos(s)) + c * sin(n*t);
	float z = b * t * INV_TWOPI
	        + a * (1.0-t * INV_TWOPI) * sin(s);
	position = vec3(-y,x,z);

	float tx = a * (t * INV_TWOPI - 1.0) * cos(n*t) * sin(s);
	float ty = a * (t * INV_TWOPI - 1.0) * sin(n*t) * sin(s);
	float tz = a * (1.0 - t * INV_TWOPI) * cos(s);
	tangent = normalize(vec3(-ty,tx,tz));

	float bx = a * (1.0 + cos(s))
	         * (n * sin(n*t) * (t * INV_TWOPI - 1.0) - cos(n*t) * INV_TWOPI)
		     - c * n * sin(n*t);
	float by = a * (1.0 + cos(s))
	         * (n * cos(n*t) * (1.0 - t * INV_TWOPI) - sin(n*t) * INV_TWOPI)
		     + c * n * cos(n*t);
	float bz = b * INV_TWOPI - a * INV_TWOPI * sin(s);
	bitangent = normalize(vec3(-by,bx,bz));

	normal = normalize(cross(bitangent, tangent));
}

// ==================================================
// shell
// http://paulbourke.net/geometry/shell/
// http://www.foundalis.com/mat/hornsnail.htm
// http://www.econym.demon.co.uk/isotut/shells.htm
void ps_shell(in vec2 parametricCoords,
              out vec3 position,
              out vec3 normal,
              out vec3 tangent,
              out vec3 bitangent) {
	const float a = 0.22; // radius
	const float c = 0.06; // inner radius
	const float n = 1.20; // spiral count
	const float b = 0.40; // height
	float s = (1.0 - parametricCoords.x) * TWOPI;
	float t = parametricCoords.y * 0.99 * TWOPI;
	float x = a * (1.0 - t * INV_TWOPI) * cos(n*t)
	        * (1.0 + cos(s)) + c * cos(n*t);
	float y = a * (1.0 - t * INV_TWOPI) * sin(n*t)
	        * (1.0 + cos(s)) + c * sin(n*t);
	float z = b * t * INV_TWOPI
	        + a * (1.0-t * INV_TWOPI) * sin(s);
	position = vec3(x,y,z);

	float tx = a * (t * INV_TWOPI - 1.0) * cos(n*t) * sin(s);
	float ty = a * (t * INV_TWOPI - 1.0) * sin(n*t) * sin(s);
	float tz = a * (1.0 - t * INV_TWOPI) * cos(s);
	tangent = normalize(vec3(tx,ty,tz));

	float bx = a * (1.0 + cos(s))
	         * (n * sin(n*t) * (t * INV_TWOPI - 1.0) - cos(n*t) * INV_TWOPI)
		     - c * n * sin(n*t);
	float by = a * (1.0 + cos(s))
	         * (n * cos(n*t) * (1.0 - t * INV_TWOPI) - sin(n*t) * INV_TWOPI)
		     + c * n * cos(n*t);
	float bz = b * INV_TWOPI - a * INV_TWOPI * sin(s);
	bitangent = normalize(vec3(bx,by,bz));

	normal = normalize(cross(bitangent, tangent));
}

// ==================================================
// kleinbottle http://paulbourke.net/geometry/klein/
void ps_kleinbottle(in vec2 parametricCoords,
                    out vec3 position,
                    out vec3 normal,
                    out vec3 tangent,
                    out vec3 bitangent) {
	const float a = 2.0;
	const float n = 3.0;
	const float m = 1.0;
	float u = parametricCoords.x * TWOPI * 2.0;
	float v = parametricCoords.y * TWOPI;
	float x = cos(m*u*0.5)
	        * (a + cos(n*u*0.5) * sin(v) - sin(n*u*0.5) * sin(2.0*v));
	float y = sin(m*u*0.5)
	        * (a + cos(n*u*0.5) * sin(v) - sin(n*u*0.5) * sin(2.0*v));
	float z = sin(n*u*0.5) * sin(v) + cos(n*u*0.5) * sin(2.0*v);
	position = 0.1575 * vec3(x,y,z);

	float tx = cos(m*u*0.5)
	         * (-n * 0.5 * sin(n*u*0.5) * sin(v) - sin(2.0*v) * 0.5 * n * cos(n*u*0.5))
	         - m * 0.5 * sin(m*u*0.5)
	         * (a + cos(n*u*0.5) * sin(v) - sin(n*u*0.5) * sin(2.0*v));
	float ty = sin(m*u*0.5)
	         * (-n * 0.5 * sin(n*u*0.5) * sin(v) - sin(2.0*v) * 0.5 * n * cos(n*u*0.5))
	         + m * 0.5 * cos(m*u*0.5)
	         * (a + cos(n*u*0.5) * sin(v) - sin(n*u*0.5) * sin(2.0*v));
	float tz = n * 0.5 * sin(v) * cos(n*u*0.5)
	         - n * 0.5 * sin(2.0*v) * sin(n*u*0.5);
	tangent = normalize(vec3(tx,ty,tz));

	float bx = cos(m*u*0.5)
	         * (cos(n*u*0.5) * cos(v) - 2.0 * sin(n*u*0.5) * cos(2.0*v));
	float by = sin(m*u*0.5)
	         * (cos(n*u*0.5) * cos(v) - 2.0 * sin(n*u*0.5) * cos(2.0*v));
	float bz = sin(n*u*0.5) * cos(v) + 2.0 * cos(n*u*0.5) * cos(2.0*v);
	bitangent = normalize(vec3(bx,by,bz));

	normal  = normalize(cross(tangent, bitangent));
}

// ==================================================
// macro clean up
#if PS_PI
#	undef PI
#endif
#if PS_TWOPI
#	undef TWOPI
#endif
#if PS_INV_TWOPI
#	undef INV_TWOPI
#endif

#endif // SURFACES_GLSL


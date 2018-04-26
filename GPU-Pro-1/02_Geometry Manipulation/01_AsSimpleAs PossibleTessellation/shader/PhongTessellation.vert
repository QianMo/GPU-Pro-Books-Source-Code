// ---------------------------------------------------------
// Phong Tessellation Vertex Shader
// Author : Tamy Boubekeur (boubek@gmail.com).
// Copyright (C) 2008-2009 Tamy Boubekeur.
// All rights reserved.
// ---------------------------------------------------------

const float EPSILON = 0.0001;
uniform vec3 n0, n1, n2, n3, p0, p1, p2, p3;  // Input Coarse Polygon
uniform float alpha; // Shape Factor
uniform int quadMode; // Use bilinear interpolation over 4 vertices
uniform int quadID  ; // Quad upsampled using a 

varying vec4 P;
varying vec3 N;

// Linear (flat) Tessellation
vec3 LinearGeometry (float u, float v, float w) {
    vec3 p = w*p0 + u*p1 + v*p2;
    return p;
}

// ----------------------------
// Phong Tessellation operators
// ----------------------------

vec3 project (vec3 p, vec3 c, vec3 n) {
    return p - dot (p - c, n)*n;
} 

// Phong Tessellation for triangles
vec3 PhongGeometry (float u, float v, float w) {
    vec3 p = w*p0 + u*p1 + v*p2;
  
    vec3 c0 = project (p, p0, n0); 
    vec3 c1 = project (p, p1, n1); 
    vec3 c2 = project (p, p2, n2);
    
    vec3 q = w*c0 + u*c1 + v*c2;
    vec3 r = mix (p, q, alpha);
  
    return r;
}

vec3 bilinearInterpolation (float w, float u, vec3 p0, vec3 p1, vec3 p2, vec3 p3) {
  return u*(w*p0 + (1-w)*p1) + (1-u)*(w*p3+(1-w)*p2);
}

// Phong Tessellation for quads
vec3 quadPhongGeometry (float u, float v, float w) {
    vec3 p;
    if (quadID == 0)
        p = bilinearInterpolation (u, v, p0, p1, p2, p3);
    else
        p = bilinearInterpolation (u, v, p2, p3, p0, p1);
  
    vec3 c0 = project (p, p0, n0); 
    vec3 c1 = project (p, p1, n1); 
    vec3 c2 = project (p, p2, n2);
    vec3 c3 = project (p, p3, n3); 

    vec3 q;
    if (quadID == 0)
        q = bilinearInterpolation (u, v, c0, c1, c2, c3);
    else
        q = bilinearInterpolation (u, v, c2, c3, c0, c1);
    vec3 r = mix (p, q, alpha);
  
    return r;
}

// -------------------------------
// Other curved geometry operators
// -------------------------------

// Quadratic Patch operator for positions (equivalent to Phong Tessellation)
vec3 QuadraticGeometry (float u, float v, float w) {
    vec3 p = w*p0 + u*p1 + v*p2;
  
    vec3 p01 = (project (p0, p1, n1) + project (p1, p0, n0)); 
    vec3 p12 = (project (p1, p2, n2) + project (p2, p1, n1)); 
    vec3 p20 = (project (p2, p0, n0) + project (p0, p2, n2)); 
    vec3 q = w*(w*p0 + u*p01 + v*p20) + u*(u*p1 + v*p12) + v*v*p2;
    vec3 r = mix (p, q, alpha);
  
    return r;
}

vec3 evalPos (float u, float v, float w) {
    if (quadMode == 1)
        return quadPhongGeometry (u, v, w);
    else
        return PhongGeometry (u, v, w);
} 

// -------------
// Normal vector
// -------------

// Barycentric interpolation (Linear Phong over triangle)
vec3 PhongNormal (float u, float v, float w) {
    return normalize (w*n0 + u*n1 + v*n2);
}

// Bilinear interpolation (Linear Phong over quad)
vec3 quadPhongNormal (float u, float v, float w) {
    if (quadID == 0)
        return normalize (bilinearInterpolation (u, v, n0, n1, n2, n3));
    return normalize (bilinearInterpolation (u, v, n2, n3, n0, n1));
}

// Numerical differentials for normals
vec3 numDiffNormal (float u, float v, float w) {
    vec3 p = evalPos (u, v, w);
    vec3 du = evalPos (u+EPSILON, v, w-EPSILON) - p;
    vec3 dv = evalPos (u, v+EPSILON, w-EPSILON) - p;
    return normalize (cross (du, dv));
}

// Analytic differentials for normals
vec3 diffNormal (float u, float v, float w) {
    vec3 p01 = (project (p0, p1, n1) + project (p1, p0, n0)); 
    vec3 p12 = (project (p1, p2, n2) + project (p2, p1, n1)); 
    vec3 p20 = (project (p2, p0, n0) + project (p0, p2, n2)); 
    vec3 du = -p0 + p01 - p0 + 2.0*u*p0 + v*p0 - 2.0*u*p01 - v*p20 + v*p0 - v*p01 + 2.0*u*p1 + v*p12;
    vec3 dv = -p0 + p20 + u*p0 - u*p20 - p0 + u*p0 + 2.0*v*p0 - u*p01 - 2.0*v*p20 + u*p12 + 2.0*v*p2; 
    return normalize ( cross (du, dv));
}

vec3 evalNormal (float u, float v, float w) {
    if (normalMode == 0) {
        if (quadMode == 1)
            return quadPhongNormal (u, v, w);
        return PhongNormal (u, v, w);
    } else if (normalMode == 1)
        return diffNormal (u, v, w);
    return numDiffNormal (u, v, w);
}

// ---------------------------------------

void main(void)
{
    float u = gl_Vertex.y;
    float v = gl_Vertex.z;
    float w = gl_Vertex.x; // 1.0 - u - v
  
    gl_Vertex = vec4 (evalPos (u, v, w), gl_Vertex.w);
    gl_Normal = evalNormal (u, v, w);
  
    P = gl_Vertex;
    N = gl_Normal;
  
    gl_Position = ftransform ();
    gl_FrontColor = gl_Color;
}

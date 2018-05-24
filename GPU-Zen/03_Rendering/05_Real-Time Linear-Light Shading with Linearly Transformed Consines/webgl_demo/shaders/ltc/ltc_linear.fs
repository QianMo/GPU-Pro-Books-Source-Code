// bind roughness   {label:"Roughness", default:0.2, min:0.01, max:1, step:0.001}
// bind dcolor      {label:"Diffuse Color",  r:1.0, g:1.0, b:1.0}
// bind scolor      {label:"Specular Color", r:1.0, g:1.0, b:1.0}
// bind intensity   {label:"Light Intensity", default:10, min:0, max:20, step:1}
// bind L           {label:"Length", default: 10, min:0.1, max:15, step:0.1}
// bind R           {label:"Radius", default: 0.2, min:0.05, max:1, step:0.01}

// bind roty        {label:"Rotation Y", default: 0, min:0, max:1, step:0.001}
// bind rotz        {label:"Rotation Z", default: 0, min:0, max:1, step:0.001}

// bind analytic    {label:"Analytic", default:true}
// bind endCaps     {label:"End Caps", default:false}

uniform float roughness;
uniform vec3  dcolor;
uniform vec3  scolor;

uniform float intensity;
uniform float L;
uniform float R;

uniform bool analytic;
uniform bool endCaps;

uniform sampler2D ltc_mat;
uniform sampler2D ltc_mag;

uniform float roty;
uniform float rotz;

uniform mat4  view;
uniform vec2  resolution;

const float LUT_SIZE  = 64.0;
const float LUT_SCALE = (LUT_SIZE - 1.0)/LUT_SIZE;
const float LUT_BIAS  = 0.5/LUT_SIZE;

const float PI = 3.14159265;

// Math
////////////////

vec3 mul(mat3 m, vec3 v)
{
    return m * v;
}

mat3 mul(mat3 m1, mat3 m2)
{
    return m1 * m2;
}

vec3 rotation_y(vec3 v, float a)
{
    vec3 r;
    r.x =  v.x*cos(a) + v.z*sin(a);
    r.y =  v.y;
    r.z = -v.x*sin(a) + v.z*cos(a);
    return r;
}

vec3 rotation_z(vec3 v, float a)
{
    vec3 r;
    r.x =  v.x*cos(a) - v.y*sin(a);
    r.y =  v.x*sin(a) + v.y*cos(a);
    r.z =  v.z;
    return r;
}

vec3 rotation_yz(vec3 v, float ay, float az)
{
    return rotation_z(rotation_y(v, ay), az);
}

vec3 rotation_yz_inv(vec3 v, float ay, float az)
{
    return rotation_y(rotation_z(v, -az), -ay);
}

mat3 transpose(mat3 v)
{
    mat3 tmp;
    tmp[0] = vec3(v[0].x, v[1].x, v[2].x);
    tmp[1] = vec3(v[0].y, v[1].y, v[2].y);
    tmp[2] = vec3(v[0].z, v[1].z, v[2].z);

    return tmp;
}

// Cylinder helpers
////////////////

vec3 cylinderCenter()  {return vec3(0, 6, 32);}
vec3 cylinderTangent() {return rotation_yz(vec3(1, 0, 0), roty * 2.0*PI, rotz * 2.0*PI);}
vec3 cylinderP1()      {return cylinderCenter() - 0.5 * L * cylinderTangent();}
vec3 cylinderP2()      {return cylinderCenter() + 0.5 * L * cylinderTangent();}

// Camera functions
///////////////////

struct Ray
{
    vec3 origin;
    vec3 dir;
};

Ray GenerateCameraRay(float u1, float u2)
{
    Ray ray;

    vec2 xy = 2.0*(gl_FragCoord.xy)/resolution - vec2(1.0);

    ray.dir = normalize(vec3(xy, 2.0));

    float focalDistance = 2.0;
    float ft = focalDistance/ray.dir.z;
    vec3 pFocus = ray.dir*ft;

    ray.origin = vec3(0);
    ray.dir    = normalize(pFocus - ray.origin);

    // Apply camera transform
    ray.origin = (view*vec4(ray.origin, 1)).xyz;
    ray.dir    = (view*vec4(ray.dir,    0)).xyz;

    return ray;
}

bool RayPlaneIntersect(Ray ray, vec4 plane, out float t)
{
    t = -dot(plane, vec4(ray.origin, 1.0))/dot(plane.xyz, ray.dir);
    return t > 0.0;
}

bool quadratic(float a, float b, float c, out float t0, out float t1) {

    float d = b*b - 4.0*a*c;
    if (d < 0.0) return false;
    float sqrt_d = sqrt(d);
    float q;
    if (b < 0.0)
        q = -0.5 * (b - sqrt_d);
    else
        q = -0.5 * (b + sqrt_d);
    t0 = q / a;
    t1 = c / q;
    return true;
}

bool RayDisksIntersect(Ray ray, out float t)
{
    ray.origin -= cylinderCenter();

    ray.origin = rotation_yz_inv(ray.origin, roty * 2.0*PI, rotz * 2.0*PI);
    ray.dir    = rotation_yz_inv(ray.dir,    roty * 2.0*PI, rotz * 2.0*PI);

    if (abs(ray.dir.x) < 1e-7) return false;

    float t1, t2, height;
    bool d1 = false, d2 = false;

    t1 = (-0.5*L - ray.origin.x) / ray.dir.x;
    if (t1 > 0.0)
    {
        vec3 point = ray.origin + t1*ray.dir;
        float dist2 = point.y * point.y + point.z * point.z;
        if (dist2 < R * R)
            d1 = true;
    }

    t2 = (+0.5*L - ray.origin.x) / ray.dir.x;
    if (t2 > 0.0)
    {
        vec3 point = ray.origin + t2*ray.dir;
        float dist2 = point.y * point.y + point.z * point.z;
        if (dist2 < R * R)
            d2 = true;
    }

    if (d1 && d2)
    {
        t = min(t1, t2);
        return true;
    }
    if (d1)
    {
        t = t1;
        return true;
    }
    if (d2)
    {
        t = t2;
        return true;
    }

    return false;
}

bool RayCylinderIntersect(Ray ray, out float t)
{
    if (endCaps && RayDisksIntersect(ray, t))
        return true;

    ray.origin -= cylinderCenter();

    ray.origin = rotation_yz_inv(ray.origin, roty * 2.0*PI, rotz * 2.0*PI);
    ray.dir    = rotation_yz_inv(ray.dir,    roty * 2.0*PI, rotz * 2.0*PI);

    float A = ray.dir.z*ray.dir.z + ray.dir.y*ray.dir.y;
    float B = 2.0 * (ray.dir.z*ray.origin.z + ray.dir.y*ray.origin.y);
    float C = ray.origin.z*ray.origin.z + ray.origin.y*ray.origin.y - R*R;

    float t0, t1;
    if (!quadratic(A, B, C, t0, t1))
        return false;

    if (t0 < 0.0 && t1 < 0.0)
        return false;

    t = min(t0, t1);
    if (t0 < 0.0)
        t = t0;
    if (t1 < 0.0)
        t = t1;

    // intersection
    vec3 point = ray.origin + t * ray.dir;

    if (abs(point.x) > 0.5*L)
        return false;

    return true;
}

// code from [Frisvad2012]
void buildOrthonormalBasis(
in vec3 n, out vec3 b1, out vec3 b2)
{
    if (n.z < -0.9999999)
    {
        b1 = vec3( 0.0, -1.0, 0.0);
        b2 = vec3(-1.0,  0.0, 0.0);
        return;
    }
    float a = 1.0 / (1.0 + n.z);
    float b = -n.x*n.y*a;
    b1 = vec3(1.0 - n.x*n.x*a, b, -n.x);
    b2 = vec3(b, 1.0 - n.y*n.y*a, -n.y);
}

float determinant(mat3 m)
{
    return + m[0][0]*(m[1][1]*m[2][2] - m[2][1]*m[1][2])
           - m[1][0]*(m[0][1]*m[2][2] - m[2][1]*m[0][2])
           + m[2][0]*(m[0][1]*m[1][2] - m[1][1]*m[0][2]);
}

mat3 inverse(mat3 m)
{
    float a00 = m[0][0], a01 = m[0][1], a02 = m[0][2];
    float a10 = m[1][0], a11 = m[1][1], a12 = m[1][2];
    float a20 = m[2][0], a21 = m[2][1], a22 = m[2][2];

    float b01 =  a22 * a11 - a12 * a21;
    float b11 = -a22 * a10 + a12 * a20;
    float b21 =  a21 * a10 - a11 * a20;

    float det = a00 * b01 + a01 * b11 + a02 * b21;

    return mat3(b01, (-a22 * a01 + a02 * a21), ( a12 * a01 - a02 * a11),
                b11, ( a22 * a00 - a02 * a20), (-a12 * a00 + a02 * a10),
                b21, (-a21 * a00 + a01 * a20), ( a11 * a00 - a01 * a10)) / det;
}

mat3 Minv;
float D(vec3 w)
{
    vec3 wo = Minv * w;
    float lo = length(wo);
    float res = 1.0/PI * max(0.0, wo.z/lo) * abs(determinant(Minv)) / (lo*lo*lo);
    return res;
}

float I_cylinder_numerical(vec3 p1, vec3 p2, float R)
{
    // init orthonormal basis
    float L = length(p2 - p1);
    vec3 wt = normalize(p2 - p1);
    vec3 wt1, wt2;
    buildOrthonormalBasis(wt, wt1, wt2);

    // integral discretization
    float I = 0.0;
    const int nSamplesphi = 20;
    const int nSamplesl   = 100;
    for (int i = 0; i < nSamplesphi; ++i)
    for (int j = 0; j < nSamplesl;   ++j)
    {
        // normal
        float phi = 2.0 * PI * float(i)/float(nSamplesphi);
        vec3 wn = cos(phi)*wt1 + sin(phi)*wt2;

        // position
        float l = L * float(j)/float(nSamplesl - 1);
        vec3 p = p1 + l*wt + R*wn;

        // normalized direction
        vec3 wp = normalize(p);

        // integrate
        I += D(wp) * max(0.0, dot(-wp, wn)) / dot(p, p);
    }

    I *= 2.0 * PI * R * L / float(nSamplesphi*nSamplesl);
    return I;
}

float Fpo(float d, float l)
{
    return l/(d*(d*d + l*l)) + atan(l/d)/(d*d);
}

float Fwt(float d, float l)
{
    return l*l/(d*(d*d + l*l));
}

float I_diffuse_line(vec3 p1, vec3 p2)
{
    // tangent
    vec3 wt = normalize(p2 - p1);

    // clamping
    if (p1.z <= 0.0 && p2.z <= 0.0) return 0.0;
    if (p1.z < 0.0) p1 = (+p1*p2.z - p2*p1.z) / (+p2.z - p1.z);
    if (p2.z < 0.0) p2 = (-p1*p2.z + p2*p1.z) / (-p2.z + p1.z);

    // parameterization
    float l1 = dot(p1, wt);
    float l2 = dot(p2, wt);

    // shading point orthonormal projection on the line
    vec3 po = p1 - l1*wt;

    // distance to line
    float d = length(po);

    // integral
    float I = (Fpo(d, l2) - Fpo(d, l1)) * po.z +
              (Fwt(d, l2) - Fwt(d, l1)) * wt.z;
    return I / PI;
}

float I_ltc_line(vec3 p1, vec3 p2)
{
    // transform to diffuse configuration
    vec3 p1o = Minv * p1;
    vec3 p2o = Minv * p2;
    float I_diffuse = I_diffuse_line(p1o, p2o);

    // width factor
    vec3 ortho = normalize(cross(p1, p2));
    float w =  1.0 / length(inverse(transpose(Minv)) * ortho);

    return w * I_diffuse;
}

float I_disks_numerical(vec3 p1, vec3 p2, float R)
{
    // init orthonormal basis
    float L = length(p2 - p1);
    vec3 wt = normalize(p2 - p1);
    vec3 wt1, wt2;
    buildOrthonormalBasis(wt, wt1, wt2);

    // integration
    float Idisks = 0.0;
    const int nSamplesphi = 20;
    const int nSamplesr   = 200;
    for (int i = 0; i < nSamplesphi; ++i)
    for (int j = 0; j < nSamplesr;   ++j)
    {
        float phi = 2.0 * PI * float(i)/float(nSamplesphi);
        float r = R * float(j)/float(nSamplesr - 1);
        vec3 p, wp;

        p = p1 + r * (cos(phi)*wt1 + sin(phi)*wt2);
        wp = normalize(p);
        Idisks += r * D(wp) * max(0.0, dot(wp, +wt)) / dot(p, p);

        p = p2 + r * (cos(phi)*wt1 + sin(phi)*wt2);
        wp = normalize(p);
        Idisks += r * D(wp) * max(0.0, dot(wp, -wt)) / dot(p, p);
    }

    Idisks *= 2.0 * PI * R / float(nSamplesr*nSamplesphi);
    return Idisks;
}

float I_ltc_disks(vec3 p1, vec3 p2, float R)
{
    float A = PI * R * R;
    vec3 wt  = normalize(p2 - p1);
    vec3 wp1 = normalize(p1);
    vec3 wp2 = normalize(p2);
    float Idisks = A * (
    D(wp1) * max(0.0, dot(+wt, wp1)) / dot(p1, p1) +
    D(wp2) * max(0.0, dot(-wt, wp2)) / dot(p2, p2));
    return Idisks;
}

vec3 LTC_Evaluate(vec3 N, vec3 V, vec3 P)
{
    // construct orthonormal basis around N
    vec3 T1, T2;
    T1 = normalize(V - N*dot(V, N));
    T2 = cross(N, T1);

    mat3 B = transpose(mat3(T1, T2, N));

    vec3 p1 = mul(B, cylinderP1() - P);
    vec3 p2 = mul(B, cylinderP2() - P);

    if (analytic) // analytic integration
    {
        float Iline = R * I_ltc_line(p1, p2);
        float Idisks = endCaps ? I_ltc_disks(p1, p2, R) : 0.0;
        return vec3(min(1.0, Iline + Idisks));
    }
    else // numerical integration
    {
        float Icylinder = I_cylinder_numerical(p1, p2, R);
        float Idisks = endCaps ? I_disks_numerical(p1, p2, R) : 0.0;
        return vec3(Icylinder + Idisks);
    }
}

// Misc. helpers
////////////////

float saturate(float v)
{
    return clamp(v, 0.0, 1.0);
}

vec3 PowVec3(vec3 v, float p)
{
    return vec3(pow(v.x, p), pow(v.y, p), pow(v.z, p));
}

const float gamma = 2.2;

vec3 ToLinear(vec3 v) { return PowVec3(v,     gamma); }
vec3 ToSRGB(vec3 v)   { return PowVec3(v, 1.0/gamma); }

// Main
////////////////

void main()
{
    vec4 floorPlane = vec4(0, 1, 0, 0);

    vec3 lcol = vec3(intensity);
    vec3 dcol = ToLinear(dcolor);
    vec3 scol = ToLinear(scolor);

    vec3 col = vec3(0);

    {
        Ray ray = GenerateCameraRay(0.0, 0.0);

        float distToFloor;
        bool hitFloor = RayPlaneIntersect(ray, floorPlane, distToFloor);
        if (hitFloor)
        {
            vec3 pos = ray.origin + ray.dir*distToFloor;

            vec3 N = floorPlane.xyz;
            vec3 V = -ray.dir;

            float theta = acos(dot(N, V));
            vec2 uv = vec2(roughness, theta/(0.5*PI));
            uv = uv*LUT_SCALE + LUT_BIAS;

            vec4 t = texture2D(ltc_mat, uv);
            Minv = mat3(
                vec3(  1,   0, t.y),
                vec3(  0, t.z,   0),
                vec3(t.w,   0, t.x)
            );

            vec3 spec = LTC_Evaluate(N, V, pos);
            spec *= texture2D(ltc_mag, uv).w;

            Minv = mat3(1);
            vec3 diff = LTC_Evaluate(N, V, pos);

            col  = lcol*(scol*spec + dcol*diff);
            col /= 2.0*PI;
        }

        float distToCylinder;
        if (RayCylinderIntersect(ray, distToCylinder))
            if ((distToCylinder < distToFloor) || !hitFloor)
                col = lcol;
    }

    gl_FragColor = vec4(col, 1.0);
}
















// Anisotropic Kuwahara Filtering on the GPU
// by Jan Eric Kyprianidis <www.kyprianidis.com>
#version 120
#extension GL_EXT_gpu_shader4 : enable

uniform sampler2D tfm;
uniform sampler2D src;
uniform float sigma;
varying out vec3 dst;

struct lic_t { 
    vec2 p; 
    vec2 t;
    float w;
    float dw;
};

void step(inout lic_t s) {
    vec2 t = texture2D(tfm, s.p).xy;
    if (dot(t, s.t) < 0.0) t = -t;
    s.t = t;

    s.dw = (abs(t.x) > abs(t.y))? 
        abs((fract(s.p.x) - 0.5 - sign(t.x)) / t.x) : 
        abs((fract(s.p.y) - 0.5 - sign(t.y)) / t.y);

    s.p += t * s.dw / vec2(textureSize2D(src, 0));
    s.w += s.dw;
}

void main (void) {
    float twoSigma2 = 2.0 * sigma * sigma;
    float halfWidth = 2.0 * sigma;
    vec2 uv = gl_FragCoord.xy / vec2(textureSize2D(src, 0));

    vec3 c = texture2D( src, uv ).xyz;
    float w = 1.0;

    lic_t a, b;
    a.p = b.p = uv;
    a.t = texture2D( tfm, uv ).xy / vec2(textureSize2D(src, 0));
    b.t = -a.t;
    a.w = b.w = 0.0;

    while (a.w < halfWidth) {
        step(a);
        float k = a.dw * exp(-a.w * a.w / twoSigma2);
        c += k * texture2D(src, a.p).xyz;
        w += k;
    }
    while (b.w < halfWidth) {
        step(b);
        float k = b.dw * exp(-b.w * b.w / twoSigma2);
        c += k * texture2D(src, b.p).xyz;
        w += k;
    }
    
    dst = c / w;
}

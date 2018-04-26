// Anisotropic Kuwahara Filtering on the GPU
// by Jan Eric Kyprianidis <www.kyprianidis.com>
#version 120
#extension GL_EXT_gpu_shader4 : enable

uniform sampler2D src;
uniform sampler2D K0;
uniform int radius;
uniform float q;

const float PI = 3.14159265358979323846;
const int N = 8;

void main (void) {
    vec2 src_size = vec2(textureSize2D(src, 0));
    vec2 uv = gl_FragCoord.xy / src_size;

    vec4 m[8];
    vec3 s[8];
    for (int k = 0; k < N; ++k) {
        m[k] = vec4(0.0);
        s[k] = vec3(0.0);
    }

    float piN = 2.0 * PI / float(N);
    mat2 X = mat2(cos(piN), sin(piN), -sin(piN), cos(piN));

    for ( int j = -radius; j <= radius; ++j ) {
        for ( int i = -radius; i <= radius; ++i ) {
            vec2 v = 0.5 * vec2(i,j) / float(radius);
            if (dot(v,v) <= 0.25) {
                vec4 c_fix = texture2D(src, uv + vec2(i,j) / src_size);
                vec3 c = c_fix.rgb; 
                for (int k = 0; k < N; ++k) {
                    float w = texture2D(K0, vec2(0.5, 0.5) + v).x;

                    m[k] += vec4(c * w, w);
                    s[k] += c * c * w;

                    v *= X;
                }
            }
        }
    }

    vec4 o = vec4(0.0);
    for (int k = 0; k < N; ++k) {
        m[k].rgb /= m[k].w;
        s[k] = abs(s[k] / m[k].w - m[k].rgb * m[k].rgb);

        float sigma2 = s[k].r + s[k].g + s[k].b;
        float w = 1.0 / (1.0 + pow(255.0 * sigma2, 0.5 * q));

        o += vec4(m[k].rgb * w, w);
    }

    gl_FragColor = vec4(o.rgb / o.w, 1.0);
}

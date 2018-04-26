// Anisotropic Kuwahara Filtering on the GPU
// by Jan Eric Kyprianidis <www.kyprianidis.com>
#version 120
#extension GL_EXT_gpu_shader4 : enable

uniform sampler2D src;
uniform float sigma;
varying out vec3 dst;

void main (void) {
    vec2 src_size = vec2(textureSize2D(src, 0));
    vec2 uv = gl_FragCoord.xy / src_size;
    float twoSigma2 = 2.0 * sigma * sigma;
    int halfWidth = int(ceil( 2.0 * sigma ));

    vec3 sum = vec3(0.0);
    float norm = 0.0;
    if (halfWidth > 0) {
        for ( int i = -halfWidth; i <= halfWidth; ++i ) {
            for ( int j = -halfWidth; j <= halfWidth; ++j ) {
                float d = length(vec2(i,j));
                float kernel = exp( -d *d / twoSigma2 );
                vec3 c = texture2D(src, uv + vec2(i,j) / src_size ).rgb;
                sum += kernel * c;
                norm += kernel;
            }
        }
    } else {
        sum = texture2D(src, uv).rgb;
        norm = 1.0;
    }
    dst = sum / norm;
}

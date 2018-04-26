// Anisotropic Kuwahara Filtering on the GPU
// by Jan Eric Kyprianidis <www.kyprianidis.com>
#version 120
#extension GL_EXT_gpu_shader4 : enable

uniform sampler2D tfm;
uniform sampler2D jet;

void main (void) {
    vec2 uv = gl_FragCoord.xy / vec2(textureSize2D(tfm, 0));
    vec4 t = texture2D( tfm, uv );
    gl_FragColor = texture2D( jet, vec2(t.w, 0.5));
}

//#version 120
//#extension GL_EXT_gpu_shader4 : require  // texelFetch

uniform sampler2D srcTex;
uniform float texDelta; // 1.0 / texWidth

varying out vec3 sum;

void main()
{

   // access bigger texture from this frag coord
	vec2 coord = (gl_FragCoord.st * 2.0 - 0.5) * texDelta;
	sum = texture2D(srcTex, coord).rgb;

	// rechts
	coord.s = (gl_FragCoord.s * 2.0 - 0.5 + 1.0) * texDelta;
	coord.t = (gl_FragCoord.t * 2.0 - 0.5 ) * texDelta;
	sum += texture2D(srcTex, coord).rgb;

	// oben
	coord.s = (gl_FragCoord.s * 2.0 - 0.5 ) * texDelta;
	coord.t = (gl_FragCoord.t * 2.0 - 0.5 + 1.0) * texDelta;
	sum += texture2D(srcTex, coord).rgb;

	// rechts oben
	coord.s = (gl_FragCoord.s * 2.0 - 0.5 + 1.) * texDelta;
	coord.t = (gl_FragCoord.t * 2.0 - 0.5 + 1.) * texDelta;
	sum += texture2D(srcTex, coord).rgb;

   sum /= 4.0;


 //  ivec2 coord = ivec2(gl_FragCoord.st);

	//vec3 val0 = texelFetch2D(srcTex, coord, 0);
	//vec3 val1 = texelFetch2D(srcTex, coord + ivec2(0, 1), 0);
	//vec3 val2 = texelFetch2D(srcTex, coord + ivec2(1, 0), 0);
	//vec3 val3 = texelFetch2D(srcTex, coord + ivec2(1, 1), 0);

 //  sum = (val0 + val1 + val2 + val3);


}
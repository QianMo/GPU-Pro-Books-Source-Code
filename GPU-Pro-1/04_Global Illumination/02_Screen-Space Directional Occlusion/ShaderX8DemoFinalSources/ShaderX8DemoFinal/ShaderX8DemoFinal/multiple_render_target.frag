
varying vec4 position;      // interpolated world space position
varying vec3 normal;        // interpolated world space normal
varying vec4 color;

// multiple render target: output position, normal and color buffer

void main()
{	
	gl_FragData[0] = position;
	gl_FragData[1] = vec4(normal, 0.0);
	gl_FragData[2] = color;
}

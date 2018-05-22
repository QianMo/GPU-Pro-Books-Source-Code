///////////////////////////////////////////////////////////////////////
// Real-time Near-Field Global Illumination based on a Voxel Model
// GPU Pro 3
///////////////////////////////////////////////////////////////////////
// Authors : Sinje Thiedemann, Niklas Henrich, 
//           Thorsten Grosch, Stefan Mueller
// Created:  August 2011
///////////////////////////////////////////////////////////////////////

varying vec3 P;

void main()
{	
   // Transform atlas texture coordinate into NDC
	gl_Position = vec4((gl_MultiTexCoord0.xy * 2.0) - vec2(1.0), 0.0, 1.0);

	// Pass world-space position to fragment shader
	P = (gl_ModelViewMatrix * gl_Vertex).xyz;
}
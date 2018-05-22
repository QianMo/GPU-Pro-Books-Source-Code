#version 120 // because of mat3(mat4) cast

varying vec3 P;
varying vec3 N;
varying vec3 P_eye;

uniform mat4 inverseViewMatrix;

void main()
{
	gl_Position = ftransform();	// fixed function pipeline functionality
	gl_TexCoord[0] = gl_MultiTexCoord0;

	// transform normal and position to world space
	N = mat3(inverseViewMatrix) * gl_NormalMatrix * gl_Normal; // normalize in fragment shader
   vec4 camPos = gl_ModelViewMatrix * gl_Vertex;
   P_eye = camPos.xyz;
	P = (inverseViewMatrix * camPos).xyz; 

   gl_FrontColor = gl_Color;
}
#version 120 // because of mat3(mat4) cast

varying vec3 P;
varying vec3 N;

varying vec3 P_world;
varying vec3 N_world;

uniform mat4 viewMatrix;

void main()
{
	gl_Position = gl_ProjectionMatrix * viewMatrix * gl_ModelViewMatrix * gl_Vertex;	
	gl_TexCoord[0] = gl_MultiTexCoord0;

	// transform normal and position to world space
	N_world = gl_NormalMatrix * gl_Normal; // normalize in fragment shader
   N = (viewMatrix * vec4(N_world, 0)).xyz;
	P_world = (gl_ModelViewMatrix * gl_Vertex).xyz; 
   P = (viewMatrix * vec4(P_world, 1.0)).xyz;

   //// back facing?
   //if(dot(normalize(N), normalize(P)) > 0.0)
   //{
   //   N *= -1.0;
   //   N_world *= -1.0;
   //}

}
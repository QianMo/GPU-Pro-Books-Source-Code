#version 120

varying vec3 P;
varying vec3 N;
varying vec3 P_eye; 

uniform sampler2D diffuseTexture; 

varying out vec4 position;
varying out vec4 normal;
varying out vec4 material; // rhoDiffuse / PI

void main()
{
   // POSITION
   position = vec4(P, abs(P_eye.z)/*length(P_eye)*/);

   // NORMAL
   normal = vec4(normalize(gl_FrontFacing ? N : -N), 0); // interpolated normals

   // Lambert BRDF
   material.rgb = gl_FrontMaterial.diffuse.rgb //gl_Color.rgb
       / 3.14159265359     // PI
      * pow(texture2D(diffuseTexture, gl_TexCoord[0].xy).rgb, vec3(2.2));

   material.a = gl_FrontMaterial.diffuse.a;//0.0;
}
uniform vec3 lightPosition;
uniform vec3 eyePosition;

varying vec3 viewDirectionOut;
varying vec3 lightDirectionOut;
varying vec3 normalOut;
   
void main( void )
{
   gl_Position = ftransform();
   
   vec4 objectPosition = gl_ModelViewMatrix * gl_Vertex;
   
   viewDirectionOut  = eyePosition - objectPosition.xyz;
   lightDirectionOut = lightPosition - objectPosition.xyz;
   normalOut         = gl_NormalMatrix * gl_Normal;   
   
}
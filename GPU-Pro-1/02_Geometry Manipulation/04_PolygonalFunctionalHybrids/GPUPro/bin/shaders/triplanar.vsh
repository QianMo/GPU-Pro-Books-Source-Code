uniform vec4 lightPosition;
uniform vec4 eyePosition;

varying vec3 positionNoTransformOut;
varying vec3 viewDirectionOut;
varying vec3 lightDirectionOut;
varying vec3 normalOut;
varying vec3 normalNoTransformOut;
   
void main( void )
{
   gl_Position = ftransform();       
   
   vec4 objectPosition  =  gl_ModelViewMatrix * gl_Vertex;
   
   positionNoTransformOut  =  vec3(gl_Vertex);
   
   viewDirectionOut     =  (eyePosition - objectPosition).xyz;
   lightDirectionOut    =  (lightPosition - objectPosition).xyz;
   normalOut            =  vec3(gl_NormalMatrix * gl_Normal);
   normalNoTransformOut =  gl_Normal.xyz;
   
}

uniform vec3 lightPosition;
uniform vec3 eyePosition;
uniform float tick;
uniform float scale;

varying vec3 positionOut1;
varying vec3 positionOut2;
varying vec3 viewOut;
varying vec3 lightOut;
varying vec3 halfOut;
varying vec3 normalOut;
   
void main( void )
{
   gl_Position = ftransform();
   
   vec4 objectPosition = gl_ModelViewMatrix * gl_Vertex;
   
   positionOut1       = gl_Vertex.xyz * scale + tick * 0.025;

   float tickScaled  = tick * 0.005;
   positionOut2      = gl_Vertex.xyz * scale * 0.75 - vec3(tickScaled, tickScaled * 1.5, tickScaled * 2.2);

   viewOut  = eyePosition - objectPosition.xyz;
   lightOut = lightPosition - objectPosition.xyz;
   normalOut = gl_NormalMatrix * gl_Normal;     
   halfOut = normalize ( lightOut + viewOut );
   
}
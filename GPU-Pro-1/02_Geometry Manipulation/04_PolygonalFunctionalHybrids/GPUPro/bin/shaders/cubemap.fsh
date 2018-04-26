
uniform float offset;
uniform float scale;

uniform sampler2D planarTexture;
uniform samplerCube cubeMap;

varying vec3 viewDirectionOut;
varying vec3 lightDirectionOut;
varying vec3 normalOut;
	
void main( void )
{
   vec3  lightDirection = normalize( lightDirectionOut );
   vec3  normal         = normalize( normalOut );
   float NDotL          = dot( normal, lightDirection );   
  
   vec3 reflection   =  reflect(-viewDirectionOut, normal);  
   vec4 texColor     =  textureCube(cubeMap, reflection);
   
   gl_FragColor = texColor * NDotL;
       
}
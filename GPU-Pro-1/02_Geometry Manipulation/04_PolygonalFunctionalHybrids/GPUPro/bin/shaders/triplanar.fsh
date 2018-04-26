
uniform float offset;
uniform float scale;

uniform sampler2D planarTexture;

uniform float specularPower;

varying vec3 positionNoTransformOut;
varying vec3 viewDirectionOut;
varying vec3 lightDirectionOut;
varying vec3 normalOut;
varying vec3 normalNoTransformOut;
	
void main( void )
{
   vec3  lightDirection =  normalize( lightDirectionOut );
   vec3  normal         =  normalize( normalOut );
   float NDotL          =  dot( normal, lightDirection ); 
   
   vec3  reflection     = normalize( vec3( 2.0 * normal * NDotL ) - lightDirection ); 
   vec3  viewDirection  = normalize( viewDirectionOut );
   float RDotV          = max( 0.0, dot( reflection, viewDirection ) );  
  
   vec4  specular =  vec4(pow( RDotV, specularPower ));  
  
   vec3 normSqr   =  normalize(normalNoTransformOut * normalNoTransformOut);
   
   vec3 position = positionNoTransformOut * scale;   
   
   float a1 = (normSqr.x);
   float a2 = (normSqr.y);   
   float a3 = (normSqr.z);
   
   vec4 texColor1 = texture2D(planarTexture, vec2(position.z+offset, position.y+offset));
   vec4 texColor2 = texture2D(planarTexture, vec2(position.x+offset, position.z+offset));
   vec4 texColor3 = texture2D(planarTexture, vec2(position.x+offset, position.y+offset));   
   
   vec4 texColor = texColor1 * a1 + texColor2 * a2 + texColor3 * a3 ;

   gl_FragColor = texColor * NDotL + specular;
       
}
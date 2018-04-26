
uniform sampler3D volumeTexture;

varying vec3 positionOut1;
varying vec3 positionOut2;
varying vec3 viewOut;
varying vec3 lightOut;
varying vec3 halfOut;
varying vec3 normalOut;
	
void main( void )
{  
   const float	specPower = 100.0;

   vec3	n     =  normalize ( normalOut );
   vec3	l     =  normalize ( lightOut );
   vec3	h     =  normalize ( halfOut );
   float spec =  0.4 * pow ( max ( dot ( n, h ), 0.0 ), specPower );

   float noise1 = texture3D(volumeTexture, positionOut1).x;
   float noise2 = texture3D(volumeTexture, positionOut2).x;

   vec3  lightDirection = normalize( lightOut );
   vec3  normal         = normalize( normalOut );
   float NDotL          = dot( normal, lightDirection );

   float turbulence1 = 0.3 + 2.0 * abs(noise1 - 0.5);
   float turbulence2 = 0.5 * (0.5 + 4.0 * abs(noise2 - 0.5));

   vec4 color1 = vec4(0.95, 0.94, 0.96, 1.0);
   vec4 color2 = vec4(0.46, 0.57, 0.0, 1.0);
 
   
   gl_FragColor = vec4((turbulence1 * color1 + turbulence2 * color2) * NDotL + spec);
       
}
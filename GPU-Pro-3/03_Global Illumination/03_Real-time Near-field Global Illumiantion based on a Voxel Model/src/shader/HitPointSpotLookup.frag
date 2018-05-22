///////////////////////////////////////////////////////////////////////
// Real-time Near-Field Global Illumination based on a Voxel Model
// GPU Pro 3
///////////////////////////////////////////////////////////////////////
// Authors : Sinje Thiedemann, Niklas Henrich, 
//           Thorsten Grosch, Stefan Mueller
// Created:  August 2011
///////////////////////////////////////////////////////////////////////

#define INV_PI 0.31831
#define DIV_PI2 0.1591549431  // = 1.0 / (2*PI)

uniform sampler2D hitBuffer; // hit points in world space; get luminance from here...
uniform sampler2D hitRayOriginBuffer; // ...to here (starting points = g-buffer world positions)

uniform sampler2D positionSpotMap; 
uniform sampler2D normalSpotMap;
uniform sampler2D colorSpotMap; 

uniform mat4 mapLookupMatrix; // transforms world position to spot light space 

uniform float sampleContrib; // contrast * invRayNum 
uniform float distanceThresholdScale;
uniform float voxelDiagonal;
uniform float pixelSide_zNear; // pixelSide / zNear

uniform vec3 spotDirection; // in world space (but reversed, pointing to spot light)

uniform bool normalCheck;

varying out vec3 result;

void main()
{
   // initialize resulting luminance with black
   result = vec3(0);

   // get intersection point with voxel scene
   // and transform it into spot light space
   vec3 hitPos = texture2D(hitBuffer, gl_TexCoord[0].st).xyz;
   vec4 projCoord = mapLookupMatrix * vec4(hitPos, 1.0); 

   // fetch RSM values
   vec4 position  = texture2DProj(positionSpotMap, projCoord);
   vec4 color = texture2DProj(colorSpotMap, projCoord); // luminance
   vec3 normal = texture2DProj(normalSpotMap, projCoord).rgb;

   // compute ray from hit position to g-buffer position
   vec3 ray = normalize(texture2D(hitRayOriginBuffer, gl_TexCoord[0].st).xyz - hitPos);
   float lightZ = position.w;
   float pixelSide = pixelSide_zNear * lightZ; 

   // only front faces are lit and senders of indirect light
   bool normalCondition = normalCheck ? dot(normal, ray) >= 0.0 : true;
   float cosAlpha = normalCheck ? max(0.001, dot(normal, spotDirection)) : 1.0;

   // check if the hit point is valid
   if(hitPos.z < 100.0
      && (projCoord.w > 0.0)
      && distance(hitPos, position.xyz) <  min(4.0 * voxelDiagonal, max(voxelDiagonal, pixelSide/cosAlpha*distanceThresholdScale))
      && normalCondition)
   {
      // output luminance
      // sampleContrib = (user-def. contrast / numberOfSamples)
      result = color.rgb * sampleContrib;
   }
}

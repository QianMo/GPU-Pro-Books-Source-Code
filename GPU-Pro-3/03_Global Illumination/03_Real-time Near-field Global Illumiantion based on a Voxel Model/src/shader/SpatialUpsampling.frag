uniform vec2 filterDirection;
// X Direction: vec2(inverse_inputTex_size.x, 0.0);
// Y Direction: vec2(0.0, inverse_inputTex_size.y);

uniform sampler2D positionBuffer; // high res
uniform sampler2D normalBuffer;   // high res
uniform sampler2D inputTex; // low res
uniform sampler2D materialBufferLowRes; // low res (same as original input)
uniform sampler2D materialBufferHighRes; // high res (from g buffer)

uniform float lowResDiag;

uniform int filterRadius;
uniform float distanceLimit_sqr; // "positionThreshold"
uniform float normalLimit_sqr;   // "normalThreshold"
uniform float materialLimit;

varying out vec3 result;

float falloff(in float x, in float sigma_sqr)
{
   float g = max(0.01, 1.0 - x*x / sigma_sqr);
   return g*g*g;
}

float kImageSpace(in float pixelDistance)
{
   float g = max(0.01, 1.0 - pixelDistance / lowResDiag);
   return g*g*g;
}

void main()
{
   // x = center pixel (this pixel)
   // y = pixel in the domain around center pixel x

   // domain: m x m with m = 2 * filterRadius + 1

   result = vec3(0.3);

   vec2 texCoord = gl_TexCoord[0].xy;

   vec4 thisWorldPos = texture2D(positionBuffer, texCoord).rgba;
   vec3 thisNormal   = texture2D(normalBuffer, texCoord).rgb;
   vec3 thisMaterial = texture2D(materialBufferHighRes, texCoord).rgb; 

   if (thisWorldPos.z < 100.0)
   {
      vec3 weightedResult = vec3(0.0);
      float weightSum = 0.0;

      for(int j = -filterRadius; j <= filterRadius; j++) {

         vec2 sampleTexCoord = texCoord + float(j) * filterDirection;
         //vec3 sample         = texture2D(inputTex, sampleTexCoord).rgb;
         //vec4 samplePosition = texture2D(positionBuffer, sampleTexCoord).rgba;
         //vec3 sampleNormal   = texture2D(normalBuffer, sampleTexCoord).rgb;

         float dist = distance(thisWorldPos.xyz, texture2D(positionBuffer, sampleTexCoord).rgb);

         float weight = falloff(max(0.0, 1.0 - dot(thisNormal, texture2D(normalBuffer, sampleTexCoord).rgb)), normalLimit_sqr)
            //* falloff(distance(thisWorldPos, samplePosition), distanceLimit_sqr)
            * falloff(dist, distanceLimit_sqr)
            * kImageSpace(abs(float(j)))
            * ( max(0.01, 1.0 - distance(thisMaterial,  texture2D(materialBufferLowRes, sampleTexCoord).rgb)/materialLimit) );

         weightSum += weight;
         weightedResult += weight * texture2D(inputTex, sampleTexCoord).rgb;

      }

      if(weightSum > 0.0)
         result = weightedResult / weightSum;         // average of valid pixels		

   }

}
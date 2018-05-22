#version 120
#extension GL_EXT_gpu_shader4 : require 

uniform sampler2D positionBuffer; // world space
uniform sampler2DShadow shadowMap; // slot 1
uniform sampler2D randTex; // .rg = cos(angle), sin(angle)

uniform mat4 mapLookupMatrix;

uniform vec2 pixelOffset; // ( 1.0 / shadowMapWidth, 1.0 / shadowMapHeight )
uniform int randTexSize;

uniform bool lowQuality;
uniform float shadowMapEps;

varying out vec3 shadowValue;

vec2 rand = texelFetch2D(randTex, ivec2(mod(ivec2(gl_FragCoord.st), ivec2(randTexSize))), 0).rg;

float shadowLookup(in vec2 offset, in vec4 shadowCoord)
{
	// Offset is multiplied by shadowCoord.w because shadow2DProj does a division by w. 
   vec2 fetchOffset = vec2(dot(offset, vec2(1, -1)*rand), dot(offset, rand.yx)) * shadowCoord.w;

	return shadow2DProj(shadowMap, shadowCoord + vec4(fetchOffset, 0, shadowMapEps) ).r;
}

void main()
{
   vec3 P = texture2D(positionBuffer, gl_TexCoord[0].st).xyz;

   if(P.z < 100.0)
   {
      vec4 shadowCoord = mapLookupMatrix * vec4(P, 1.0); 

      float shadow = 1.0;
      if(shadowCoord.w > 0.0)
      {
         shadow = 0.0;
         float x, y;

         float border = float(!lowQuality) * 1.5;
         int num = 0;
         for (y = -border; y <= border; y += 1.0)
            for (x = -border; x <= border; x += 1.0)
            {
               shadow += shadowLookup(pixelOffset*vec2(x, y), shadowCoord);
               num++;
            }
         shadow /= float(num);
      }
      shadowValue = vec3(shadow);
   }
   else
   {
      shadowValue = vec3(1.0);
   }

}
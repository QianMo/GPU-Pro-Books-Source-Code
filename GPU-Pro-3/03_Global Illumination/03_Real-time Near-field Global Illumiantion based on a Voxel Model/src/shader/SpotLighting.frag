#define PI 3.14159265359
#define PI2 6.2831853

uniform sampler2D positionBuffer; // world space
uniform sampler2D normalBuffer;   // world space
uniform sampler2D materialBuffer; 

uniform sampler2D envMap;

uniform vec3 up;
uniform vec3 right;
uniform vec3 leftBottomNear;

uniform vec3 lightPos;
uniform vec3 eyePos;
uniform vec3 I; // Luminosity [candela]
uniform vec3 spotDirection; // normalized
uniform float spotCosCutoff; // cosinus of cutoff angle
uniform float spotExponent;
uniform float constantAttenuation;
uniform float quadraticAttenuation;
uniform float spotInnerCosCutoff;

uniform float envMapRotationAngle;

varying out vec3 luminance;

void main()
{
   luminance = vec3(0);
   vec3 P = texture2D(positionBuffer, gl_TexCoord[0].xy).xyz;
   vec4 material = texture2D(materialBuffer, gl_TexCoord[0].xy).rgba;
   if(P.z < 100.0)
   {
      vec3 N = texture2D(normalBuffer, gl_TexCoord[0].xy).xyz;
      // material.rgb = rho_diffuse / PI

      // vector from this position to the lightsource
      vec3 lightVec = lightPos - P;
      float lightDistance = length(lightVec);
      lightVec /= lightDistance; // normalize

      float att = 0.0; // attenuation
      float spotEffect = dot(spotDirection, -lightVec);

      // this point lies inside the illumination cone by the spotlight
      if (spotEffect >= spotCosCutoff)
      {
         float falloff = clamp((spotEffect-spotInnerCosCutoff) / 
            (spotInnerCosCutoff - spotCosCutoff), 0.0, 1.0);
         //float falloff = 1.0;

         att = falloff * pow(spotEffect, spotExponent) / (constantAttenuation +
            quadraticAttenuation * lightDistance * lightDistance);

         // diffuse
         float cosalpha = max(0.0, dot(N, lightVec));

         luminance = I * att * cosalpha * material.rgb;
      }

   }
   else
   {
      vec3 eyeRay = leftBottomNear + gl_TexCoord[0].x * right + gl_TexCoord[0].y * up - eyePos;
      eyeRay = normalize(eyeRay);

      float envTheta = acos(eyeRay.y);              
      float envPhi = atan(eyeRay.z, eyeRay.x);
      envPhi += envMapRotationAngle;
      if (envPhi < 0.0) envPhi += PI2;
      if (envPhi > PI2) envPhi -= PI2;

      // light from environment map
      luminance = texture2D(envMap, vec2( envPhi / (PI2), 1.0 - envTheta  / PI ) ).rgb;
   }

}
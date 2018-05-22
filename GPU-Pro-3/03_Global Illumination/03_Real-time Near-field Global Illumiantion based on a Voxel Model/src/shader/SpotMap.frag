varying vec3 P;
varying vec3 N;
varying vec3 P_world;
varying vec3 N_world;

uniform sampler2D diffuseTexture;  // slot 0

uniform vec3 I; // Luminosity (german Lichtstaerke [candela])

uniform float spotCosCutoff; // cosinus of (outer) cutoff angle
uniform float spotInnerCosCutoff; // cosinus of (inner) cutoff angle
uniform float spotExponent;
uniform float constantAttenuation;
uniform float quadraticAttenuation;

varying out vec4 luminance;
varying out vec4 position;
varying out vec4 normal;
varying out vec4 material; 

uniform mat4 inverseViewMatrix;

void main()
{
   position = vec4(P_world, abs(P.z));
   normal = vec4(normalize(N_world), 0);
   luminance = vec4(0);

   material.rgb = gl_FrontMaterial.diffuse.rgb * gl_FrontMaterial.diffuse.a // color * materialSenderScaleFactor
      / 3.14159265359
      * pow(texture2D(diffuseTexture, gl_TexCoord[0].st).rgb, vec3(2.2));
 //* texture2D(diffuseTexture, gl_TexCoord[0].st).rgb;
   material.a = gl_FrontMaterial.shininess;


   float att = 0.0;
   // lightPos: (0, 0, 0) = camera = light
   float lightDistance = length(P);
   vec3 lightVec = -P/lightDistance; // normalize

   //(0, 0, -1) : standard camera viewing direction
   vec3 spotDirection = vec3(0, 0, -1);

   float spotEffect = lightVec.z; // = dot(spotDirection, -lightVec);

   if (spotEffect >= spotCosCutoff)
   {
      //float falloff = 1.0;
      float falloff = clamp((spotEffect-spotInnerCosCutoff) / 
         (spotInnerCosCutoff - spotCosCutoff), 0.0, 1.0);

      att = falloff * pow(spotEffect, spotExponent) / (constantAttenuation +
         quadraticAttenuation * lightDistance * lightDistance);
      // else: this point lies outside the cone of illumination produced by the spotlight.


      luminance.rgb = I * max(0.0, dot(normalize(N), lightVec)) * att * material.rgb;


      material.rgb = material.rgb * pow(spotEffect, spotExponent);

   }
   else
   {
      luminance *= 0.0;
      material  *= 0.0;
   }

}
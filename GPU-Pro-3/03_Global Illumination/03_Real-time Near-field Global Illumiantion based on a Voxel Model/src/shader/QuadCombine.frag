#define PI 3.14159265359

uniform sampler2D directLightBuffer;
uniform sampler2D indirectLightBuffer;
uniform sampler2D materialBuffer;
uniform sampler2D positionBuffer;

uniform float scaleIL;
uniform float scaleLdir;

uniform bool addDirectLight;


varying out vec3 result;

void main()
{
   vec4 material = texture2D(materialBuffer, gl_TexCoord[0].st);
   vec3 L_dir = texture2D(directLightBuffer, gl_TexCoord[0].st).rgb;
   if(material.z < 100.0)
   {
      vec3 indir = texture2D(indirectLightBuffer, gl_TexCoord[0].st).rgb;

      // assume all pixels have diffuse material

      // L_indir = E_indir * brdf_diffuse
      result = scaleIL * indir * material.rgb;


      if(addDirectLight)
         result += scaleLdir * L_dir;
      //gl_FragColor.a = 1.0;

      result = max(vec3(0.00001), result);

   }
   else
   {
      result = L_dir;
   }

}
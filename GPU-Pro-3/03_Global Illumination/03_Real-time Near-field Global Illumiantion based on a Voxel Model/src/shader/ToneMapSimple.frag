uniform sampler2D Lrgb;

uniform float maxRadiance; // some magic user set value
uniform float c;    // 1.0 / log2(1.0 + maxRadiance)
uniform float gammaExponent; // = 1.0 / gamma

uniform bool logToneMap;

// see SSDO Demo
void linearTM(inout vec3 color)
{
   // simple gamma tone mapper 
   float greyRadiance = dot(vec3(0.30, 0.59, 0.11), color);
   float mappedRadiance = min(greyRadiance / maxRadiance, 1.0);
   color *= (mappedRadiance / greyRadiance); 
}

void logTM(inout vec3 color)
{
   float greyRadiance = dot(vec3(0.30, 0.59, 0.11), color);
   float mappedRadiance = log2(1.0 + greyRadiance) * c;  
   color *= (mappedRadiance / greyRadiance); 
}

void main()
{ 
   vec3 color = texture2D(Lrgb, gl_TexCoord[0].st).rgb;
   if(logToneMap)
      logTM(color);
   else
      linearTM(color);
   gl_FragColor = vec4(pow(color, vec3(gammaExponent)), 1.0);

}
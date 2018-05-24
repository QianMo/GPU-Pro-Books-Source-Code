layout(binding = COLOR_TEX_BP) uniform sampler2D colorMap;

out vec4 fragColor;

#define A  0.4            // shoulderStrength
#define B  0.3            // linearStrength
#define C  0.1            // linearAngle
#define D  0.2            // toeStrength
#define E  0.01           // toeNumerator
#define F  0.3            // toeDenominator
#define LINEAR_WHITE 11.2 

vec3 FilmicFunc(in vec3 x) 
{
  return ((x*(A*x+C*B)+D*E)/(x*(A*x+B)+D*F))-E/F;
}

vec3 TonemapFilmic(in vec3 color)
{
  vec3 numerator = FilmicFunc(color);    
  vec3 denominator = FilmicFunc(vec3(LINEAR_WHITE, LINEAR_WHITE, LINEAR_WHITE));
  return numerator/denominator;
}

void main()
{
  vec4 color = texelFetch(colorMap, ivec2(gl_FragCoord.xy), 0);

  // perform filmic tone-mapping with constant exposure
  const float exposure = 1.2;
  color.rgb *= exposure;
  color.rgb = TonemapFilmic(color.rgb);

  fragColor = color;
}


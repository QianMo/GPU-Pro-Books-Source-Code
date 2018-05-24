layout(binding = COLOR_TEX_BP) uniform sampler2D colorMap;
layout(binding = NORMAL_TEX_BP) uniform sampler2D normalMap;
layout(binding = SPECULAR_TEX_BP) uniform sampler2D specularMap;

in VS_Output
{
  vec2 texCoords;
  vec3 normal;
  vec3 tangent;
  vec3 bitangent;
} inputFS;

layout(location = 0, index = 0) out vec4 fragColor0;
layout(location = 1, index = 0) out vec4 fragColor1; 

#define ALPHA_THRESHOLD 0.3

void main() 
{
  vec4 albedo = texture(colorMap, inputFS.texCoords);   
  
#ifdef ALPHA_TEST
  if(albedo.a < ALPHA_THRESHOLD)
    discard;
#endif

  mat3 tangentMatrix;
  tangentMatrix[0] = normalize(inputFS.tangent);
  tangentMatrix[1] = normalize(inputFS.bitangent);
  tangentMatrix[2] = normalize(inputFS.normal);
  
  vec3 bump = texture(normalMap, inputFS.texCoords).xyz*2.0-1.0;
  bump = tangentMatrix*bump;
  bump = normalize(bump);
  bump = (bump*0.5)+0.5;
  float gloss = texture(specularMap, inputFS.texCoords).r;

  fragColor0 = vec4(albedo.rgb, gloss);

  fragColor1 = vec4(bump, 0.0);
}


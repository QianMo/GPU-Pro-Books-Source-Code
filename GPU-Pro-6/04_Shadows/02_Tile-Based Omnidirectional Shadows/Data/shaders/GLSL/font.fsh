layout(binding = COLOR_TEX_BP) uniform sampler2D colorMap;

in GS_Output
{
  vec2 texCoords;
  vec3 color;
} inputFS;

out vec4 fragColor;

void main() 
{
  vec4 base = texture(colorMap, inputFS.texCoords);
  base.rgb *= inputFS.color;
  fragColor = base;
}


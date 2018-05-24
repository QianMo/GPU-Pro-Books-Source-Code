layout(binding = COLOR_TEX_BP) uniform sampler2D shadowMap;

in GS_Output
{
  vec2 texCoords;
} inputFS;

out vec4 fragColor;

void main()
{
  float depth = texture(shadowMap, inputFS.texCoords).r;
  depth = pow(depth, 1000.0f);
  fragColor = vec4(depth, depth, depth, 1.0);
}


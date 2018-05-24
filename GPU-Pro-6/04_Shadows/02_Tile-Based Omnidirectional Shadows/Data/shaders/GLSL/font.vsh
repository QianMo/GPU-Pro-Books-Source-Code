layout(location = POSITION_ATTRIB) in vec3 inputPosition; 
layout(location = TEXCOORDS_ATTRIB) in vec2 inputTexCoords; 
layout(location = COLOR_ATTRIB) in vec3 inputColor; 

out VS_Output
{
  vec2 texCoords;
  vec3 color;
} outputVS;

void main()
{
  gl_Position = vec4(inputPosition, 1.0);
  outputVS.texCoords = inputTexCoords;
  outputVS.color = inputColor;
}


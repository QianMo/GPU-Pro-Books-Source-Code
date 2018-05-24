in VS_Output
{
  vec2 texCoords;
  vec3 color;
} inputGS[];

out GS_Output
{
  vec2 texCoords;
  vec3 color;
} outputGS;

layout(lines) in;
layout(triangle_strip, max_vertices = 4) out;
void main()
{
  // generate a quad from input line (2 vertices)
  // ->generate 1 triangle-strip

  const vec4 mins = gl_in[0].gl_Position;
  const vec4 maxes = gl_in[1].gl_Position;

  // left/ lower vertex
  gl_Position = vec4(mins.x, mins.y, mins.z, 1.0);
  outputGS.texCoords = vec2(inputGS[0].texCoords.x, inputGS[0].texCoords.y);
  outputGS.color = inputGS[0].color;
  EmitVertex();
  
  // right/ lower vertex
  gl_Position = vec4(maxes.x, mins.y, mins.z, 1.0);
  outputGS.texCoords = vec2(inputGS[1].texCoords.x, inputGS[0].texCoords.y);
  outputGS.color = inputGS[1].color;
  EmitVertex();
  
  // left/ upper vertex
  gl_Position = vec4(mins.x, maxes.y, mins.z, 1.0);
  outputGS.texCoords = vec2(inputGS[0].texCoords.x, inputGS[1].texCoords.y);
  outputGS.color = inputGS[0].color;
  EmitVertex();
  
  // right/ upper vertex
  gl_Position = vec4(maxes.x, maxes.y, mins.z, 1.0);
  outputGS.texCoords = vec2(inputGS[1].texCoords.x, inputGS[1].texCoords.y);
  outputGS.color = inputGS[1].color;
  EmitVertex();
  EndPrimitive();
}


out GS_Output
{
  vec2 texCoords;
} outputGS;

layout(lines) in;
layout(triangle_strip, max_vertices = 4) out;
void main()
{
  // generate a quad from input line (2 vertices)
  // ->generate 1 triangle-strip

  const vec3 mins = gl_in[0].gl_Position.xyz;
  const vec3 maxes = gl_in[1].gl_Position.xyz;

  // left/ lower vertex
  gl_Position = vec4(mins.x, mins.y, mins.z, 1.0);
  outputGS.texCoords = vec2(0.0, 0.0);
  EmitVertex();
  
  // right/ lower vertex
  gl_Position = vec4(maxes.x, mins.y, mins.z, 1.0);
  outputGS.texCoords = vec2(1.0, 0.0);
  EmitVertex();
  
  // left/ upper vertex
  gl_Position = vec4(mins.x, maxes.y, mins.z, 1.0);
  outputGS.texCoords = vec2(0.0, 1.0);
  EmitVertex();
  
  // right/ upper vertex
  gl_Position = vec4(maxes.x, maxes.y, mins.z, 1.0);
  outputGS.texCoords = vec2(1.0, 1.0);
  EmitVertex();
  EndPrimitive();
}


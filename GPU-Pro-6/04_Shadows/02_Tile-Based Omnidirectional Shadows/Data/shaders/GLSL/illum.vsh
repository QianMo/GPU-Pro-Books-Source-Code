const vec4 positionTexCoords[3] =
{
  vec4(-1.0, -1.0, 0.0, 0.0),
  vec4(3.0, -1.0, 2.0, 0.0),
  vec4(-1.0, 3.0, 0.0, 2.0) 
};

out VS_Output
{
  vec2 texCoords;
} outputVS;

void main()
{
  vec4 outputPositionTexCoords = positionTexCoords[gl_VertexID];
  gl_Position = vec4(outputPositionTexCoords.xy, 0.0, 1.0);
  outputVS.texCoords = outputPositionTexCoords.zw;
}


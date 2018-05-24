GLOBAL_CAMERA_UB(cameraUB);

const vec2 positions[2] = 
{
  vec2(-1.0, -1.0),
  vec2(1.0, 1.0)
};

void main()
{
  vec4 scalePosition = vec4(0.5/cameraUB.aspectRatio, 0.5, -0.65, -0.45);
  gl_Position = vec4(positions[gl_VertexID]*scalePosition.xy+scalePosition.zw, 0.0, 1.0);
}


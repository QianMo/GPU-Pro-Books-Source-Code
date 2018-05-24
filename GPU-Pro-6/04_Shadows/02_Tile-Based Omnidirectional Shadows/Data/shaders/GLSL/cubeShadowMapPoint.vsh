layout(location = POSITION_ATTRIB) in vec3 inputPosition; 

void main()
{
  gl_Position = vec4(inputPosition, 1.0);
}


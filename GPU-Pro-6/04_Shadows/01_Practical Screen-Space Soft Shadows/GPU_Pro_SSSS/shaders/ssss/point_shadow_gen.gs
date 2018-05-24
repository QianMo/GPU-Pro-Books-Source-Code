#version 430 core

//get me a triangle
layout(triangles) in; 
//3 vertices per triangle, 6 triangles overall
layout(triangle_strip, max_vertices=18) out;

//per face view projection matrices
uniform mat4 cube_viewproj[6];

void main()
{
  //redirect to 6 cubemap faces
  for(int layer = 0; layer < 6; ++layer)
  {
    gl_Layer = layer;

    for( int i = 0; i < 3; ++i )
    {
      gl_Position = cube_viewproj[layer] * gl_in[i].gl_Position;
      EmitVertex();
    }
    
    EndPrimitive();
  }
}

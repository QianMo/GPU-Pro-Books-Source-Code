#ifndef fixed_pos_h
#define fixed_pos_h

vec3 fixed_positions[] = 
{
  //fixed pos 0
  vec3( 0 ), //pos
  vec3( 0, 0, -1 ), //view dir
  vec3( 0, 1, 0 ), //up vector

  //fixed pos 1
  vec3( -20.6267, 6.79272, -14.4537 ),
  vec3( 0.158339, -0.434445, -0.886669 ),
  vec3( 0.0763741, 0.900697, -0.427679 ),

  //fixed pos 2
  vec3( 47.4563, 9.04135, -14.0328 ),
  vec3( 0.0397615, -0.496215, -0.867286 ),
  vec3( 0.0227259, 0.868197, -0.495695 ),

  //fixed pos 3
  vec3( 4.14832, 8.57227, -16.4794 ),
  vec3( 0.46578, -0.554358, -0.689738 ),
  vec3( 0.310243, 0.832277, -0.459415 ),

  //POSITIONS THAT WERE USED IN THE BOOK

  //fixed pos 4
  vec3( 42.2075, 87.5613, 9.20957 ),
  vec3( 0.442782, 0.104527, 0.890514 ),
  vec3( -0.0465379, 0.994522, -0.093596 ),

  //fixed pos 5
  vec3( -115.731, 57.6722, 0.0908885 ),
  vec3( 0.999613, -0.0130896, -0.024539 ),
  vec3( 0.0130861, 0.999915, -0.00032124 ),

  //fixed pos 6
  vec3( -26.1523, 4.85538, -9.49967 ),
  vec3( 0.592158, 0.108867, -0.798433 ),
  vec3( -0.0648519, 0.994056, 0.0874427 ),

  //fixed pos 7
  vec3( 50.8205, 13.7232, -4.76363 ),
  vec3( 0.00484472, -0.160742, -0.986983 ),
  vec3( 0.000789, 0.986996, -0.16074 ),

  //fixed pos 8
  vec3( -117.438, 13.407, -2.57179 ),
  vec3( 0.993549, -0.108868, -0.0317124 ),
  vec3( 0.108812, 0.994056, -0.00347301 )
};

  /**
  //Applies to Blender Cycles

  //blender position works like
  //x == x, y == -z, z = y
  vec3 blender_position;
  blender_position.x = cam.pos.x;
  blender_position.y = -cam.pos.z;
  blender_position.z = cam.pos.y;

  //conversion to blender rotations
  //view dir
  vec3 blender_rotation_degrees;
  //may need further offset by whole degrees...
  blender_rotation_degrees.x = 90-degrees(get_angle( vec3( 0, 0, -1 ), cam.view_dir ));
  //would be right vector (1, 0, 0)
  blender_rotation_degrees.y = 0;
  //up vector
  //cross up, view dir
  vec3 rightvec = cross( cam.up_vector, cam.view_dir );
  //may need 360-z
  blender_rotation_degrees.z = degrees(get_angle( vec3( -1, 0, 0 ), rightvec ));

  cout << "Blender position: " << blender_position << endl;
  cout << "Blender rotation (degrees): " << blender_rotation_degrees << endl;

  //to get fov, go here:
  //http://www.bdimitrov.de/kmp/technology/fov.html
  //insert fov in degrees into the diagonal fov input box
  //get focal length in mm
  //insert that into the focal length box above it
  //get the fov on the right
  /**/

#endif
#ifndef common_h
#define common_h

vec3 right_vec[6] = 
{
  vec3( 0, 0,  1),
  vec3( 0, 0, -1),
  vec3( 1, 0,  0),
  vec3( 1, 0,  0),
  vec3(-1, 0,  0),
  vec3( 1, 0,  0)
};

vec3 up_vec[6] = 
{
  vec3( 0, 1,  0),
  vec3( 0, 1,  0),
  vec3( 0, 0,  1),
  vec3( 0, 0, -1),
  vec3( 0, 1,  0),
  vec3( 0, 1,  0)
};

float find_blocker_point( int layer, float bias, float proj_a, float proj_b, vec3 right, vec3 up, samplerCubeArray tex, vec3 texcoord, vec4 shad_coord, float search_width, int num_samples )
{
	float step_size = 2 * search_width / float(num_samples);
	
	texcoord.xyz -= search_width * (right+up);
	
	float blocker_sum = 0;
	float receiver = shad_coord.z;
	float blocker_count = 0;
	float found_blocker = 0;
	
	for( int c = 0; c < num_samples; ++c )
	{
		for( int d = 0; d < num_samples; ++d )
		{
      float depth = texture( tex, vec4( texcoord.xyz + c * step_size * up + d * step_size * right, layer ) ).x; //[0...1]
      float lightndcdepth = depth * 2 - 1; //[-1...1]
      float lightviewdepth = -proj_b / (lightndcdepth + proj_a);
      float lightprojdepth = lightndcdepth * -lightviewdepth;
      
      if( depth-bias < receiver )
      {
        blocker_sum += lightprojdepth;
        ++blocker_count;
        found_blocker = 1;
      }
    }
	}
	
	float result = 0;
	
	if( found_blocker == 0 )
	{
		result = 999;
	}
	else
	{
		result = blocker_sum / blocker_count;
	}
	
	return result;
}

bool check_sanity( float x )
{
  return isnan(x) || isinf(x);
}

bool bounds_check( float x )
{
  return x < 1.0 && x > 0.0;
}

float recip( float x )
{
  return 1.0 / x;
}

float sqr( float x )
{
	return x * x;
}

//decode linear depth into view space position
vec3 decode_linear_depth( float linear_depth, vec2 position, float far ) 
{
  return vec3( position, far ) * linear_depth;
}

//input: float value in range [0...1]
uint float_to_r8( float val )
{
  const uint bits = 8;
  uint precision_scaler = uint(pow( uint(2), bits )) - uint(1);
  return uint(floor( precision_scaler * val ));
}

uint rgba8_to_uint( vec4 val )
{
  uint res  = float_to_r8(val.x) << 24;
       res |= float_to_r8(val.y) << 16;
       res |= float_to_r8(val.z) << 8;
       res |= float_to_r8(val.w) << 0;
  return res;
}

vec4 uint_to_rgba8( uint val )
{
  uint tmp = val;
  uint r = (tmp & uint(0xff000000)) >> uint(24);
  uint g = (tmp & uint(0x00ff0000)) >> uint(16);
  uint b = (tmp & uint(0x0000ff00)) >> uint(8);
  uint a = (tmp & uint(0x000000ff)) >> uint(0);
  return vec4( r / 255.0, g / 255.0, b / 255.0, a / 255.0 );
}

float estimate_penumbra( float receiver, float blocker )
{
	return (receiver - blocker) / blocker;
}

float avg( vec4 val )
{
  return (val.x + val.y + val.z + val.w) * 0.25;
}

#endif
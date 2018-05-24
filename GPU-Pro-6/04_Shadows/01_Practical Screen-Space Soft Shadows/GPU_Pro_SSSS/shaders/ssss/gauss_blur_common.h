#ifndef gauss_blur_common_h
#define gauss_blur_common_h

/**
const int num_steps = 5-1;

const float weights[5] =
{
  0.294118,
  0.117647,
  0.235294,
  0.117647,
  0.235294
};

const float offsets[4] = 
{
  2.0,
  1.0,
  -2.0,
  -1.0
};
/**/

/**/
const int num_steps = 11-1;

const float weights[11] =
{
  0.209857,
  0.00556439,
  0.0222576,
  0.0612083,
  0.122417,
  0.183625,
  0.00556439,
  0.0222576,
  0.0612083,
  0.122417,
  0.183625
};

const float offsets[10] = 
{
  5.0,
  4.0,
  3.0,
  2.0,
  1.0,
  -5.0,
  -4.0, 
  -3.0,
  -2.0,
  -1.0
};
/**/

/**
const int num_steps = 23-1;

const float weights[23] =
{
  0.154981,
  4.84288e-006,
  3.87431e-005,
  0.000222773,
  0.000980199,
  0.0034307,
  0.00980199,
  0.0232797,
  0.0465595,
  0.0791511,
  0.115129,
  0.143911,
  4.84288e-006,
  3.87431e-005,
  0.000222773,
  0.000980199,
  0.0034307,
  0.00980199,
  0.0232797,
  0.0465595,
  0.0791511,
  0.115129,
  0.143911
};

const float offsets[22] = 
{
  11.0,
  10.0,
  9.0,
  8.0,
  7.0,
  6.0,
  5.0,
  4.0,
  3.0,
  2.0,
  1.0,
  -11.0,
  -10.0,
  -9.0,
  -8.0,
  -7.0,
  -6.0,
  -5.0,
  -4.0,
  -3.0,
  -2.0,
  -1.0
};
/**/

float blur( sampler2D penumbratex, float initial_shadow, vec2 step_size, float depth, float penumbra, const int layer )
{
  if(penumbra > 0.0)
	{		
		float final_color  = initial_shadow * weights[0];
		
		float sum_weights_ok = weights[0];
		
		for( int c = 0; c < num_steps; ++c )
		{      
      vec2 sample_loc = tex_coord + offsets[c] * step_size * penumbra;
			float adepth = texture( depth_tex, sample_loc ).x;
			
			if( abs( depth - adepth ) < err_depth )
			{
				sum_weights_ok += weights[c + 1];

#ifdef GAUSS_SUPERSAMPLE
  #ifdef EXPONENTIAL_SHADOWS
        switch( layer ) //stupid "constant integral check..."
        {
        case 0:
          final_color += unpack_shadow( textureGather( penumbratex, sample_loc, 0 ), layer ) * weights[c + 1];
          break;
        case 1:
          final_color += unpack_shadow( textureGather( penumbratex, sample_loc, 1 ), layer ) * weights[c + 1];
          break;
        case 2:
          final_color += unpack_shadow( textureGather( penumbratex, sample_loc, 2 ), layer ) * weights[c + 1];
          break;
        case 3:
          final_color += unpack_shadow( textureGather( penumbratex, sample_loc, 3 ), layer ) * weights[c + 1];
          break;
        }
  #else
        final_color += unpack_shadow( textureGather( penumbratex, sample_loc, 0 ), layer ) * weights[c + 1];
  #endif
#endif

#ifdef GAUSS2_SUPERSAMPLE
        switch( layer ) //stupid "constant integral check..."
        {
        case 0:
          final_color += avg( textureGather( penumbratex, sample_loc, 0 ) ) * weights[c + 1];
          break;
        case 1:
          final_color += avg( textureGather( penumbratex, sample_loc, 1 ) ) * weights[c + 1];
          break;
        case 2:
          final_color += avg( textureGather( penumbratex, sample_loc, 2 ) ) * weights[c + 1];
          break;
        case 3:
          final_color += avg( textureGather( penumbratex, sample_loc, 3 ) ) * weights[c + 1];
          break;
        }
#endif

#if !defined GAUSS_SUPERSAMPLE && !defined GAUSS2_SUPERSAMPLE
        final_color += unpack_shadow( texture( penumbratex, sample_loc ), layer ) * weights[c + 1];
#endif
			}
		}
		
		final_color  /= sum_weights_ok;
		
		return final_color;
	}
	else
	{
		return initial_shadow;
	}
}

vec4 blur_translucency( usampler2D penumbratex, vec4 initial_shadow, vec2 step_size, float depth, float penumbra, const int layer )
{
  if(penumbra > 0.0)
	{		
		vec4 final_color  = initial_shadow * weights[0];
		
		float sum_weights_ok = weights[0];
		
		for( int c = 0; c < num_steps; ++c )
		{      
      vec2 sample_loc = tex_coord + offsets[c] * step_size * penumbra;
			float adepth = texture( depth_tex, sample_loc ).x;
			
			if( abs( depth - adepth ) < err_depth )
			{
				sum_weights_ok += weights[c + 1];

#if !defined GAUSS_SUPERSAMPLE && !defined GAUSS2_SUPERSAMPLE
        final_color += uint_to_rgba8( uint( texture( penumbratex, sample_loc ).x ) ) * weights[c + 1];
#endif
			}
		}
		
		final_color  /= sum_weights_ok;
		
		return final_color;
	}
	else
	{
		return initial_shadow;
	}
}

vec2 get_step_size( vec2 direction, vec3 normal, float depth, float threshold )
{
  return direction 
         //* light_size * light_size //included in the penumbra
         * sqrt( max( dot( vec3( 0, 0, 1 ), normal ), threshold ) )
         * (1 / (depth/* * depth * 100*/));
}

#endif
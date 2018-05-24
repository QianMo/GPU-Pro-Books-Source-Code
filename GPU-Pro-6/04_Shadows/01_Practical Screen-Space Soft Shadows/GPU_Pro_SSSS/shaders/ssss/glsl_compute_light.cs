//#version 430 core

//#define PCF
//#define PCSS
//#define SSSS
//#define HARD_SHADOW
//#define EXPONENTIAL_SHADOWS

layout(local_size_x = 16, local_size_y = 16) in; //local workgroup size
	
layout(binding=0) uniform sampler2D normals;
layout(binding=1) uniform sampler2D depth;
layout(binding=2) uniform sampler2DArray spot_shadow_tex;
layout(binding=3) uniform samplerCubeArray point_shadow_tex;
layout(binding=4) uniform sampler2D layered_shadow_tex;
//layout(binding=5) uniform usampler2D layered_translucency_tex;
layout(location=0) writeonly uniform image2D result;
//layout(location=1) writeonly uniform image1D local_lights_result;

uniform vec2 nearfar;
uniform int num_lights;
uniform vec4 far_plane0;
uniform vec2 far_plane1;
uniform mat4 proj_mat;
uniform mat4 inv_view;
uniform mat4 inv_mv;

const float pi = 3.14159265;

float proj_a, proj_b;

shared float local_far, local_near;
shared int local_lights_num;
shared vec4 local_ll, local_ur;
shared int local_lights[1024];
shared int local_num_of_lights;
shared uint local_max_depth;
shared uint local_min_depth;
shared uint local_depth_mask;

#include "common.h"

uniform vec4 diffuse_color_data[16];
uniform vec4 specular_color_data[16];
uniform float light_size_data[16];
uniform vec4 vs_position_data[16];
uniform vec4 ms_position_data[16];
uniform float attenuation_end_data[16];
uniform float attenuation_cutoff_data[16];
uniform float radius_data[16];
uniform float spot_exponent_data[16];
uniform int attenuation_type_data[16];
uniform int lighting_type_data[16];
uniform int layer_data[16];
uniform vec4 spot_direction_data[16];
uniform float spot_cutoff_data[16];
uniform mat4 spot_shadow_mat_data[16];
uniform mat4 point_shadow_mat_data[16*6];
uniform uint light_indices[16];

vec3 tonemap_func(vec3 x, float a, float b, float c, float d, float e, float f)
{
  return ( ( x * ( a * x + c * b ) + d * e ) / ( x * ( a * x + b ) + d * f ) ) - e / f;
}
  
vec3 tonemap(vec3 col)
{
  //vec3 x = max( vec3(0), col - vec3(0.004));
  //return ( x * (6.2 * x + 0.5) ) / ( x * ( 6.2 * x + 1.7 ) + 0.06 );
  
  float a = 0.22; //Shoulder Strength
  float b = 0.30; //Linear Strength
  float c = 0.10; //Linear Angle
  float d = 0.20; //Toe Strength
  float e = 0.01; //Toe Numerator
  float f = 0.30; //Toe Denominator
  float linear_white = 11.2; //Linear White Point Value (11.2)
  //Note: E/F = Toe Angle
  
  return tonemap_func( col, a, b, c, d, e, f ) / tonemap_func( vec3(linear_white), a, b, c, d, e, f );
}

vec3 gamma_correct( vec3 col )
{
  return pow( col, vec3( 1 / 2.2 ) );
}

float roughness_to_spec_power(float m) 
{
  return 2.0 / (m * m) - 2.0;
}

float spec_power_to_roughness(float s) 
{
  return sqrt(2.0 / (s + 2.0));
}

float toksvig_aa(vec3 bump, float roughness)
{	
	//this is the alu based version
	float s = roughness_to_spec_power( roughness );
	float len = length( bump );
	float gloss = max(len / mix(s, 1.0, len), 0.01);
	
	return spec_power_to_roughness(gloss * s);
}

float fresnel_schlick( float v_dot_h, float f0 )
{
  float base = 1.0 - v_dot_h;
  float exponential = pow( base, 5.0 );
  return exponential + f0 * ( 1.0 - exponential );
}

float distribution_ggx( float n_dot_h, float alpha )
{  
  float cos_sqr = sqr(n_dot_h);
  float alpha_sqr = sqr(alpha);
  
  return alpha_sqr / ( pi * sqr( ( alpha_sqr - 1 ) * cos_sqr + 1 ) );
}

float geometric_torrance_sparrow( float n_dot_h, float n_dot_v, float v_dot_h, float n_dot_l )
{
	return min( 1.0, min( 2.0 * n_dot_h * n_dot_v / v_dot_h, 2.0 * n_dot_h * n_dot_l / v_dot_h ) );
}

float geometric_schlick_smith( float n_dot_v, float roughness )
{
  return n_dot_v / ( n_dot_v * (1 - roughness) + roughness );
}

float diffuse_lambert()
{
  return 1.0 / pi;
}

float diffuse_oren_nayar( float roughness, float n_dot_v, float n_dot_l, float v_dot_h )
{
  float v_dot_l = 2 * v_dot_h - 1;
  float m = sqr( roughness );
  float m2 = sqr( m );
  float c1 = 1 - 0.5 * m2 / ( m2 + 0.33 );
  float cos_ri = v_dot_l - n_dot_v * n_dot_l;
  float c2 = 0.45 * m2 / ( m2 + 0.09 ) * cos_ri;
  
  if( cos_ri >= 0 )
    c2 *= min( 1, n_dot_l / n_dot_v );
  else
    c2 *= n_dot_l;
  
  return diffuse_lambert() * ( n_dot_l * c1 + c2 );
}

vec3 brdf( int index, vec3 raw_albedo, vec3 raw_normal, vec3 light_dir, vec3 view_dir, float intensity, float attenuation, vec3 diffuse_color, vec3 specular_color )
{
  vec3 result = vec3( 0 );

  vec3 light_diffuse_color = diffuse_color.xyz;
  vec3 light_specular_color = specular_color.xyz;
  
  vec3 half_vector = normalize( light_dir + view_dir ); // * 0.5;

  float n_dot_l = clamp( dot( raw_normal, light_dir ), 0.0, 1.0 );
  float n_dot_h = clamp( dot( raw_normal, half_vector ), 0.0, 1.0 );
  float v_dot_h = clamp( dot( view_dir, half_vector ), 0.0, 1.0 );
  float n_dot_v = clamp( dot( raw_normal, view_dir ), 0.0, 1.0 );
  float l_dot_h = clamp( dot( light_dir, half_vector ), 0.0, 1.0 );

  float roughness = max( intensity * 0.3, 0.001 );
  //float roughness = intensity * 0.5;
  
  //float diffuse = diffuse_lambert() * n_dot_l;
  float diffuse = diffuse_oren_nayar( roughness, n_dot_v, n_dot_l, v_dot_h );

  result = diffuse * raw_albedo * light_diffuse_color.xyz; 
  
  if( n_dot_l > 0.0 )
  {
    /**/
    //F term
    float F = fresnel_schlick( l_dot_h, 0.028 );
    
    //D term
    //float D = distribution_ggx( n_dot_h, roughness );
    //float D = distribution_ggx( n_dot_h, pow(1 - roughness*0.7, 6.0) ); //TODO: remapped roughness?
    float D = distribution_ggx( n_dot_h, roughness );

    //G term
    //float G = geometric_torrance_sparrow( n_dot_h, n_dot_v, v_dot_h, n_dot_l );
    //float G = geometric_schlick_smith( n_dot_v, pow(0.8 + 0.5 * roughness, 2) * 0.5 );
    float G = geometric_schlick_smith( n_dot_v, roughness );

    float denom = (n_dot_l * n_dot_v) * 4.0; //TODO: do we need pi here?
    float specular = (D * G) * F 
                      * recip( denom > 0.0 ? denom : 1.0 ); //avoid div by 0 
    /**/
    
    //cheap blinn-phong specular
    //float specular = pow( n_dot_h, roughness );
  
    //result = vec3(float(isnan(specular) || isinf(specular)));
    result += n_dot_l * specular * light_specular_color.xyz;
  }

  return result * attenuation;
}

void main()
{
	ivec2 global_id = ivec2( gl_GlobalInvocationID.xy );
	vec2 global_size = textureSize( normals, 0 ).xy;
	ivec2 local_id = ivec2( gl_LocalInvocationID.xy );
	ivec2 local_size = ivec2( gl_WorkGroupSize.xy );
	ivec2 group_id = ivec2( gl_WorkGroupID.xy );
  ivec2 group_size = ivec2(global_size) / local_size;
	uint workgroup_index = gl_LocalInvocationIndex;
	vec2 texel = global_id / global_size;
	vec2 pos_xy;
	
	vec4 raw_albedo = vec4(vec3(1), 0.1); //texture( albedo, texel );
	vec4 raw_normal = vec4( 0 );
	vec4 raw_depth = texture( depth, texel );
	vec4 raw_ssdo = vec4( 0 );
	
	vec4 out_color = vec4( 0 );
	
	float max_depth = 0;
	float min_depth = 1;
	
	if( workgroup_index == 0 )
	{
		local_ll = vec4( far_plane0.xyz, 1.0 );
		local_ur = vec4( far_plane0.w, far_plane1.xy, 1.0 );
    local_far = nearfar.y; //-1000
    local_near = nearfar.x; //-2.5
    local_lights_num = num_lights;
		
		local_num_of_lights = 0;
		
		local_max_depth = 0;
		local_min_depth = 0x7f7fffff; // max float value
		local_depth_mask = 0;
	}
	
	barrier(); //local memory barrier
  
  float far = local_far;
  float near = local_near;
  
  //WARNING: need to linearize the depth in order to make it work...
  proj_a = -(far + near) / (far - near);
  proj_b = (-2 * far * near) / (far - near);
  float linear_depth = -proj_b / (raw_depth.x * 2 - 1 + proj_a);
  raw_depth.x = linear_depth / -far;
  
  int num_of_lights = local_lights_num;
  vec3 ll, ur;
  ll = local_ll.xyz;
  ur = local_ur.xyz;
	
	//check for skybox
	bool early_rejection = ( raw_depth.x > 0.999 || raw_depth.x < 0.001 );
  
	if( !early_rejection )
	{
		float tmp_depth = raw_depth.x;

		min_depth = min( min_depth, tmp_depth );
		max_depth = max( max_depth, tmp_depth );

		if( max_depth >= min_depth )
		{
			atomicMin( local_min_depth, floatBitsToUint( min_depth ) );
			atomicMax( local_max_depth, floatBitsToUint( max_depth ) );
		}
	}
	
	barrier(); //local memory barrier
	
	max_depth = uintBitsToFloat( local_max_depth );
	min_depth = uintBitsToFloat( local_min_depth );
	
	vec2 tile_scale = vec2( global_size.x, global_size.y ) * recip( local_size.x + local_size.y );
	vec2 tile_bias = tile_scale - vec2( group_id.x, group_id.y );
	
	float proj_11 = proj_mat[0].x;
	float proj_22 = proj_mat[1].y;
	
	vec4 c1 = vec4( proj_11 * tile_scale.x, 0.0, -tile_bias.x, 0.0 );
	vec4 c2 = vec4( 0.0, proj_22 * tile_scale.y, -tile_bias.y, 0.0 );
	vec4 c4 = vec4( 0.0, 0.0, -1.0, 0.0 );
	
	vec4 frustum_planes[6];
	
	frustum_planes[0] = c4 - c1;
	frustum_planes[1] = c4 + c1;
	frustum_planes[2] = c4 - c2;
	frustum_planes[3] = c4 + c2;
	frustum_planes[4] = vec4( 0.0, 0.0, 1.0, -min_depth * far ); //0, 0, 1, 2.5
	frustum_planes[5] = vec4( 0.0, 0.0, 1.0, -max_depth * far ); //0, 0, 1, 1000
	
	frustum_planes[0].xyz = normalize( frustum_planes[0].xyz );
	frustum_planes[1].xyz = normalize( frustum_planes[1].xyz );
	frustum_planes[2].xyz = normalize( frustum_planes[2].xyz );
	frustum_planes[3].xyz = normalize( frustum_planes[3].xyz );
  
  /*
   * Calculate per tile depth mask for 2.5D light culling
   */
   
  /**/
  float vs_min_depth = min_depth * -far;
  float vs_max_depth = max_depth * -far;
  float vs_depth = raw_depth.x * -far;
  
  float range = abs( vs_max_depth - vs_min_depth + 0.00001 ) / 32.0; //depth range in each tile
  
  vs_depth -= vs_min_depth; //so that min = 0
  float depth_slot = floor(vs_depth / range);
  
  //determine the cell for each pixel in the tile
  if( !early_rejection )
  {	
    //depth_mask = depth_mask | (1 << depth_slot)
    atomicOr( local_depth_mask, 1 << uint(depth_slot) );
  }
  
  barrier();
  /**/
	
	for( uint c = workgroup_index; c < num_of_lights; c += local_size.x * local_size.y )
	{
		bool in_frustum = true;
    int index = int(c);
    int lighting_type = lighting_type_data[index];

    float att_end = attenuation_end_data[index];
    vec3 light_pos = vs_position_data[index].xyz;
		vec4 lp = vec4( light_pos, 1.0 );
    
    /**/
    //calculate per light bitmask
    uint light_bitmask = 0;
    
    float light_z_min = -(light_pos.z + att_end); //light z min [0 ... 1000]
    float light_z_max = -(light_pos.z - att_end); //light z max [0 ... 1000]
    light_z_min -= vs_min_depth; //so that min = 0
    light_z_max -= vs_min_depth; //so that min = 0
    float depth_slot_min = floor(light_z_min / range);
    float depth_slot_max = floor(light_z_max / range);
    
    if( !( ( depth_slot_max > 31.0 && 
        depth_slot_min > 31.0 ) ||
      ( depth_slot_min < 0.0 && 
       depth_slot_max < 0.0 ) ) )
    {
      if( depth_slot_max > 30.0 )
        light_bitmask = uint(~0);
      else
        light_bitmask = (1 << (uint(depth_slot_max) + 1)) - 1;
        
      if( depth_slot_min > 0.0 )
        light_bitmask -= (1 << uint(depth_slot_min)) - 1;
    }
      
    in_frustum = in_frustum && bool(local_depth_mask & light_bitmask);
    /**/

		//manual unroll
		{
			float e = dot( frustum_planes[0], lp );
			in_frustum = in_frustum && ( e >= -att_end );
		}
		{
			float e = dot( frustum_planes[1], lp );
			in_frustum = in_frustum && ( e >= -att_end );
		}
		{
			float e = dot( frustum_planes[2], lp );
			in_frustum = in_frustum && ( e >= -att_end );
		}
		{
			float e = dot( frustum_planes[3], lp );
			in_frustum = in_frustum && ( e >= -att_end );
		}
		{
			float e = dot( frustum_planes[4], lp );
			in_frustum = in_frustum && ( e <= att_end );
		}
		{
			float e = dot( frustum_planes[5], lp );
			in_frustum = in_frustum && ( e >= -att_end );
		}

		if( lighting_type == 2 )
		{
			in_frustum = true;
		}

		if( in_frustum )
		{
			int li = atomicAdd( local_num_of_lights, 1 );
			local_lights[li] = int(index);
		}
	}
	
	barrier(); //local memory barrier
	
	if( !early_rejection )
	{
		pos_xy = mix( local_ll.xy, local_ur.xy, texel.xy );
    
		raw_depth.xyz = decode_linear_depth( raw_depth.x, pos_xy, far );
    
    vec4 ms_pos = inv_mv * vec4( raw_depth.xyz, 1 );
		raw_normal = texture( normals, texel );
		raw_normal.xyz = normalize( raw_normal.xyz * 2.0 - 1.0 );
    
    vec4 shadow_layers = 1-texture( layered_shadow_tex, texel );
    //vec4 translucent_shadow = uint_to_rgba8( texture( layered_translucency_tex, texel ).x );
    
    float gloss_factor = toksvig_aa( raw_normal.xyz, raw_albedo.w );
    
		vec3 view_dir = normalize( -raw_depth.xyz );
		
		for( int c = 0; c < local_num_of_lights; ++c )
		{
			int index = local_lights[c];
      int shadow_index = int(light_indices[index]);
      vec3 light_pos = vs_position_data[index].xyz;
      float light_radius = radius_data[index];
			float rcp_light_radius = recip( light_radius );
      int attenuation_type = attenuation_type_data[index];
      int lighting_type = lighting_type_data[index];
			//not used for now
			//int lighting_model = lighting_model_data[i];
			
			vec3 light_dir;
			float attenuation = 0.0;
			
			light_dir = light_pos - raw_depth.xyz;
			float distance = length( light_dir );
			light_dir = normalize( light_dir );
      
      //out_color.xyz += vec3( distance > light_radius && distance < light_radius + 0.5 ? 1.0 : 0.0 ); // && dot( -light_dir, spot_direction_data[i].xyz ) > spot_cutoff_data[i]
			
			float coeff = 0.0;
			
			if( attenuation_type == 0 )
			{
        float att_cutoff = attenuation_cutoff_data[index];
				coeff = max( distance - light_radius, 0.0 ) * rcp_light_radius + 1.0;
				attenuation = ( recip( coeff * coeff ) - att_cutoff ) * recip( 1.0 - att_cutoff );
			}
			else
			{
				attenuation = ( light_radius - distance ) * recip( light_radius );
			}
			
			if( lighting_type == 1 )
			{
        vec3 spot_direction = spot_direction_data[index].xyz;
        float spot_cos_cutoff = spot_cutoff_data[index];
				float spot_effect = dot( -light_dir, spot_direction );

				if( spot_effect > spot_cos_cutoff )
				{
          float spot_exponent = spot_exponent_data[index];
					spot_effect = pow( spot_effect, spot_exponent );

					//if( attenuation > 0.0 )
					//{
						attenuation = spot_effect * recip( 1.0 - attenuation ) * attenuation + 1.0;
					//	attenuation = spot_effect * native_recip( attenuation ) + 1.0f;
					//	attenuation = spot_effect * attenuation + 1.0f;
					//}
				}

				attenuation -= 1.0;
			}
			
			if( lighting_type == 2 )
			{
				light_dir = light_pos;
				attenuation = 1.0;
				//attenuation = 0;
			}
      
			if( attenuation > 0.0 )
			{
        float shadow = 1;
      
#if defined SSSS && !defined PCF && !defined PCSS && !defined HARD_SHADOW
        /**/        
        switch( layer_data[index] )
        {
          case 0:
          {
            shadow = shadow_layers.x;
            break;
          }
          case 1:
          {
            shadow = shadow_layers.y;
            break;
          }
          case 2:
          {
            shadow = shadow_layers.z;
            break;
          }
          case 3:
          {
            shadow = shadow_layers.w;
            break;
          }
        }
        /**/
#endif
      
#if !defined SSSS && (defined PCF || defined PCSS || defined HARD_SHADOW)
        /**/
      
       if( lighting_type == 1 ) //spot light shadow
        {
          vec4 ls_pos = spot_shadow_mat_data[index] * vec4(raw_depth.xyz, 1);
          
          if( ls_pos.w > 0.0 )
          {
            vec4 shadow_coord = ls_pos / ls_pos.w; //transform to tex coords (0...1)
            
            if( bounds_check(shadow_coord.x) &&
                bounds_check(shadow_coord.y) &&
                bounds_check(shadow_coord.z) )
            {
              float bias = 0.005;
              shadow_coord.z -= bias;
              
              float distance_from_light = texture( spot_shadow_tex, vec3(shadow_coord.xy, index) ).x;
              shadow = float( shadow_coord.z <= distance_from_light );
              
              //out_color.xyz += abs(vec3( distance_from_light ));
            }
          }
        }
        else if( lighting_type == 0 ) //point light shadow
        {
          vec4 model_light_dir = inv_view * vec4( light_dir.xyz, 0 );
          
          float axis[6];
          axis[0] = -model_light_dir.x;
          axis[1] = model_light_dir.x;
          axis[2] = -model_light_dir.y;
          axis[3] = model_light_dir.y;
          axis[4] = -model_light_dir.z;
          axis[5] = model_light_dir.z;
          
          int max_axis = 0;
          for( int d = 0; d < 6; ++d )
            if( axis[max_axis] < axis[d] )
              max_axis = d;
              
          vec3 ls_model_pos = ms_pos.xyz - ms_position_data[index].xyz;
          
          vec4 ls_pos = point_shadow_mat_data[index*6+max_axis] * vec4(ls_model_pos, 1);
          
          if( ls_pos.w > 0 )
          {
            //transform to tex coords (0...1)
            vec4 shadow_coord = ls_pos / ls_pos.w; 
            shadow_coord = shadow_coord * 0.5 + 0.5;
            
            if( bounds_check(shadow_coord.x) &&
                bounds_check(shadow_coord.y) &&
                bounds_check(shadow_coord.z) )
            {
              vec3 texcoord = normalize(ls_model_pos);
              shadow = 0;
              
              int count = 0;
              int size = 11;
              float scale = 1.0;
              float k = 50;
              float bias = 0.001;
              int search_samples = 5;
          
#ifdef PCSS
              //PCSS
              float ls = light_size_data[index];
              float lsndcpos = abs(shadow_coord.z) * 2 - 1;
              float lsviewdepth = -proj_b / (lsndcpos + proj_a);
              float lsprojz = lsndcpos * -lsviewdepth;
              proj_a = -(radius_data[index] + 1) / (radius_data[index] - 1);
              proj_b = (-2 * radius_data[index] * 1) / (radius_data[index] - 1);   
              float blocker = find_blocker_point( index, bias, proj_a, proj_b, right_vec[max_axis], up_vec[max_axis], point_shadow_tex, texcoord, shadow_coord, ls / abs(lsprojz)*10, search_samples );
              float penumbra = abs(estimate_penumbra( abs(lsprojz), abs(blocker) )) * ls * ls;
              float filter_radius = 2 * penumbra / size;
              scale = abs(filter_radius) / (linear_depth);

              //out_color = vec4(abs(filter_radius));
#endif

#ifdef PCF
              scale /= 1024;
              scale *= 3;
#endif
      
#if defined PCF || defined PCSS
              //PCF
              texcoord -= scale * (right_vec[max_axis] + up_vec[max_axis]) * 0.5;
              for( int y = 0; y < size; ++y )
                for( int x = 0; x < size; ++x )
                {
                  float distance_from_light = texture( point_shadow_tex, vec4(texcoord + scale * (x * right_vec[max_axis] + y * up_vec[max_axis]), shadow_index) ).x;
#ifdef EXPONENTIAL_SHADOWS
                  shadow += clamp( 2.0 - exp((abs(shadow_coord.z) - distance_from_light) * k), 0, 1 );
#else
                  shadow += float( abs(shadow_coord.z) - bias < distance_from_light );
#endif
                  ++count;
                }
              shadow = shadow / float(count);
              
#ifdef PCSS
              /*shadow = float(check_sanity(blocker)||
                             check_sanity(penumbra)||
                             check_sanity(scale)||
                             check_sanity(ls)||
                             check_sanity(lsndcpos)||
                             check_sanity(lsviewdepth)||
                             check_sanity(lsprojz)||
                             check_sanity(proj_a)||
                             check_sanity(proj_b));*/
              //shadow = scale*1000;
              //shadow = -lsprojz/100;
              //shadow = count/100.0;
#endif

#endif

#ifdef HARD_SHADOW
              //hard shadow
              float distance_from_light = texture( point_shadow_tex, vec4(texcoord, shadow_index) ).x;
#ifdef EXPONENTIAL_SHADOWS
              shadow = clamp( 2.0 - exp((abs(shadow_coord.z) - distance_from_light) * k), 0, 1 );
#else
              shadow = float( abs(shadow_coord.z) - bias < distance_from_light );
#endif
#endif
              
              //out_color.xyz = vec3(distance_from_light);
              //shadow = 1;
              //out_color.xyz = texcoord;
            }
            //else shadow = 0;
            
            //out_color.xyz = vec3(float((index+1)*0.5));
          }
        }
        /**/
#endif

        if( shadow > 0.0 )
        {
          /**/
          //if( index != 13 )
          {
            out_color.xyz += brdf( index,
                         raw_albedo.xyz,
                         raw_normal.xyz,
                         light_dir,
                         view_dir,
                         raw_albedo.w,
                         attenuation,
                         diffuse_color_data[index].xyz,
                         specular_color_data[index].xyz )
                         * shadow;
          }
          /*else
          {
            out_color.xyz += brdf( index,
                         raw_albedo.xyz,
                         raw_normal.xyz,
                         light_dir,
                         view_dir,
                         raw_albedo.w,
                         attenuation,
                         data ) * translucent_shadow.xyz * (1-translucent_shadow.w);
            //out_color = translucent_shadow;
          }*/
                       
          /**/
          //out_color.xyz += vec3( 0, shadow, 0 );
          //out_color.xyz += vec3(1, 0, 0) * float(attenuation>0);
          //out_color.xyz = vec3(float((index+1)*0.5));
          //out_color = vec4(shadow_layers.w);
        }
        
        //out_color.xyz += float(attenuation>0.0 && attenuation<0.1);
        out_color.xyz += attenuation * vec3(0, 0.75, 1) * 0.01;
        //out_color = vec4(layer_data[index]*0.33);
        //out_color = shadow_layers;
			}
		}
		
		//out_color.xyz = raw_normal.xyz; //view space normal
		//out_color.xyz = raw_albedo.xyz; //albedo
		//out_color.xyz = (float3)(raw_albedo.w); //specular intensity
		//out_color.xyz = vec3(raw_depth.x); //view space linear depth
		//out_color.xyz = raw_depth.xyz; //view space position

		float lcp = local_num_of_lights * 0.1;
	  //out_color += vec4( lcp );
		
		//out_color.xyz = vec3(abs(normalize(vs_position_data[13].xyz)));
		//out_color = vec4(float(out_color.x > 0));
	}
  else
  {
    out_color.xyz = vec3(0, 0.75, 1);
  }
	
  //out_color = vec4(raw_ssdo);
  
	if( global_id.x <= global_size.x && global_id.y <= global_size.y )
	{
		imageStore( result, global_id, vec4( clamp(gamma_correct(tonemap(out_color.xyz)), 0.0, 1.0), 1.0 ) );
    
    //for( uint c = workgroup_index; c < local_num_of_lights; c += local_size.x * local_size.y )
    {
      //imageStore( local_lights_result, int(group_id.x * group_size.y + group_id.y + c), vec4(intBitsToFloat(local_lights[c])) );
      //imageStore( local_lights_result, int(0), vec4(intBitsToFloat(1)) );
    }
    
    //imageStore( result, global_id, vec4( clamp(raw_normal.xyz, 0.0, 1.0), 1.0 ) );
    //imageStore( result, global_id, vec4( clamp(raw_depth.xyz, 0.0, 1.0), 1.0 ) );
    //imageStore( result, global_id, vec4( vec3(clamp(abs(raw_depth.z/100), 0.0, 1.0)), 1.0 ) );
	}
}
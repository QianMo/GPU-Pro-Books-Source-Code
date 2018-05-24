/*

Copyright 2013,2014 Sergio Ruiz, Benjamin Hernandez

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>

In case you, or any of your employees or students, publish any article or
other material resulting from the use of this  software, that publication
must cite the following references:

Sergio Ruiz, Benjamin Hernandez, Adriana Alvarado, and Isaac Rudomin. 2013.
Reducing Memory Requirements for Diverse Animated Crowds. In Proceedings of
Motion on Games (MIG '13). ACM, New York, NY, USA, , Article 55 , 10 pages.
DOI: http://dx.doi.org/10.1145/2522628.2522901

Sergio Ruiz and Benjamin Hernandez. 2015. A Parallel Solver for Markov Decision Process
in Crowd Simulations. Fourteenth Mexican International Conference on Artificial
Intelligence (MICAI), Cuernavaca, 2015, pp. 107-116.
DOI: 10.1109/MICAI.2015.23

*/
//current_positions[index+0]	= current_position.x;
//current_positions[index+1]	= current_tile;
//current_positions[index+2]	= current_position.y;
//current_positions[index+3]	= orientation;

//original_positions[index+0]	= original_position.x;
//original_positions[index+1]	= density_threshold;
//original_positions[index+2]	= original_position.y;
//original_positions[index+3]	= displacement_delta;

#ifndef __MDP_HEXAGON_PATHS_KERNEL
#define __MDP_HEXAGON_PATHS_KERNEL

//#define DEBUG_TIME
#define DEG2RAD	0.01745329251994329576f
#define RAD2DEG 57.29577951308232087679f
#define SQRT3 1.73205080757f
#define SIGN(a) ((a) > 0.0f ? +1.0f : ((a) < 0.0f ? -1.0f : 0.0f))
//#include <stdio.h>
//
//=======================================================================================
//
__global__ void mdp_hexagon_kernel(	float*			current_positions,
									float*			original_positions,
									float*			policy,
									float*			density,
									float			scene_width,
									float			scene_height,
									float			scene_width_in_tiles,
									float			scene_height_in_tiles,
									float			S,
									float			H,
									float			time,
									bool			reset				)
{
    unsigned int	x				= blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int	y				= blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int	index			= 4 * ( x + y * blockDim.x * gridDim.x );

	if( current_positions[index+1] > -2.0f )
	{
		unsigned int	num_tiles		= (unsigned int)(scene_width_in_tiles * scene_height_in_tiles);
		float			R				= H / SQRT3;
		float			hH				= H / 2.0f;
		float			delta			= original_positions[index+3];

		float2			hscene_dims;
		hscene_dims.x					= scene_width  / 2.0f;
		hscene_dims.y					= scene_height / 2.0f;

		//->RESET_POSITIONS
		if( reset )
		{
			original_positions[index+0]		= current_positions[index+0];
			current_positions[index+1]		= -1.0f;
			original_positions[index+2]		= current_positions[index+2];
		}
		//<-RESET_POSITIONS

		float2			curr_pos;
		curr_pos.x						= current_positions[index+0];
		curr_pos.y						= current_positions[index+2];
		float2			curr_abs_pos;
		curr_abs_pos.x					= curr_pos.x + hscene_dims.x;					// x
		curr_abs_pos.y					= curr_pos.y + hscene_dims.y;					// y
		float2			curr_tileXY;
		curr_tileXY.x					= floor( curr_abs_pos.x / S );					// it
		float cyDisp					= fmod( curr_tileXY.x, 2.0f ) * hH;
		float cYts						= curr_abs_pos.y - cyDisp;						// yts
		curr_tileXY.y					= floor( cYts / H );							// jt
		float2			curr_in_tileXY;
		curr_in_tileXY.x				= curr_abs_pos.x - curr_tileXY.x * S;			// xt
		curr_in_tileXY.y				= cYts           - curr_tileXY.y * H;			// yt
		float2			curr_hex_tileXY;
		float			cTestX			= R * abs( 0.5f - curr_in_tileXY.y / H );
		if( curr_in_tileXY.x > cTestX )
		{
			curr_hex_tileXY.x			= curr_tileXY.x;
			curr_hex_tileXY.y			= curr_tileXY.y;
		}
		else
		{
			curr_hex_tileXY.x			= curr_tileXY.x - 1.0f;
			float		deltaJ			= 0.0f;
			if( curr_in_tileXY.y > hH )
			{
				deltaJ					= 1.0f;
			}
			curr_hex_tileXY.y			= curr_tileXY.y - fmod( curr_hex_tileXY.x, 2.0f ) + deltaJ;
		}
		unsigned int	curr_tile		= (unsigned int)(curr_hex_tileXY.y * scene_width_in_tiles + curr_hex_tileXY.x);
		unsigned int	prev_tile		= num_tiles;
		if( current_positions[index+1] > -1.0f )
		{
			prev_tile					= (unsigned int)current_positions[index+1];
		}

		if( curr_tile != prev_tile )
		{
			float2			orig_pos;
			orig_pos.x						= original_positions[index+0];
			float			dThreshold		= original_positions[index+1];
			orig_pos.y						= original_positions[index+2];
			float2			orig_abs_pos;
			orig_abs_pos.x					= orig_pos.x + hscene_dims.x;					// x
			orig_abs_pos.y					= orig_pos.y + hscene_dims.y;					// y
			float2			orig_tileXY;
			orig_tileXY.x					= floor( orig_abs_pos.x / S );					// it
			float oyDisp					= fmod( orig_tileXY.x, 2.0f ) * hH;
			float oYts						= orig_abs_pos.y - oyDisp;						// yts
			orig_tileXY.y					= floor( oYts / H );							// jt
			float2			orig_in_tileXY;
			orig_in_tileXY.x				= orig_abs_pos.x - orig_tileXY.x * S;			// xt
			orig_in_tileXY.y				= oYts           - orig_tileXY.y * H;			// yt
			float2			orig_hex_tileXY;
			float			oTestX			= R * abs( 0.5f - orig_in_tileXY.y / H );
			if( orig_in_tileXY.x > oTestX )
			{
				orig_hex_tileXY.x			= orig_tileXY.x;
				orig_hex_tileXY.y			= orig_tileXY.y;
			}
			else
			{
				orig_hex_tileXY.x			= orig_tileXY.x - 1.0f;
				float		deltaJ			= 0.0f;
				if( orig_in_tileXY.y > hH )
				{
					deltaJ					= 1.0f;
				}
				orig_hex_tileXY.y			= orig_tileXY.y - fmod( orig_hex_tileXY.x, 2.0f ) + deltaJ;
			}
			float2			orig_hex_tile_center;
			orig_hex_tile_center.x			= (orig_hex_tileXY.x * S) + R;
			orig_hex_tile_center.y			= (orig_hex_tileXY.y * H) + oyDisp + hH;
			float2			orig_disp;
			orig_disp.x						= orig_abs_pos.x - orig_hex_tile_center.x;
			orig_disp.y						= orig_abs_pos.y - orig_hex_tile_center.y;

			float2			curr_hex_tile_center;
			curr_hex_tile_center.x			= (curr_hex_tileXY.x * S) + R;
			curr_hex_tile_center.y			= (curr_hex_tileXY.y * H) + cyDisp + hH;
			float2			curr_disp;
			curr_disp.x						= curr_abs_pos.x - curr_hex_tile_center.x;
			curr_disp.y						= curr_abs_pos.y - curr_hex_tile_center.y;

			float			dX				= curr_disp.x - orig_disp.x;
			dX							   *= dX;
			float			dY				= curr_disp.y - orig_disp.y;
			dY							   *= dY;
			float			d2				= dX + dY;
			float			r2				= 4*delta*delta;

			float curr_policy				= policy[curr_tile];

			if( d2 < r2 )
			{

				if( curr_policy == 7.0f )
				{
					current_positions[index+0]	= 1000000.0f;
					current_positions[index+1]	= -2.0f;
					current_positions[index+2]	= 1000000.0f;
					if( prev_tile < num_tiles )
					{
						density[prev_tile] -= 1.0f;
					}
				}




//->ADJUST_CURR_POLICY_ACCORDING_TO_DENSITY
				float deg_angle = 270.0f - ((curr_policy + 1.0f) * 60.0f);
				float rad_angle = DEG2RAD * deg_angle;
				float xFactor = SIGN( cos( rad_angle ) );
				float nextX = curr_tileXY.x + xFactor;
				float yFactor = SIGN( sin( rad_angle ) ) - fmod( nextX, 2.0f );
				unsigned int next_tile = (unsigned int)((curr_tileXY.y + yFactor) * scene_width_in_tiles + nextX);

				if( density[next_tile] >= dThreshold )
				{
					float alt_dirs[5] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
					alt_dirs[0] = fmod( curr_policy + 5.0f, 6.0f );
					alt_dirs[1] = fmod( curr_policy + 1.0f, 6.0f );
					alt_dirs[2] = fmod( curr_policy + 4.0f, 6.0f );
					alt_dirs[3] = fmod( curr_policy + 2.0f, 6.0f );
					alt_dirs[4] = fmod( curr_policy + 3.0f, 6.0f );
					float least_dense_dir = 0.0f;
					float least_density = dThreshold;
					bool found = false;
					for( unsigned int i = 0; i < 5; i++ )
					{
						float test_deg_angle = 270.0f - ((alt_dirs[i] + 1.0f) * 60.0f);
						float test_rad_angle = DEG2RAD * test_deg_angle;
						float test_xFactor = SIGN( cos( test_rad_angle ) );
						float testX = curr_tileXY.x + test_xFactor;
						float test_yFactor = SIGN( sin( test_rad_angle ) ) - fmod( testX, 2.0f );
						float testY = curr_tileXY.y + test_yFactor;						
						if(	testX >= 0.0f && 
							testX < scene_width_in_tiles && 
							testY >= 0.0f && 
							testY < scene_height_in_tiles )
						{
							unsigned int test_tile = (unsigned int)(testY * scene_width_in_tiles + testX);
							if( density[test_tile] < least_density && policy[test_tile] != 6.0f )
							{
								found = true;
								next_tile = test_tile;
								least_dense_dir = alt_dirs[i];
								least_density = density[test_tile];
							}
						}
						if( i % 2 == 1 && found )
						{
							break;
						}
					}
					if( found )
					{
						curr_policy = least_dense_dir;
					}
				}
//<-ADJUST_CURR_POLICY_ACCORDING_TO_DENSITY




				float alpha					= ((curr_policy + 1.0f) * 60.0f) - 30.0f; // 30, 90, 150, ..., 330
				float gamma					= alpha - 150.0f;
				float phi					= DEG2RAD * gamma;

				current_positions[index+0]	= curr_hex_tile_center.x + orig_disp.x - hscene_dims.x;
				current_positions[index+1]	= (float)curr_tile;
				current_positions[index+2]	= curr_hex_tile_center.y + orig_disp.y - hscene_dims.y;
				current_positions[index+3]	= phi;

				if( prev_tile < num_tiles )
				{
					density[prev_tile] -= 1.0f;
					density[curr_tile] += 1.0f;
				}

			}
			else
			{
				float theta					= (RAD2DEG * current_positions[index+3]) + 270.0f; //150+120
				theta					   *= DEG2RAD;

				current_positions[index+0]	= curr_pos.x + delta * cos( theta );
				current_positions[index+2]	= curr_pos.y - delta * sin( theta );
			}
		}
		else
		{
			float theta					= (RAD2DEG * current_positions[index+3]) + 270.0f; //150+120
			theta					   *= DEG2RAD;

			current_positions[index+0]	= curr_pos.x + delta * cos( theta );
			current_positions[index+2]	= curr_pos.y - delta * sin( theta );
		}
	}
}
//
//=======================================================================================
//
extern "C" void launch_mdp_hexagon_kernel(	float*			current_positions,
											float*			original_positions,
											float*			policy,
											float*			density,
											float			scene_width,
											float			scene_height,
											float			scene_width_in_tiles,
											float			scene_height_in_tiles,
											float			tile_side,
											float			tile_height,
											unsigned int	mesh_width,
											unsigned int	mesh_height,
											float			time,
											bool			policy_reset		)
{
#ifdef DEBUG_TIME
	cudaEvent_t										start;
	cudaEvent_t										stop;
	float											elapsedTime		= 0.0f;
#endif

	dim3 block( 4, 4, 1 );
    dim3 grid( mesh_width / block.x, mesh_height / block.y, 1 );

#ifdef DEBUG_TIME
	cudaEventCreate( &start );
	cudaEventCreate( &stop  );
	cudaEventRecord( start, 0 );
#endif

	mdp_hexagon_kernel<<<grid, block>>>(	current_positions,
											original_positions,
											policy,
											density,
											scene_width,
											scene_height,
											scene_width_in_tiles,
											scene_height_in_tiles,
											tile_side,
											tile_height,
											time,
											policy_reset		);

#ifdef DEBUG_TIME
	cudaEventRecord( stop, 0 );
	cudaEventSynchronize( stop );
	cudaEventElapsedTime( &elapsedTime, start, stop );
	printf( "HEXAGON_PATHS_TIME:  %010.6f(ms)\n", elapsedTime );
	//->CLEAN_UP
	cudaEventDestroy( start );
	cudaEventDestroy( stop  );
	//<-CLEAN_UP
#endif

}
//
//=======================================================================================
//
#endif

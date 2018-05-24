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
//current_positions[index+1]	= previous_tile;
//current_positions[index+2]	= current_position.y;
//current_positions[index+3]	= orientation;

//original_positions[index+0]	= original_position.x;
//original_positions[index+1]	= density_threshold;
//original_positions[index+2]	= original_position.y;
//original_positions[index+3]	= displacement_delta;

#ifndef __MDP_SQUARE_PATHS_KERNEL
#define __MDP_SQUARE_PATHS_KERNEL

#define DEG2RAD	0.01745329251994329576f
#define RAD2DEG 57.29577951308232087679f
#define SIGN(a) ((a) > 0.0f ? +1.0f : ((a) < 0.0f ? -1.0f : 0.0f))
//
//=======================================================================================
//
__global__ void mdp_square_kernel(	float*			current_positions,
									float*			original_positions,
									float*			policy,
									float*			density,
									float			scene_width_in_tiles,
									float			scene_height_in_tiles,
									float			tile_width,
									float			tile_height,
									float			time,
									bool			reset				)
{
    unsigned int	x				= blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int	y				= blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int	index			= 4 * ( x + y * blockDim.x * gridDim.x );

	if( current_positions[index+1] > -2.0f )
	{
		unsigned int	num_tiles		= (unsigned int)(scene_width_in_tiles * scene_height_in_tiles);
		float2			scene_dims;
		scene_dims.x					= scene_width_in_tiles * tile_width;
		scene_dims.y					= scene_height_in_tiles * tile_height;
		float2			hscene_dims;
		hscene_dims.x					= scene_dims.x / 2.0f;
		hscene_dims.y					= scene_dims.y / 2.0f;

		//->RESET_POSITIONS
		//if( reset )
		//{
			//original_positions[index+0]		= current_positions[index+0];
			//current_positions[index+1]		= -1.0f;
			//original_positions[index+2]		= current_positions[index+2];
		//}
		//<-RESET_POSITIONS

		float2			orig_pos;
		orig_pos.x						= original_positions[index+0];
		float			dThreshold		= original_positions[index+1];
		orig_pos.y						= original_positions[index+2];
		float2			orig_abs_pos;
		orig_abs_pos.x					= orig_pos.x+hscene_dims.x;
		orig_abs_pos.y					= orig_pos.y+hscene_dims.y;
		float2			orig_tileXY;
		orig_tileXY.x					= trunc ( orig_abs_pos.x / tile_width	);
		orig_tileXY.y					= trunc	( orig_abs_pos.y / tile_height	);
		float2			orig_tile_center;
		orig_tile_center.x				= (orig_tileXY.x * tile_width)  + tile_width  / 2.0f;
		orig_tile_center.y				= (orig_tileXY.y * tile_height) + tile_height / 2.0f;
		float2			orig_disp;
		orig_disp.x						= orig_abs_pos.x - orig_tile_center.x;
		orig_disp.y						= orig_abs_pos.y - orig_tile_center.y;


		float2			curr_pos;
		curr_pos.x						= current_positions[index+0];
		curr_pos.y						= current_positions[index+2];
		float2			curr_abs_pos;
		curr_abs_pos.x					= curr_pos.x+hscene_dims.x;
		curr_abs_pos.y					= curr_pos.y+hscene_dims.y;
		float2			curr_tileXY;
		curr_tileXY.x					= trunc	( curr_abs_pos.x / tile_width	);
		curr_tileXY.y					= trunc	( curr_abs_pos.y / tile_height	);
		float2			curr_tile_center;
		curr_tile_center.x				= (curr_tileXY.x * tile_width)  + tile_width  / 2.0f;
		curr_tile_center.y				= (curr_tileXY.y * tile_height) + tile_height / 2.0f;
		float2			curr_disp;
		curr_disp.x						= curr_abs_pos.x - curr_tile_center.x;
		curr_disp.y						= curr_abs_pos.y - curr_tile_center.y;
		float			delta			= original_positions[index+3];
		float			delta2			= 2.0;

		float diffX = abs(curr_disp.x-orig_disp.x);
		float diffY = abs(curr_disp.y-orig_disp.y);

		unsigned int	curr_tile		= (unsigned int)(curr_tileXY.y * scene_width_in_tiles + curr_tileXY.x);
		unsigned int	prev_tile		= num_tiles;
		if( current_positions[index+1] > -1.0f )
		{
			prev_tile					= (unsigned int)current_positions[index+1];
		}

		float curr_policy				= policy[curr_tile];

		if( diffX < delta2 && diffY < delta2 && curr_tile != prev_tile )
		{
			if( curr_policy == 9.0f )
			{
				current_positions[index+0]	= 1000000.0f;
				current_positions[index+1]	= -2.0f;
				current_positions[index+2]	= 1000000.0f;
				if( prev_tile < num_tiles )
				{
					density[prev_tile] -= 1.0f;
				}
			}
			else
			{

//->ADJUST_CURR_POLICY_ACCORDING_TO_DENSITY
				float deg_angle = 225.0f - (curr_policy * 45.0f);
				float rad_angle = DEG2RAD * deg_angle;
				float yFactor = SIGN( sin( rad_angle ) );
				float xFactor = SIGN( cos( rad_angle ) );
				unsigned int next_tile = (unsigned int)((curr_tileXY.y + yFactor) * scene_width_in_tiles + curr_tileXY.x + xFactor);

				if( density[next_tile] >= dThreshold )
				{
					float alt_dirs[7] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
					alt_dirs[0] = fmod( curr_policy + 7.0f, 8.0f );
					alt_dirs[1] = fmod( curr_policy + 1.0f, 8.0f );
					alt_dirs[2] = fmod( curr_policy + 6.0f, 8.0f );
					alt_dirs[3] = fmod( curr_policy + 2.0f, 8.0f );
					alt_dirs[4] = fmod( curr_policy + 5.0f, 8.0f );
					alt_dirs[5] = fmod( curr_policy + 3.0f, 8.0f );
					alt_dirs[6] = fmod( curr_policy + 4.0f, 8.0f );
					float least_dense_dir = 0.0f;
					float least_density = dThreshold;
					bool found = false;
					for( unsigned int i = 0; i < 7; i++ )
					{
						float test_deg_angle = 225.0f - (alt_dirs[i] * 45.0f);
						float test_rad_angle = DEG2RAD * test_deg_angle;
						float test_yFactor = SIGN( sin( test_rad_angle ) );
						float test_xFactor = SIGN( cos( test_rad_angle ) );
						float testY = curr_tileXY.y + test_yFactor;
						float testX = curr_tileXY.x + test_xFactor;
						if(	testX >= 0.0f &&
							testX < scene_width_in_tiles &&
							testY >= 0.0f &&
							testY < scene_height_in_tiles )
						{
							unsigned int test_tile = (unsigned int)(testY * scene_width_in_tiles + testX);
							if( density[test_tile] < least_density && policy[test_tile] != 8.0f )
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


				curr_disp.x						= 0.0f;
				curr_disp.y						= 0.0f;

				if( curr_policy == 0.0 )
				{
					curr_disp.x -= delta;
					curr_disp.y -= delta;
					current_positions[index+3]	= DEG2RAD*-135.0f;

					if( curr_tileXY.y > 0.0 )
					{
						unsigned int test_tile = (unsigned int)((curr_tileXY.y - 1.0) * scene_width_in_tiles + curr_tileXY.x);
						if( policy[test_tile] == 8.0 )
						{
							float2 p1;
							p1.x						= curr_abs_pos.x;
							p1.y						= curr_abs_pos.y;
							float2 p2;
							p2.x						= p1.x - tile_width;
							p2.y						= p1.y - tile_height;
							float2 pA, pB, pC, pD;
							pA.x						= curr_tileXY.x * tile_width;
							pB.x						= pA.x + tile_width;
							pC.x						= pA.x;
							pD.x						= pB.x;
							pA.y						= curr_tileXY.y * tile_height;
							pB.y						= pA.y;
							pC.y						= (curr_tileXY.y - 1.0) * tile_height;
							pD.y						= pC.y;
							float q1					= p2.y-p1.y;
							float q2					= p1.x-p2.x;
							float q3					= p2.x*p1.y-p1.x*p2.y;
							float fA					= q1*pA.x+q2*pA.y+q3;
							float fB					= q1*pB.x+q2*pB.y+q3;
							float fC					= q1*pC.x+q2*pC.y+q3;
							float fD					= q1*pD.x+q2*pD.y+q3;
							if( !(fA > 0.0 && fB > 0.0 && fC > 0.0 && fD > 0.0) && !(fA < 0.0 && fB < 0.0 && fC < 0.0 && fD < 0.0) )
							{
								curr_disp.y = 0.0;
								curr_disp.x = -delta;
								current_positions[index+3] = DEG2RAD*-90.0f;
							}
						}
					}
					if( curr_tileXY.x > 0.0 )
					{
						unsigned int test_tile = (unsigned int)(curr_tileXY.y * scene_width_in_tiles + curr_tileXY.x - 1.0);
						if( policy[test_tile] == 8.0 )
						{
							float2 p1;
							p1.x						= curr_abs_pos.x;
							p1.y						= curr_abs_pos.y;
							float2 p2;
							p2.x						= p1.x - tile_width;
							p2.y						= p1.y - tile_height;
							float2 pA, pB, pC, pD;
							pA.x						= (curr_tileXY.x - 1.0) * tile_width;
							pB.x						= curr_tileXY.x * tile_width;
							pC.x						= pA.x;
							pD.x						= pB.x;
							pA.y						= (curr_tileXY.y + 1.0) * tile_height;
							pB.y						= pA.y;
							pC.y						= curr_tileXY.y * tile_height;
							pD.y						= pC.y;
							float q1					= p2.y-p1.y;
							float q2					= p1.x-p2.x;
							float q3					= p2.x*p1.y-p1.x*p2.y;
							float fA					= q1*pA.x+q2*pA.y+q3;
							float fB					= q1*pB.x+q2*pB.y+q3;
							float fC					= q1*pC.x+q2*pC.y+q3;
							float fD					= q1*pD.x+q2*pD.y+q3;
							if( !(fA > 0.0 && fB > 0.0 && fC > 0.0 && fD > 0.0) && !(fA < 0.0 && fB < 0.0 && fC < 0.0 && fD < 0.0) )
							{
								curr_disp.x = 0.0;
								curr_disp.y = -delta;
								current_positions[index+3] = DEG2RAD*180.0f;
							}
						}
					}
				}
				else if( curr_policy == 1.0 )
				{
					curr_disp.x -= delta;
					current_positions[index+3]	= DEG2RAD*-90.0f;
				}
				else if( curr_policy == 2.0 )
				{
					curr_disp.x -= delta;
					curr_disp.y += delta;
					current_positions[index+3]	= DEG2RAD*-45.0f;

					if( (curr_tileXY.y + 1.0) < scene_height_in_tiles )
					{
						unsigned int test_tile = (unsigned int)((curr_tileXY.y + 1.0) * scene_width_in_tiles + curr_tileXY.x);
						if( policy[test_tile] == 8.0 )
						{
							float2 p1;
							p1.x						= curr_abs_pos.x;
							p1.y						= curr_abs_pos.y;
							float2 p2;
							p2.x						= p1.x - tile_width;
							p2.y						= p1.y + tile_height;
							float2 pA, pB, pC, pD;
							pA.x						= curr_tileXY.x * tile_width;
							pB.x						= pA.x + tile_width;
							pC.x						= pA.x;
							pD.x						= pB.x;
							pA.y						= (curr_tileXY.y + 2.0) * tile_height;
							pB.y						= pA.y;
							pC.y						= (curr_tileXY.y + 1.0) * tile_height;
							pD.y						= pC.y;
							float q1					= p2.y-p1.y;
							float q2					= p1.x-p2.x;
							float q3					= p2.x*p1.y-p1.x*p2.y;
							float fA					= q1*pA.x+q2*pA.y+q3;
							float fB					= q1*pB.x+q2*pB.y+q3;
							float fC					= q1*pC.x+q2*pC.y+q3;
							float fD					= q1*pD.x+q2*pD.y+q3;
							if( !(fA > 0.0 && fB > 0.0 && fC > 0.0 && fD > 0.0) && !(fA < 0.0 && fB < 0.0 && fC < 0.0 && fD < 0.0) )
							{
								curr_disp.y = 0.0;
								curr_disp.x = -delta;
								current_positions[index+3]	= DEG2RAD*-90.0f;
							}
						}
					}
					if( curr_tileXY.x > 0.0 )
					{
						unsigned int test_tile = (unsigned int)(curr_tileXY.y * scene_width_in_tiles + curr_tileXY.x - 1.0);
						if( policy[test_tile] == 8.0 )
						{
							float2 p1;
							p1.x						= curr_abs_pos.x;
							p1.y						= curr_abs_pos.y;
							float2 p2;
							p2.x						= p1.x - tile_width;
							p2.y						= p1.y + tile_height;
							float2 pA, pB, pC, pD;
							pA.x						= (curr_tileXY.x - 1.0) * tile_width;
							pB.x						= curr_tileXY.x * tile_width;
							pC.x						= pA.x;
							pD.x						= pB.x;
							pA.y						= (curr_tileXY.y + 1.0) * tile_height;
							pB.y						= pA.y;
							pC.y						= curr_tileXY.y * tile_height;
							pD.y						= pC.y;
							float q1					= p2.y-p1.y;
							float q2					= p1.x-p2.x;
							float q3					= p2.x*p1.y-p1.x*p2.y;
							float fA					= q1*pA.x+q2*pA.y+q3;
							float fB					= q1*pB.x+q2*pB.y+q3;
							float fC					= q1*pC.x+q2*pC.y+q3;
							float fD					= q1*pD.x+q2*pD.y+q3;
							if( !(fA > 0.0 && fB > 0.0 && fC > 0.0 && fD > 0.0) && !(fA < 0.0 && fB < 0.0 && fC < 0.0 && fD < 0.0) )
							{
								curr_disp.x = 0.0;
								curr_disp.y = delta;
								current_positions[index+3]	= 0.0f;
							}
						}
					}
				}
				else if( curr_policy == 3.0 )
				{
					curr_disp.y += delta;
					current_positions[index+3] = 0.0f;
				}
				else if( curr_policy == 4.0 )
				{
					curr_disp.x += delta;
					curr_disp.y += delta;
					current_positions[index+3]	= DEG2RAD*45.0f;

					if( (curr_tileXY.y + 1.0) < scene_height_in_tiles )
					{
						unsigned int test_tile = (unsigned int)((curr_tileXY.y + 1.0) * scene_width_in_tiles + curr_tileXY.x);
						if( policy[test_tile] == 8.0 )
						{
							float2 p1;
							p1.x						= curr_abs_pos.x;
							p1.y						= curr_abs_pos.y;
							float2 p2;
							p2.x						= p1.x + tile_width;
							p2.y						= p1.y + tile_height;
							float2 pA, pB, pC, pD;
							pA.x						= curr_tileXY.x * tile_width;
							pB.x						= pA.x + tile_width;
							pC.x						= pA.x;
							pD.x						= pB.x;
							pA.y						= (curr_tileXY.y + 2.0) * tile_height;
							pB.y						= pA.y;
							pC.y						= (curr_tileXY.y + 1.0) * tile_height;
							pD.y						= pC.y;
							float q1					= p2.y-p1.y;
							float q2					= p1.x-p2.x;
							float q3					= p2.x*p1.y-p1.x*p2.y;
							float fA					= q1*pA.x+q2*pA.y+q3;
							float fB					= q1*pB.x+q2*pB.y+q3;
							float fC					= q1*pC.x+q2*pC.y+q3;
							float fD					= q1*pD.x+q2*pD.y+q3;
							if( !(fA > 0.0 && fB > 0.0 && fC > 0.0 && fD > 0.0) && !(fA < 0.0 && fB < 0.0 && fC < 0.0 && fD < 0.0) )
							{
								curr_disp.y = 0.0;
								curr_disp.x = delta;
								current_positions[index+3]	= DEG2RAD*90.0f;
							}
						}
					}
					if( (curr_tileXY.x + 1.0) < scene_width_in_tiles )
					{
						unsigned int test_tile = (unsigned int)(curr_tileXY.y * scene_width_in_tiles + curr_tileXY.x + 1.0);
						if( policy[test_tile] == 8.0 )
						{
							float2 p1;
							p1.x						= curr_abs_pos.x;
							p1.y						= curr_abs_pos.y;
							float2 p2;
							p2.x						= p1.x + tile_width;
							p2.y						= p1.y + tile_height;
							float2 pA, pB, pC, pD;
							pA.x						= (curr_tileXY.x + 1.0) * tile_width;
							pB.x						= (curr_tileXY.x + 2.0) * tile_width;
							pC.x						= pA.x;
							pD.x						= pB.x;
							pA.y						= (curr_tileXY.y + 1.0) * tile_height;
							pB.y						= pA.y;
							pC.y						= curr_tileXY.y * tile_height;
							pD.y						= pC.y;
							float q1					= p2.y-p1.y;
							float q2					= p1.x-p2.x;
							float q3					= p2.x*p1.y-p1.x*p2.y;
							float fA					= q1*pA.x+q2*pA.y+q3;
							float fB					= q1*pB.x+q2*pB.y+q3;
							float fC					= q1*pC.x+q2*pC.y+q3;
							float fD					= q1*pD.x+q2*pD.y+q3;
							if( !(fA > 0.0 && fB > 0.0 && fC > 0.0 && fD > 0.0) && !(fA < 0.0 && fB < 0.0 && fC < 0.0 && fD < 0.0) )
							{
								curr_disp.x = 0.0;
								curr_disp.y = delta;
								current_positions[index+3]	= 0.0f;
							}
						}
					}
				}
				else if( curr_policy == 5.0 )
				{
					curr_disp.x += delta;
					current_positions[index+3]	= DEG2RAD*90.0f;
				}
				else if( curr_policy == 6.0 )
				{
					curr_disp.x += delta;
					curr_disp.y -= delta;
					current_positions[index+3]	= DEG2RAD*135.0f;

					if( curr_tileXY.y > 0.0 )
					{
						unsigned int test_tile = (unsigned int)((curr_tileXY.y - 1.0) * scene_width_in_tiles + curr_tileXY.x);
						if( policy[test_tile] == 8.0 )
						{
							float2 p1;
							p1.x						= curr_abs_pos.x;
							p1.y						= curr_abs_pos.y;
							float2 p2;
							p2.x						= p1.x + tile_width;
							p2.y						= p1.y - tile_height;
							float2 pA, pB, pC, pD;
							pA.x						= curr_tileXY.x * tile_width;
							pB.x						= pA.x + tile_width;
							pC.x						= pA.x;
							pD.x						= pB.x;
							pA.y						= curr_tileXY.y * tile_height;
							pB.y						= pA.y;
							pC.y						= (curr_tileXY.y - 1.0) * tile_height;
							pD.y						= pC.y;
							float q1					= p2.y-p1.y;
							float q2					= p1.x-p2.x;
							float q3					= p2.x*p1.y-p1.x*p2.y;
							float fA					= q1*pA.x+q2*pA.y+q3;
							float fB					= q1*pB.x+q2*pB.y+q3;
							float fC					= q1*pC.x+q2*pC.y+q3;
							float fD					= q1*pD.x+q2*pD.y+q3;
							if( !(fA > 0.0 && fB > 0.0 && fC > 0.0 && fD > 0.0) && !(fA < 0.0 && fB < 0.0 && fC < 0.0 && fD < 0.0) )
							{
								curr_disp.y = 0.0;
								curr_disp.x = delta;
								current_positions[index+3] = DEG2RAD*90.0f;
							}
						}
					}
					if( (curr_tileXY.x + 1.0) < scene_width_in_tiles )
					{
						unsigned int test_tile = (unsigned int)(curr_tileXY.y * scene_width_in_tiles + curr_tileXY.x + 1.0);
						if( policy[test_tile] == 8.0 )
						{
							float2 p1;
							p1.x						= curr_abs_pos.x;
							p1.y						= curr_abs_pos.y;
							float2 p2;
							p2.x						= p1.x + tile_width;
							p2.y						= p1.y - tile_height;
							float2 pA, pB, pC, pD;
							pA.x						= (curr_tileXY.x + 1.0) * tile_width;
							pB.x						= (curr_tileXY.x + 2.0) * tile_width;
							pC.x						= pA.x;
							pD.x						= pB.x;
							pA.y						= (curr_tileXY.y + 1.0) * tile_height;
							pB.y						= pA.y;
							pC.y						= curr_tileXY.y * tile_height;
							pD.y						= pC.y;
							float q1					= p2.y-p1.y;
							float q2					= p1.x-p2.x;
							float q3					= p2.x*p1.y-p1.x*p2.y;
							float fA					= q1*pA.x+q2*pA.y+q3;
							float fB					= q1*pB.x+q2*pB.y+q3;
							float fC					= q1*pC.x+q2*pC.y+q3;
							float fD					= q1*pD.x+q2*pD.y+q3;
							if( !(fA > 0.0 && fB > 0.0 && fC > 0.0 && fD > 0.0) && !(fA < 0.0 && fB < 0.0 && fC < 0.0 && fD < 0.0) )
							{
								curr_disp.x = 0.0;
								curr_disp.y = -delta;
								current_positions[index+3] = DEG2RAD*180.0f;
							}
						}
					}
				}
				else if( curr_policy == 7.0 )
				{
					curr_disp.y -= delta;
					current_positions[index+3]	= DEG2RAD*180.0f;
				}
				current_positions[index+0]	= curr_tile_center.x + orig_disp.x - hscene_dims.x + curr_disp.x;
				current_positions[index+1]	= (float)curr_tile;
				current_positions[index+2]	= curr_tile_center.y + orig_disp.y - hscene_dims.y + curr_disp.y;

				if( prev_tile < num_tiles )
				{
					density[prev_tile] -= 1.0f;
					density[curr_tile] += 1.0f;
				}
			}
		}
		else
		{
			curr_disp.x	= 0.0f;
			curr_disp.y	= 0.0f;
			float curr_angle = round( RAD2DEG * current_positions[index+3] );
			if( curr_angle == -135.0f )
			{
				curr_disp.x -= delta;
				curr_disp.y -= delta;
			}
			else if( curr_angle == -90.0f )
			{
				curr_disp.x -= delta;
			}
			else if( curr_angle == -45.0f )
			{
				curr_disp.x -= delta;
				curr_disp.y += delta;
			}
			else if( curr_angle == 0.0f )
			{
				curr_disp.y += delta;
			}
			else if( curr_angle == 45.0f )
			{
				curr_disp.x += delta;
				curr_disp.y += delta;
			}
			else if( curr_angle == 90.0f )
			{
				curr_disp.x += delta;
			}
			else if( curr_angle == 135.0f )
			{
				curr_disp.x += delta;
				curr_disp.y -= delta;
			}
			else if( curr_angle == 180.0f )
			{
				curr_disp.y -= delta;
			}
			current_positions[index+0]	= curr_pos.x + curr_disp.x;
			current_positions[index+2]	= curr_pos.y + curr_disp.y;
		}
	}
}
//
//=======================================================================================
//
extern "C" void launch_mdp_square_kernel(	float*			current_positions,
											float*			original_positions,
											float*			policy,
											float*			density,
											float			scene_width_in_tiles,
											float			scene_height_in_tiles,
											float			tile_width,
											float			tile_height,
											unsigned int	mesh_width,
											unsigned int	mesh_height,
											float			time,
											bool			policy_reset		)
{
    dim3 block( 4, 4, 1 );
    dim3 grid( mesh_width / block.x, mesh_height / block.y, 1 );
	mdp_square_kernel<<<grid, block>>>(	current_positions,
										original_positions,
										policy,
										density,
										scene_width_in_tiles,
										scene_height_in_tiles,
										tile_width,
										tile_height,
										time,
										policy_reset		);
}

#endif
//
//=======================================================================================

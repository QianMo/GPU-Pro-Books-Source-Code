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

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/random.h>
#include <thrust/inner_product.h>
#include <thrust/binary_search.h>
#include <thrust/adjacent_difference.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/tabulate.h>
#include <thrust/transform.h>
#include <thrust/transform_scan.h>
#include <thrust/reduce.h>
#include <iostream>
#include <iomanip>
#include <bitset>
#include <algorithm>
#include <math.h>
#include <sstream>
#include <time.h>
#include "ccThrustUtil.h"

#ifndef DEG2RAD
	#define DEG2RAD	0.01745329251994329576f
#endif  DEG2RAD

#ifndef RAD2DEG
	#define RAD2DEG	57.29577951308232087679f
#endif  RAD2DEG

#ifndef DIRECTIONS
	#define DIRECTIONS	8
#endif

#ifndef MAX_DRIFT
	#define MAX_DRIFT	2
#endif

#define SIGNF( X ) (X < -0.1f) ? -1.0f : ((X > 0.1f) ? 1.0f : 0.0f)
#define INBOUNDS( X, LIMIT ) X > -1 && X < LIMIT

thrust::minus<int>										minus_i;
thrust::multiplies<int>									mult_i;
thrust::minus<unsigned int>								minus_ui;
thrust::divides<unsigned int>							div_ui;
typedef thrust::device_vector<unsigned int>::iterator	UI_Iterator;
//
//=======================================================================================
//
unsigned int find_npot( unsigned int side )
{
	unsigned int npot = side;
	//http://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2
	npot--;
	npot |= npot >> 1;
	npot |= npot >> 2;
	npot |= npot >> 4;
	npot |= npot >> 8;
	npot |= npot >> 16;
	npot++;
	return npot;
}
//
//=======================================================================================
//
__host__ __device__ unsigned int hash( unsigned int x )
{
	x = (x+0x7ed55d16) + (x<<12);
	x = (x^0xc761c23c) ^ (x>>19);
	x = (x+0x165667b1) + (x<<5);
	x = (x+0xd3a2646c) ^ (x<<9);
	x = (x+0xfd7046c5) + (x<<3);
	x = (x^0xb55a4f09) ^ (x>>16);
	return x;
}
//
//=======================================================================================
//
#include "ccMDP_LCA_advance.h"
//#include "ccMDP_LCA_virtual.h"
#include "ccMDP_LCA_fix.h"
//
//=======================================================================================
//
struct populate_n_agents_rnd
{
	unsigned int	scene_width_in_cells;
	unsigned int*	raw_occ_ptr;
	float			cell_width;
	float			f_directions;
	float			f_deg_inc;
	int				swic;

	populate_n_agents_rnd(	float			_cell_width,
							unsigned int	_swic,
							unsigned int*	_rop		) : cell_width			( _cell_width	),
															scene_width_in_cells( _swic			),
															raw_occ_ptr			( _rop			)
	{
		f_directions	= (float)DIRECTIONS;
		f_deg_inc		= 360.0f / (float)DIRECTIONS;
		swic			= (int)scene_width_in_cells;
	}

	template <typename Tuple>								// 0         1         2                3                4         5             6             7    8
	inline __host__ __device__ void operator()(	Tuple t	)	// NCELL_ID, AGENT_ID, AGENT_GOAL_CELL, AGENT_VIRT_CELL, AGENT_XY, LOW_AGENT_ID, UPP_AGENT_ID, UID, ISEED
	{
		unsigned int cell_id			= thrust::get<0>(t);
		unsigned int uid				= thrust::get<7>(t);
		int seed						= thrust::get<8>(t);
		unsigned int total_cells		= scene_width_in_cells * scene_width_in_cells;
		thrust::default_random_engine rng( hash( seed ) );
		thrust::random::uniform_real_distribution<float> rDist( 0.0f, 1.0f );
		//thrust::random::uniform_int_distribution<unsigned int> uiDist( 0, total_cells );
		thrust::random::uniform_int_distribution<unsigned int> uiDist( 0, scene_width_in_cells - 1 );

		unsigned int low_aid			= thrust::get<5>(t);
		unsigned int upp_aid			= thrust::get<6>(t);
		unsigned int row				= cell_id / scene_width_in_cells;
		unsigned int col				= cell_id % scene_width_in_cells;
		unsigned int offset				= uid % DIRECTIONS;

		if( low_aid + offset < upp_aid )
		{
			//float			hWidth		= (float)scene_width_in_cells * cell_width / 2.0f;
			float			t_x			= cell_width * (float)col;
			float			t_y			= cell_width * (float)row;
			//float			a_x			= (t_x + cell_width * rDist( rng )) - hWidth;
			//float			a_y			= (t_y + cell_width * rDist( rng )) - hWidth;
			float			a_x			= cell_width * rDist( rng );
			float			a_y			= cell_width * rDist( rng );

			//unsigned int	gi			= uiDist( rng );
			//int			gx			= (int)(gi % scene_width_in_cells);
			//int			gy			= (int)(gi / scene_width_in_cells);
			int				gx			= (int)scene_width_in_cells - 1;
			int				gy			= (int)uiDist( rng );
			unsigned int	gi			= (unsigned int)(gy * (int)scene_width_in_cells + gx);
			int				diffX		= (gx - (int)t_x);
			int				ncx			= diffX;
			if( ncx != 0 )	ncx			= diffX / abs( diffX );
			int				diffY		= (gy - (int)t_y);
			int				ncy			= diffY;
			if( ncy != 0 )	ncy			= diffY / abs( diffY );
			unsigned int	ntx			= (unsigned int)(t_x + (float)ncx);
			unsigned int	nty			= (unsigned int)(t_y + (float)ncy);
			unsigned int	ni			= nty * scene_width_in_cells + ntx;
			
			thrust::get<1>(t)			= low_aid + offset;			//AGENT_ID
			thrust::get<2>(t)			= gi;						//GOAL
			thrust::get<3>(t)			= ni;						//VIRTUAL
			thrust::get<4>(t)			= make_float2( a_x, a_y );
		}
		else
		{
			thrust::get<1>(t)			= total_cells * DIRECTIONS;	//AGENT_ID
			thrust::get<2>(t)			= total_cells;				//GOAL
			thrust::get<3>(t)			= total_cells;				//VIRTUAL
			thrust::get<4>(t)			= make_float2( 0.0f, 0.0f );
		}
	}
};
//
//=======================================================================================
//
struct populate_n_agents
{
	unsigned int	scene_width_in_lca_cells;
	unsigned int	scene_width_in_mdp_cells;
	unsigned int	lca_mdp_width_ratio;
	unsigned int	lca_cells_per_mdp_cell;
	unsigned int	total_lca_cells;
	unsigned int	tlcd;
	unsigned int	lca_cell_id;
	unsigned int	mdp_channels;
	unsigned int	lca_row;
	unsigned int	lca_col;
	unsigned int	divisor;
	unsigned int	channel;

	unsigned int*	raw_occ_ptr;
	unsigned int*	raw_vc1_ptr;
	unsigned int*	raw_vc2_ptr;
	unsigned int*	raw_vc3_ptr;
	unsigned int*	raw_chn_ptr;

	float			cell_width;
	float			f_directions;
	float			f_deg_inc;
	float			fhsw;
	int				swic;
	float*			raw_pos_ptr;
	float*			raw_lca_ptr;
	int				gx;
	int				gz;
	int				swimc;

	populate_n_agents(	unsigned int	_divisor,
						float			_cell_width,
						unsigned int	_swilc,
						unsigned int	_swimc,
						unsigned int	_mdpc,
						unsigned int*	_rop,
						float*			_rpp,
						float*			_rlp,
						unsigned int*	_rvc1,
						unsigned int*	_rvc2,
						unsigned int*	_rvc3,
						unsigned int*	_rchp		) : divisor					(	_divisor	),
														cell_width				(	_cell_width	),
														scene_width_in_lca_cells(	_swilc		),
														scene_width_in_mdp_cells(	_swimc		),
														mdp_channels			(	_mdpc		),
														raw_occ_ptr				(	_rop		),
														raw_pos_ptr				(	_rpp		),
														raw_lca_ptr				(	_rlp		),
														raw_vc1_ptr				(	_rvc1		),
														raw_vc2_ptr				(	_rvc2		),
														raw_vc3_ptr				(	_rvc3		),
														raw_chn_ptr				(	_rchp		)
	{
		lca_mdp_width_ratio				= scene_width_in_lca_cells / scene_width_in_mdp_cells;
		lca_cells_per_mdp_cell			= lca_mdp_width_ratio * lca_mdp_width_ratio;
		f_directions					= (float)DIRECTIONS;
		f_deg_inc						= 360.0f / (float)DIRECTIONS;
		swic							= (int)scene_width_in_lca_cells;
		swimc							= (int)scene_width_in_mdp_cells;
		fhsw							= ((float)scene_width_in_lca_cells * cell_width) / 2.0f;
		gx								= 0;
		gz								= 0;
		lca_row							= 0;
		lca_col							= 0;
		channel							= 0;
		total_lca_cells					= scene_width_in_lca_cells * scene_width_in_lca_cells;
		tlcd							= total_lca_cells * DIRECTIONS;
	}

	template <typename Tuple>								// 0         1         2                3         4             5             6
	inline __host__ __device__ void operator()(	Tuple t	)	// NCELL_ID, AGENT_ID, AGENT_GOAL_CELL, AGENT_XY, LOW_AGENT_ID, UPP_AGENT_ID, UID
	{
		lca_cell_id						= thrust::get<0>(t);
		unsigned int	uid				= thrust::get<6>(t);

		unsigned int	low_aid			= thrust::get<4>(t);
		unsigned int	upp_aid			= thrust::get<5>(t);
		unsigned int	offset			= uid % DIRECTIONS;
		lca_row							= lca_cell_id / scene_width_in_lca_cells;
		lca_col							= lca_cell_id % scene_width_in_lca_cells;

		unsigned int	index			= low_aid + offset;
		channel							= index / divisor;
		unsigned int	index4			= index * 4;
		float			agent_pos_x		= raw_pos_ptr[index4+0]  + fhsw;
		float			agent_pos_z		= raw_pos_ptr[index4+2]  + fhsw;

		if( low_aid + offset < upp_aid )
		{
			float			t_x			= cell_width * (float)lca_col;
			float			t_z			= cell_width * (float)lca_row;
			float			a_x			= fmod( agent_pos_x, cell_width );
			float			a_z			= fmod( agent_pos_z, cell_width );
			
			nearest_exit();

			unsigned int	gi			= (unsigned int)(gz * swic + gx);
			if( gi == lca_cell_id )
			{
				nearest_exit_from_goal();
				gi						= (unsigned int)(gz * swic + gx);
			}
			int				diffX		= (gx - (int)t_x);
			int				ncx			= diffX;
			if( ncx != 0 )	ncx			= diffX / abs( diffX );
			int				diffZ		= (gz - (int)t_z);
			int				ncz			= diffZ;
			if( ncz != 0 )	ncz			= diffZ / abs( diffZ );
			unsigned int	ntx			= (unsigned int)((int)lca_col + ncx);
			unsigned int	ntz			= (unsigned int)((int)lca_row + ncz);
			unsigned int	ni			= ntz * scene_width_in_lca_cells + ntx;
			
			unsigned int							niA	= tlcd;
			unsigned int							niB	= tlcd;
			if( abs(ncx) > 0 && abs(ncz) > 0 )
			{
				if( INBOUNDS( (int)ntx, swic ) )	niA	= lca_row * scene_width_in_lca_cells + ntx;
				if( INBOUNDS( (int)ntz, swic ) )	niB	= ntz * scene_width_in_lca_cells + lca_col;
			}
			else if( abs(ncx) > 0 && INBOUNDS( (int)ntx, swic ) )
			{
				if( (int)lca_row > 0 )				niA	= ((int)lca_row - 1) * scene_width_in_lca_cells + ntx;
				if( ((int)lca_row + 1) < swic )		niB	= ((int)lca_row + 1) * scene_width_in_lca_cells + ntx;
			}
			else if( INBOUNDS( (int)ntz, swic ) )
			{
				if( (int)lca_col > 0 )				niA	= ntz * scene_width_in_lca_cells + ((int)lca_col - 1);
				if( ((int)lca_col + 1) < swic )		niB	= ntz * scene_width_in_lca_cells + ((int)lca_col + 1);
			}
			
			thrust::get<1>(t)			= index;						//AGENT_ID
			thrust::get<2>(t)			= gi;							//GOAL
			thrust::get<3>(t)			= make_float2( a_x, a_z );		//AGENT_XY
			raw_vc1_ptr[uid]			= ni;							//VIRTUAL1
			raw_vc2_ptr[uid]			= niA;							//VIRTUAL2
			raw_vc3_ptr[uid]			= niB;							//VIRTUAL3
			raw_chn_ptr[uid]			= channel;						//MDP_CHANNEL
		}
		else
		{
			thrust::get<1>(t)			= total_lca_cells * DIRECTIONS;	//AGENT_ID
			thrust::get<2>(t)			= total_lca_cells;				//GOAL
			thrust::get<3>(t)			= make_float2( 0.0f, 0.0f );	//AGENT_XY
			raw_vc1_ptr[uid]			= total_lca_cells * DIRECTIONS;	//VIRTUAL1
			raw_vc2_ptr[uid]			= total_lca_cells * DIRECTIONS;	//VIRTUAL2
			raw_vc3_ptr[uid]			= total_lca_cells * DIRECTIONS;	//VIRTUAL3
			raw_chn_ptr[uid]			= mdp_channels;					//MDP_CHANNEL
		}
	}

	inline __host__ __device__ void nearest_exit_from_goal( void )
	{
		float	test_radians				= DEG2RAD * (raw_lca_ptr[channel * total_lca_cells + lca_cell_id] * f_deg_inc + 135.0f);
		int		test_lca_col				= (int)lca_col + ((int)(SIGNF(  cosf( test_radians ) )));
		int		test_lca_row				= (int)lca_row + ((int)(SIGNF( -sinf( test_radians ) )));
		unsigned int mdp_row				= (unsigned int)test_lca_row / lca_mdp_width_ratio;
		unsigned int mdp_col				= (unsigned int)test_lca_col / lca_mdp_width_ratio;
		gx									= -1;
		gz									= -1;
		
		if( INBOUNDS( (int)mdp_col, swimc ) && INBOUNDS( (int)mdp_row, swimc ) )
		{

			unsigned int goal_row			= 0;
			unsigned int goal_col			= 0;

			unsigned int lca_idx			= mdp_row * lca_mdp_width_ratio * scene_width_in_lca_cells + 
											  mdp_col * lca_mdp_width_ratio;
			
			float dist2						= 0.0f;
			float minDist2					= 100000.0f;
			float maxDist2					= 0.0f;
			float diffX						= 0.0f;
			float diffZ						= 0.0f;

			for( unsigned int startrow		= 0; startrow < lca_mdp_width_ratio; startrow++ )
			{
				unsigned int lca_offset		= lca_idx + startrow * scene_width_in_lca_cells;
				for( unsigned int i			= lca_offset; i < lca_offset + lca_mdp_width_ratio; i++ )
				{
					if( raw_lca_ptr[channel * total_lca_cells + i] < (float)DIRECTIONS )
					{
						goal_row			= i / scene_width_in_lca_cells;
						goal_col			= i % scene_width_in_lca_cells;
						diffX				= (float)goal_col - (float)test_lca_col;
						diffZ				= (float)goal_row - (float)test_lca_row;
						dist2				= diffX * diffX + diffZ * diffZ;
						if( dist2 < minDist2 )
						{
							gx				= (int)goal_col;
							gz				= (int)goal_row;
							minDist2		= dist2;
						}
					}
					else if( raw_lca_ptr[channel * total_lca_cells + i] == 9.0f )
					{
						goal_row			= i / scene_width_in_lca_cells;
						goal_col			= i % scene_width_in_lca_cells;
						if( goal_row != lca_row && goal_col != lca_col )
						{
							diffX				= (float)goal_col - (float)lca_col;
							diffZ				= (float)goal_row - (float)lca_row;
							dist2				= diffX * diffX + diffZ * diffZ;
							if( dist2 > maxDist2 )
							{
								gx				= (int)goal_col;
								gz				= (int)goal_row;
								maxDist2		= dist2;
							}
						}
					}
				}
			}
		}
	}

	inline __host__ __device__ void nearest_exit( void )
	{
		unsigned int mdp_row			= lca_row / lca_mdp_width_ratio;
		unsigned int mdp_col			= lca_col / lca_mdp_width_ratio;
		
		unsigned int goal_row			= 0;
		unsigned int goal_col			= 0;

		unsigned int lca_idx			= mdp_row * lca_mdp_width_ratio * scene_width_in_lca_cells + 
										  mdp_col * lca_mdp_width_ratio;
		
		float dist2						= 0.0f;
		float minDist2					= 100000.0f;
		float diffX						= 0.0f;
		float diffZ						= 0.0f;

		for( unsigned int startrow		= 0; startrow < lca_mdp_width_ratio; startrow++ )
		{
			unsigned int lca_offset		= lca_idx + startrow * scene_width_in_lca_cells;
			for( unsigned int i			= lca_offset; i < lca_offset + lca_mdp_width_ratio; i++ )
			{
				if( raw_lca_ptr[channel * total_lca_cells + i] < (float)DIRECTIONS )
				{
					goal_row			= i / scene_width_in_lca_cells;
					goal_col			= i % scene_width_in_lca_cells;
					diffX				= (float)goal_col - (float)lca_col;
					diffZ				= (float)goal_row - (float)lca_row;
					dist2				= diffX * diffX + diffZ * diffZ;
					if( dist2 < minDist2 )
					{
						gx				= (int)goal_col;
						gz				= (int)goal_row;
						minDist2		= dist2;
					}
				}
			}
		}
	}
};
//
//=======================================================================================
//
struct fill_cell_id_rnd
{
	float			scale;
	float			cell_width;
	unsigned int	scene_width_in_cells;

	fill_cell_id_rnd(	float			_scale,
						float			_cell_width,
						unsigned int	_swic	) : scale					(	_scale		),
													cell_width				(	_cell_width	),
													scene_width_in_cells	(	_swic		)
	{

	}

	template <typename Tuple>								// 0		1		  2			3		  4
	inline __host__ __device__ void operator()( Tuple t )	// CELL_ID, FRANDOM1, FRANDOM2, FRANDOM3, FRANDOM4
	{
		float x = scale * thrust::get<1>(t);
		float y = scale * thrust::get<2>(t);
		float s = thrust::get<3>(t);
		if( s < 0.5 )
		{
			x = -x;
		}
		s = thrust::get<4>(t);
		if( s < 0.5 )
		{
			y = -y;
		}
		x += scale;
		y += scale;

		unsigned int tx = (unsigned int)(x / cell_width);
		unsigned int ty = (unsigned int)(y / cell_width);
		unsigned int tile = ty * scene_width_in_cells + tx;
		thrust::get<0>(t) = tile;
	}
};
//
//=======================================================================================
//
struct fill_cell_id
{
	float			scale;
	float			cell_width;
	unsigned int	scene_width_in_cells;
	float*			raw_pos_ptr;
	float			half_scene_width;

	fill_cell_id(	float			_scale,
					float			_cell_width,
					unsigned int	_swic,
					float*			_rpp)		:	scale					(	_scale		),
													cell_width				(	_cell_width	),
													scene_width_in_cells	(	_swic		),
													raw_pos_ptr				(	_rpp		)
	{
		half_scene_width =  ((float)scene_width_in_cells * cell_width) / 2.0f;
	}

	template <typename Tuple>								// 0		1
	inline __host__ __device__ void operator()( Tuple t )	// CELL_ID,	UID
	{
		unsigned int index = thrust::get<1>(t) * 4;
		float x = raw_pos_ptr[index+0] + half_scene_width;
		float z = raw_pos_ptr[index+2] + half_scene_width;

		unsigned int tx = (unsigned int)(x / cell_width);
		unsigned int tz = (unsigned int)(z / cell_width);
		unsigned int tile = tz * scene_width_in_cells + tx;
		thrust::get<0>(t) = tile;
	}
};
//
//=======================================================================================
//
struct set_density
{
	unsigned int	scene_width_in_lca_cells;
	unsigned int	scene_width_in_mdp_cells;
	unsigned int	lca_mdp_width_ratio;
	unsigned int	lca_cells_per_mdp_cell;
	float*			raw_mdp_lca_density_ptr;

	set_density(	unsigned int	_swilc,
					unsigned int	_swimc,
					float*			_rmldp	)		:	scene_width_in_lca_cells(	_swilc		),
														scene_width_in_mdp_cells(	_swimc		),
														raw_mdp_lca_density_ptr	(	_rmldp		)
	{
		lca_mdp_width_ratio = scene_width_in_lca_cells / scene_width_in_mdp_cells;
		lca_cells_per_mdp_cell = lca_mdp_width_ratio * lca_mdp_width_ratio;
	}

	template <typename Tuple>								// 0			1
	inline __host__ __device__ void operator()( Tuple t )	// LCA_CELL_ID,	LCA_DENSITY
	{
		unsigned int lca_index	= thrust::get<0>( t );
		unsigned int lca_occ	= thrust::get<1>( t );
		
		unsigned int lca_row	= lca_index / scene_width_in_lca_cells;
		unsigned int lca_col	= lca_index % scene_width_in_lca_cells;
		
		unsigned int mdp_row	= lca_row / scene_width_in_mdp_cells;
		unsigned int mdp_col	= lca_col / scene_width_in_mdp_cells;
		
		unsigned int mdp_idx	= (mdp_row * scene_width_in_mdp_cells + mdp_col) * lca_cells_per_mdp_cell;
		unsigned int mdp_inr	= lca_row % lca_mdp_width_ratio;
		unsigned int mdp_inc	= lca_col % lca_mdp_width_ratio;
		unsigned int mdp_iix	= mdp_inr * lca_mdp_width_ratio + mdp_inc;
		
		raw_mdp_lca_density_ptr[ mdp_idx + mdp_iix ] = (float)lca_occ;
	}
};
//
//=======================================================================================
//
struct gen_randomf
{
	unsigned int skip;

	gen_randomf( unsigned int _skip ) : skip( _skip )
	{

	}

	template <typename Tuple>						// 0        1
	__host__ __device__ void operator()( Tuple t )	// FRESULT, ISEED
	{
		thrust::default_random_engine rng( hash( thrust::get<1>(t) ) );
		rng.discard( skip );
		thrust::random::uniform_real_distribution<float> rDist;
		thrust::get<0>(t) = rDist( rng );
	}
};
//
//=======================================================================================
//
struct ui_minus_one
{
	template <typename Tuple>
	__host__ __device__ void operator()( Tuple t )
	{
		if( (unsigned int)thrust::get<0>(t) > 0 )
		{
			thrust::get<0>(t) = thrust::get<0>(t) - 1;
		}
	}
};
//
//=======================================================================================
//
void generate_cells_and_agents(	thrust::device_vector<int>&				seeds,
								thrust::device_vector<unsigned int>&	cells_ids,
								thrust::device_vector<unsigned int>&	d_cells_ids,
								thrust::device_vector<unsigned int>&	cells_occ,
								thrust::device_vector<unsigned int>&	agents_ids,
								thrust::device_vector<unsigned int>&	agents_dids,
								thrust::device_vector<unsigned int>&	agents_goal_cell,
								thrust::device_vector<unsigned int>&	agents_virt_cell,
								thrust::device_vector<float2>&			agents_xy,
								float									scale,
								float									cell_width,
								unsigned int							scene_width_in_cells,
								unsigned int							num_agents			)
{
	unsigned int total_cells = scene_width_in_cells * scene_width_in_cells;

	thrust::sequence( cells_ids.begin(), cells_ids.end() );
	repeated_range<UI_Iterator>	n_cells_ids	( cells_ids.begin(), cells_ids.end(), DIRECTIONS );
	thrust::copy( n_cells_ids.begin(), n_cells_ids.end(), d_cells_ids.begin() );

	thrust::device_vector<unsigned int> low_aids( total_cells );
	thrust::device_vector<unsigned int> upp_aids( total_cells );
	thrust::device_vector<unsigned int> agent_cell_ids( num_agents );

	thrust::counting_iterator<unsigned int> aids_begin( 0 );
	thrust::counting_iterator<unsigned int> aids_end = aids_begin + num_agents;
	thrust::device_vector<float> r0( num_agents );
	thrust::device_vector<float> r1( num_agents );
	thrust::device_vector<float> r2( num_agents );
	thrust::device_vector<float> r3( num_agents );
	thrust::for_each(	thrust::make_zip_iterator( thrust::make_tuple( r0.begin(), seeds.begin() ) ), 
						thrust::make_zip_iterator( thrust::make_tuple( r0.end(), seeds.begin() + r0.size() ) ), 
						gen_randomf( 0 * num_agents ) );
	thrust::for_each(	thrust::make_zip_iterator( thrust::make_tuple( r1.begin(), seeds.begin() ) ), 
						thrust::make_zip_iterator( thrust::make_tuple( r1.end(), seeds.begin() + r1.size() ) ), 
						gen_randomf( 1 * num_agents ) );
	thrust::for_each(	thrust::make_zip_iterator( thrust::make_tuple( r2.begin(), seeds.begin() ) ), 
						thrust::make_zip_iterator( thrust::make_tuple( r2.end(), seeds.begin() + r2.size() ) ), 
						gen_randomf( 2 * num_agents ) );
	thrust::for_each(	thrust::make_zip_iterator( thrust::make_tuple( r3.begin(), seeds.begin() ) ), 
						thrust::make_zip_iterator( thrust::make_tuple( r3.end(), seeds.begin() + r3.size() ) ), 
						gen_randomf( 3 * num_agents ) );

	thrust::for_each(	
		thrust::make_zip_iterator(
			thrust::make_tuple(
				agent_cell_ids.begin(),
				r0.begin(),
				r1.begin(),
				r2.begin(),
				r3.begin()
			)
		),
		thrust::make_zip_iterator(
			thrust::make_tuple(
				agent_cell_ids.end(),
				r0.end(),
				r1.end(),
				r2.end(),
				r3.end()
			)
		),
		fill_cell_id_rnd(	scale, 
							cell_width, 
							scene_width_in_cells )
	);
	thrust::device_vector<unsigned int> complement( total_cells );
	thrust::sequence( complement.begin(), complement.end() );
	thrust::device_vector<unsigned int> agent_cell_ids2( num_agents + total_cells );
	thrust::copy( agent_cell_ids.begin(), agent_cell_ids.end(), agent_cell_ids2.begin() );
	thrust::copy( complement.begin(), complement.end(), agent_cell_ids2.begin() + num_agents );
	dense_histogram( agent_cell_ids2, cells_occ );
	thrust::for_each(
		thrust::make_zip_iterator(
			thrust::make_tuple(
				cells_occ.begin()
			)
		), 
		thrust::make_zip_iterator(
			thrust::make_tuple(
				cells_occ.end()
			)
		),
		ui_minus_one()
	);
	thrust::inclusive_scan( cells_occ.begin(), cells_occ.end(), upp_aids.begin() );
	thrust::transform( upp_aids.begin(), upp_aids.end(), cells_occ.begin(), low_aids.begin(), minus_ui );
	
	// DCELL_ID, AGENT_ID, AGENT_GOAL_CELL, AGENT_VIRT_CELL, AGENT_X, AGENT_Y, NLOW_AGENT_ID, NUPP_AGENT_ID, UID
	repeated_range<UI_Iterator> n_low_aids( low_aids.begin(), low_aids.end(), DIRECTIONS );
	repeated_range<UI_Iterator> n_upp_aids( upp_aids.begin(), low_aids.end(), DIRECTIONS );
	thrust::sequence( agents_dids.begin(), agents_dids.end() );
	unsigned int* raw_occ_ptr = thrust::raw_pointer_cast( cells_occ.data() );

	thrust::for_each(
		thrust::make_zip_iterator( 
			thrust::make_tuple(
				d_cells_ids.begin(),
				agents_ids.begin(),
				agents_goal_cell.begin(),
				agents_virt_cell.begin(),
				agents_xy.begin(),
				n_low_aids.begin(),
				n_upp_aids.begin(),
				agents_dids.begin(),
				seeds.begin()
			)
		),
		thrust::make_zip_iterator( 
			thrust::make_tuple(
				d_cells_ids.end(),
				agents_ids.end(),
				agents_goal_cell.end(),
				agents_virt_cell.end(),
				agents_xy.end(),
				n_low_aids.end(),
				n_upp_aids.end(),
				agents_dids.end(),
				seeds.end()
			)
		),
		populate_n_agents_rnd( cell_width, scene_width_in_cells, raw_occ_ptr )
	);
}
//
//=======================================================================================
//
void init_cells_and_agents	(	thrust::device_vector<float>&			agents_pos,
								thrust::device_vector<float>&			lca,
								thrust::device_vector<int>&				seeds,
								thrust::device_vector<unsigned int>&	cells_ids,
								thrust::device_vector<unsigned int>&	d_cells_ids,
								thrust::device_vector<unsigned int>&	cells_occ,
								thrust::device_vector<unsigned int>&	agents_ids,
								thrust::device_vector<unsigned int>&	agents_dids,
								thrust::device_vector<unsigned int>&	agents_goal_cell,
								thrust::device_vector<unsigned int>&	agents_virt_cell1,
								thrust::device_vector<unsigned int>&	agents_virt_cell2,
								thrust::device_vector<unsigned int>&	agents_virt_cell3,
								thrust::device_vector<unsigned int>&	agents_channel,
								thrust::device_vector<float>&			displace_param,
								thrust::device_vector<float2>&			agents_xy,
								float									scale,
								float									cell_width,
								unsigned int							scene_width_in_lca_cells,
								unsigned int							scene_width_in_mdp_cells,
								unsigned int							num_agents,
								unsigned int							mdp_channels			)
{
	unsigned int total_lca_cells = scene_width_in_lca_cells * scene_width_in_lca_cells;

	//thrust::device_vector<unsigned int> div( agents_channel.size() );
	thrust::fill( agents_channel.begin(), agents_channel.end(), mdp_channels );
	unsigned int divisor = (num_agents / mdp_channels) + (num_agents % mdp_channels);
	//thrust::fill( div.begin(), div.end(), divisor );
	//thrust::sequence( agents_channel.begin(), agents_channel.end() );
	//thrust::transform( agents_channel.begin(), agents_channel.end(), div.begin(), agents_channel.begin(), div_ui );

	thrust::sequence( cells_ids.begin(), cells_ids.end() );
	repeated_range<UI_Iterator>	n_cells_ids	( cells_ids.begin(), cells_ids.end(), DIRECTIONS );
	thrust::copy( n_cells_ids.begin(), n_cells_ids.end(), d_cells_ids.begin() );
	thrust::fill( displace_param.begin(), displace_param.end(), 0.0f );

	thrust::device_vector<unsigned int> low_aids( total_lca_cells );
	thrust::device_vector<unsigned int> upp_aids( total_lca_cells );
	thrust::device_vector<unsigned int> agent_cell_ids( num_agents );

	thrust::counting_iterator<unsigned int> aids_begin( 0 );
	thrust::counting_iterator<unsigned int> aids_end = aids_begin + num_agents;

	float* raw_pos_ptr	= thrust::raw_pointer_cast( agents_pos.data()	);

	thrust::for_each(	
		thrust::make_zip_iterator(
			thrust::make_tuple(
				agent_cell_ids.begin(),
				cells_ids.begin()
			)
		),
		thrust::make_zip_iterator(
			thrust::make_tuple(
				agent_cell_ids.end(),
				cells_ids.end()
			)
		),
		fill_cell_id(	scale,
						cell_width, 
						scene_width_in_lca_cells,
						raw_pos_ptr				)
	);
	thrust::device_vector<unsigned int> complement( total_lca_cells );
	thrust::sequence( complement.begin(), complement.end() );
	thrust::device_vector<unsigned int> agent_cell_ids2( num_agents + total_lca_cells );
	thrust::copy( agent_cell_ids.begin(), agent_cell_ids.end(), agent_cell_ids2.begin() );
	thrust::copy( complement.begin(), complement.end(), agent_cell_ids2.begin() + num_agents );
	dense_histogram( agent_cell_ids2, cells_occ );
	thrust::for_each(
		thrust::make_zip_iterator(
			thrust::make_tuple(
				cells_occ.begin()
			)
		), 
		thrust::make_zip_iterator(
			thrust::make_tuple(
				cells_occ.end()
			)
		),
		ui_minus_one()
	);
	thrust::inclusive_scan( cells_occ.begin(), cells_occ.end(), upp_aids.begin() );
	thrust::transform( upp_aids.begin(), upp_aids.end(), cells_occ.begin(), low_aids.begin(), minus_ui );
	
	// DCELL_ID, AGENT_ID, AGENT_GOAL_CELL, AGENT_X, AGENT_Y, NLOW_AGENT_ID, NUPP_AGENT_ID, UID
	repeated_range<UI_Iterator> n_low_aids( low_aids.begin(), low_aids.end(), DIRECTIONS );
	repeated_range<UI_Iterator> n_upp_aids( upp_aids.begin(), low_aids.end(), DIRECTIONS );
	thrust::sequence( agents_dids.begin(), agents_dids.end() );
	unsigned int*	raw_occ_ptr	= thrust::raw_pointer_cast( cells_occ.data()			);
	unsigned int*	raw_vc1_ptr	= thrust::raw_pointer_cast( agents_virt_cell1.data()	);
	unsigned int*	raw_vc2_ptr	= thrust::raw_pointer_cast( agents_virt_cell2.data()	);
	unsigned int*	raw_vc3_ptr	= thrust::raw_pointer_cast( agents_virt_cell3.data()	);
	unsigned int*	raw_chn_ptr = thrust::raw_pointer_cast( agents_channel.data()		);
	float*			raw_lca_ptr	= thrust::raw_pointer_cast( lca.data()					);

	thrust::for_each(
		thrust::make_zip_iterator( 
			thrust::make_tuple(
				d_cells_ids.begin(),
				agents_ids.begin(),
				agents_goal_cell.begin(),
				agents_xy.begin(),
				n_low_aids.begin(),
				n_upp_aids.begin(),
				agents_dids.begin()
			)
		),
		thrust::make_zip_iterator( 
			thrust::make_tuple(
				d_cells_ids.end(),
				agents_ids.end(),
				agents_goal_cell.end(),
				agents_xy.end(),
				n_low_aids.end(),
				n_upp_aids.end(),
				agents_dids.end()
			)
		),
		populate_n_agents(	divisor,
							cell_width,
							scene_width_in_lca_cells,
							scene_width_in_mdp_cells,
							mdp_channels,
							raw_occ_ptr,
							raw_pos_ptr,
							raw_lca_ptr,
							raw_vc1_ptr,
							raw_vc2_ptr,
							raw_vc3_ptr,
							raw_chn_ptr				)
	);
}
//
//=======================================================================================
//
void update_mdp_density(	thrust::device_vector<float>&			mdp_lca_density,
							thrust::device_vector<float>&			mdp_density,
							thrust::device_vector<unsigned int>&	lca_cells_ids,
							thrust::device_vector<unsigned int>&	lca_occupancy,
							unsigned int							lca_width,
							unsigned int							mdp_width		)
{
	float* raw_mdp_lca_density_ptr = thrust::raw_pointer_cast( mdp_lca_density.data() );
	unsigned int lca_mdp_width_ratio = lca_width / mdp_width;
	unsigned int lca_cells_per_mdp_cell = lca_mdp_width_ratio * lca_mdp_width_ratio;
	thrust::device_vector<unsigned int> cells_ids( mdp_density.size() );
	thrust::sequence( cells_ids.begin(), cells_ids.end() );
	repeated_range<UI_Iterator>	n_cells_ids( cells_ids.begin(), cells_ids.end(), lca_cells_per_mdp_cell );

	thrust::for_each(
		thrust::make_zip_iterator( 
			thrust::make_tuple(
				lca_cells_ids.begin(),
				lca_occupancy.begin()
			)
		),
		thrust::make_zip_iterator( 
			thrust::make_tuple(
				lca_cells_ids.end(),
				lca_occupancy.end()
			)
		),
		set_density( lca_width, mdp_width, raw_mdp_lca_density_ptr )
	);

	thrust::reduce_by_key(	n_cells_ids.begin(),
							n_cells_ids.end(),
							mdp_lca_density.begin(),
							cells_ids.begin(),
							mdp_density.begin()		);
}
//
//=======================================================================================
//
unsigned int count_collisions( thrust::device_vector<unsigned int>& occupancy )
{
	unsigned int result = 0;
	thrust::device_vector<unsigned int> occ_cpy = occupancy;
	thrust::for_each(
		thrust::make_zip_iterator(
			thrust::make_tuple(
				occ_cpy.begin()
			)
		), 
		thrust::make_zip_iterator(
			thrust::make_tuple(
				occ_cpy.end()
			)
		),
		ui_minus_one()
	);
	result = thrust::reduce( occ_cpy.begin(), occ_cpy.end() );
	return result;
}
//
//=======================================================================================

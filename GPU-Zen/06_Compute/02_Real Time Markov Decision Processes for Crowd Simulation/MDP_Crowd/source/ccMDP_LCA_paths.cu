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
#ifndef __MDP_LCA_PATHS_KERNEL_H_
#define __MDP_LCA_PATHS_KERNEL_H_

//#define __DEBUG
//#define __PRINT_AGENTS
//#define __PRINT_OCC
//#define __PRINT_FUTURE_OCC
//#define __DO_PAUSE

#pragma once
#include "ccMDP_LCA.h"
//
//=======================================================================================
//
float			HALF_SCENE_WIDTH_IN_LCA_CELLS;
float			LCA_TILE_WIDTH;
unsigned int	NUM_AGENTS;
unsigned int	MDP_CHANNELS;
unsigned int	SCENE_WIDTH_IN_LCA_CELLS;
unsigned int	SCENE_WIDTH_IN_MDP_CELLS;
unsigned int	TOTAL_LCA_CELLS;
unsigned int	TOTAL_MDP_CELLS;
unsigned int	COUNTER;
unsigned int	COLLISIONS;

float			agents_time;
float			mem_alloc_time;
float			total_time;
float			fix_coll_time;			//Racing-QW
float			virtual_advance_time;
float			advance_time;			//Scatter-Gather
float			temp_time;

cudaEvent_t		start;
cudaEvent_t		stop;

thrust::device_vector<unsigned int> cells_ids;
thrust::device_vector<unsigned int> d_cells_ids;
thrust::device_vector<unsigned int> cells_row;
thrust::device_vector<unsigned int> cells_col;
thrust::device_vector<unsigned int> cells_occ;
thrust::device_vector<unsigned int> agents_channel;
thrust::device_vector<unsigned int> agents_channel_copy;
thrust::device_vector<unsigned int> agents_flags;					//0=HOLD, 1=GO1, 2=GO2, 3=GO3
thrust::device_vector<unsigned int> agents_flags_copy;
thrust::device_vector<int>			agents_drift_counter;
thrust::device_vector<int>			agents_drift_counter_copy;
thrust::device_vector<unsigned int>	agents_ids;
thrust::device_vector<unsigned int>	agents_dids;
thrust::device_vector<unsigned int>	agents_goal_cell;

thrust::device_vector<unsigned int>	agents_virt_cell1;
thrust::device_vector<unsigned int>	agents_virt_cell1_copy;
thrust::device_vector<unsigned int>	agents_virt_cell2;
thrust::device_vector<unsigned int>	agents_virt_cell2_copy;
thrust::device_vector<unsigned int>	agents_virt_cell3;
thrust::device_vector<unsigned int>	agents_virt_cell3_copy;

thrust::device_vector<float>		displace_param;					//dt, from 0 to 1
thrust::device_vector<float>		displace_delta;					//per agent speed
thrust::device_vector<float2>		agents_offset_xy;
thrust::device_vector<unsigned int> agents_ids_copy;
thrust::device_vector<unsigned int>	agents_goal_cell_copy;

thrust::device_vector<float>		displace_param_copy;
thrust::device_vector<float2>		agents_offset_xy_copy;
thrust::device_vector<unsigned int> future_occ;
thrust::host_vector<int>			h_seeds;
thrust::device_vector<unsigned int> agents_future_ids;
thrust::device_vector<int>			d_seeds;
thrust::device_vector<float>		agents_pos;
thrust::device_vector<float>		lca_data;
thrust::device_vector<float>		mdp_density;
thrust::device_vector<float>		mdp_lca_density;
thrust::device_vector<float>		mdp_policy;
thrust::device_vector<float>		mdp_speeds;						//scenario speeds
//
//=======================================================================================
//
class random
{
public:
    int operator() ()
    {
        return rand();
    }
};
//
//=======================================================================================
//
void print_mem( const char* msg, unsigned long long bytes )
{
	std::cout <<
		std::setw( 23 ) << msg <<
		std::setw( 10 ) << bytes / 1024 / 1024 << " MB " << 
		std::setw( 12 ) << bytes << " B" << std::endl;
}
//
//=======================================================================================
//
void print_mem2(	const char*			msg,
					const char*			typ, 
					size_t				tsz, 
					unsigned int		num,
					unsigned long long	bytes )
{
	std::cout << std::setfill( ' ' ) <<
		std::setw( 23 ) << msg <<
		"  " <<
		std::setw( 12 ) << typ << 
		" (" <<
		std::setw(  1 ) << tsz <<
		" B) X " <<
		std::setw(  7 ) << num << 
		" = " <<
		std::setw( 10 ) << bytes << 
		" B (" << 
		std::setw(  3 ) << bytes / 1024 / 1024 << 
		" MB)" << std::endl; 
}
//
//=======================================================================================
//
extern "C" void update_mdp_lca	(	std::vector< std::vector<float> >&	lca,
									std::vector< std::vector<float> >&	policy,
									float								u_time )
{
	cudaEventRecord( start, 0 );
	{
		for( unsigned int c = 0; c < MDP_CHANNELS; c++ )
		{
			thrust::copy( lca[c].begin(), lca[c].end(), lca_data.begin() + c * TOTAL_LCA_CELLS );
			thrust::copy( policy[c].begin(), policy[c].end(), mdp_policy.begin() + c * TOTAL_MDP_CELLS );
		}
	}
	cudaEventRecord( stop, 0 );
	cudaEventSynchronize( stop );
	cudaEventElapsedTime( &u_time, start, stop );
}
//
//=======================================================================================
//
extern "C" void init_cells_and_agents(	unsigned int						mdp_channels,
										std::vector<float>&					pos,
										std::vector<float>&					speed,
										std::vector<float>&					scene_speeds,
										std::vector< std::vector<float> >&	lca,
										std::vector< std::vector<float> >&	policy,
										float								scene_width,
										float								scene_depth,
										unsigned int						lca_width,
										unsigned int						lca_depth,
										unsigned int						mdp_width,
										unsigned int						mdp_depth,
										unsigned int						num_agents,
										bool&								result		)
{
	result							= true;

	cudaEventCreate( &start );
	cudaEventCreate( &stop  );	
	MDP_CHANNELS					= mdp_channels;
	NUM_AGENTS						= num_agents;
	//SCENE_WIDTH_IN_LCA_CELLS		= ( (unsigned int)(3.0 * sqrtf( (float)NUM_AGENTS )) );
	SCENE_WIDTH_IN_LCA_CELLS		= lca_width;
	SCENE_WIDTH_IN_MDP_CELLS		= mdp_width;
	LCA_TILE_WIDTH					= scene_width / (float)SCENE_WIDTH_IN_LCA_CELLS;
	HALF_SCENE_WIDTH_IN_LCA_CELLS	= (float)SCENE_WIDTH_IN_LCA_CELLS / 2.0f;
	TOTAL_LCA_CELLS					= SCENE_WIDTH_IN_LCA_CELLS * SCENE_WIDTH_IN_LCA_CELLS;
	TOTAL_MDP_CELLS					= SCENE_WIDTH_IN_MDP_CELLS * SCENE_WIDTH_IN_MDP_CELLS;
	virtual_advance_time			= 0.0f;
	fix_coll_time					= 0.0f;
	advance_time					= 0.0f;
	temp_time						= 0.0f;
	COUNTER							= 0;
	COLLISIONS						= 0;

	size_t reserved, total;
	unsigned long long	required,
						future_occ_size, 
						cells_ids_size,
						d_cells_ids_size,
						cells_occ_size,
						agents_flags_size,
						agents_flags_copy_size,
						agents_drift_counter_size,
						agents_drift_counter_copy_size,
						agents_ids_size,
						agents_dids_size,
						agents_goal_cell_size,
						agents_virt_cell1_size,
						agents_virt_cell2_size,
						agents_virt_cell3_size,
						agents_channel_size,
						agents_channel_copy_size,
						displace_param_size,
						agents_offset_xy_size,
						agents_ids_copy_size,
						seeds_size,
						agents_goal_cell_copy_size,
						agents_virt_cell1_copy_size,
						agents_virt_cell2_copy_size,
						agents_virt_cell3_copy_size,
						displace_param_copy_size,
						agents_offset_xy_copy_size,
						agents_future_ids_size,
						lca_data_size,
						mdp_density_size,
						mdp_lca_density_size,
						mdp_policy_size,
						mdp_speeds_size;

	cudaMemGetInfo( &reserved, &total );
	future_occ_size					= (unsigned long long) sizeof( unsigned int	) * (TOTAL_LCA_CELLS					);
	cells_ids_size					= (unsigned long long) sizeof( unsigned int	) * (TOTAL_LCA_CELLS					);
	d_cells_ids_size				= (unsigned long long) sizeof( unsigned int	) * (TOTAL_LCA_CELLS	* DIRECTIONS	);
	cells_occ_size					= (unsigned long long) sizeof( unsigned int	) * (TOTAL_LCA_CELLS					);
	agents_flags_size				= (unsigned long long) sizeof( unsigned int	) * (TOTAL_LCA_CELLS	* DIRECTIONS	);
	agents_flags_copy_size			= (unsigned long long) sizeof( unsigned int	) * (TOTAL_LCA_CELLS	* DIRECTIONS	);
	agents_drift_counter_size		= (unsigned long long) sizeof( int			) * (TOTAL_LCA_CELLS	* DIRECTIONS	);
	agents_drift_counter_copy_size	= (unsigned long long) sizeof( int			) * (TOTAL_LCA_CELLS	* DIRECTIONS	);
	agents_ids_size					= (unsigned long long) sizeof( unsigned int	) * (TOTAL_LCA_CELLS	* DIRECTIONS	);
	agents_dids_size				= (unsigned long long) sizeof( unsigned int	) * (TOTAL_LCA_CELLS	* DIRECTIONS	);
	agents_goal_cell_size			= (unsigned long long) sizeof( unsigned int	) * (TOTAL_LCA_CELLS	* DIRECTIONS	);
	agents_virt_cell1_size			= (unsigned long long) sizeof( unsigned int	) * (TOTAL_LCA_CELLS	* DIRECTIONS	);
	agents_virt_cell1_copy_size		= (unsigned long long) sizeof( unsigned int	) * (TOTAL_LCA_CELLS	* DIRECTIONS	);
	agents_virt_cell2_size			= (unsigned long long) sizeof( unsigned int	) * (TOTAL_LCA_CELLS	* DIRECTIONS	);
	agents_virt_cell2_copy_size		= (unsigned long long) sizeof( unsigned int	) * (TOTAL_LCA_CELLS	* DIRECTIONS	);
	agents_virt_cell3_size			= (unsigned long long) sizeof( unsigned int	) * (TOTAL_LCA_CELLS	* DIRECTIONS	);
	agents_virt_cell3_copy_size		= (unsigned long long) sizeof( unsigned int	) * (TOTAL_LCA_CELLS	* DIRECTIONS	);
	agents_channel_size				= (unsigned long long) sizeof( unsigned int	) * (TOTAL_LCA_CELLS	* DIRECTIONS	);
	agents_channel_copy_size		= (unsigned long long) sizeof( unsigned int	) * (TOTAL_LCA_CELLS	* DIRECTIONS	);
	displace_param_size				= (unsigned long long) sizeof( float		) * (TOTAL_LCA_CELLS	* DIRECTIONS	);
	agents_offset_xy_size			= (unsigned long long) sizeof( float2		) * (TOTAL_LCA_CELLS	* DIRECTIONS	);
	agents_ids_copy_size			= (unsigned long long) sizeof( unsigned int	) * (TOTAL_LCA_CELLS	* DIRECTIONS	);
	agents_goal_cell_copy_size		= (unsigned long long) sizeof( unsigned int	) * (TOTAL_LCA_CELLS	* DIRECTIONS	);
	displace_param_copy_size		= (unsigned long long) sizeof( float		) * (TOTAL_LCA_CELLS	* DIRECTIONS	);
	agents_offset_xy_copy_size		= (unsigned long long) sizeof( float2		) * (TOTAL_LCA_CELLS	* DIRECTIONS	);
	seeds_size						= (unsigned long long) sizeof( int			) * (TOTAL_LCA_CELLS	* DIRECTIONS	);
	agents_future_ids_size			= (unsigned long long) sizeof( unsigned int	) * (TOTAL_LCA_CELLS	* DIRECTIONS	);
	lca_data_size					= (unsigned long long) sizeof( float		) * (TOTAL_LCA_CELLS	* MDP_CHANNELS	);
	mdp_density_size				= (unsigned long long) sizeof( float		) * (TOTAL_MDP_CELLS					);
	mdp_lca_density_size			= (unsigned long long) sizeof( float		) * (TOTAL_LCA_CELLS					);
	mdp_policy_size					= (unsigned long long) sizeof( float		) * (TOTAL_MDP_CELLS	* MDP_CHANNELS	);
	mdp_speeds_size					= (unsigned long long) sizeof( float		) * (TOTAL_MDP_CELLS					);
	required						=	future_occ_size					+ 
										cells_ids_size					+
										d_cells_ids_size				+
										cells_occ_size					+
										agents_flags_size				+
										agents_flags_copy_size			+
										agents_drift_counter_size		+
										agents_drift_counter_copy_size	+
										agents_ids_size					+
										agents_dids_size				+
										agents_goal_cell_size			+
										agents_virt_cell1_size			+
										agents_virt_cell2_size			+
										agents_virt_cell3_size			+
										agents_offset_xy_size			+
										agents_ids_copy_size			+
										agents_goal_cell_copy_size		+
										agents_virt_cell1_copy_size		+
										agents_virt_cell2_copy_size		+
										agents_virt_cell3_copy_size		+
										agents_channel_size				+
										agents_channel_copy_size		+
										displace_param_size				+
										displace_param_copy_size		+
										agents_offset_xy_copy_size		+
										seeds_size						+
										agents_future_ids_size			+
										lca_data_size					+
										mdp_density_size				+
										mdp_lca_density_size			+
										mdp_policy_size					+
										mdp_speeds_size;
	std::cout << std::endl;
	std::cout << "NUM_AGENTS:\t\t" << num_agents << std::endl;
	std::cout << "SCENE_WIDTH:\t\t" << scene_width << std::endl;
	std::cout << "SCENE_WIDTH_IN_LCA_CELLS:\t" << SCENE_WIDTH_IN_LCA_CELLS << std::endl;
	std::cout << "SCENE_WIDTH_IN_MDP_CELLS:\t" << SCENE_WIDTH_IN_MDP_CELLS << std::endl;
	std::cout << "LCA_TILE_WIDTH:\t\t" << LCA_TILE_WIDTH << std::endl;
	std::cout << "TOTAL_LCA_CELLS:\t\t" << TOTAL_LCA_CELLS << std::endl;
	std::cout << "TOTAL_MDP_CELLS:\t\t" << TOTAL_MDP_CELLS << std::endl;
	
	std::cout << std::endl;
	std::cout << "-------------------------------------------------" << std::endl;
	print_mem( "Total memory",			(unsigned long long) total		);
	print_mem( "Reserved memory",		(unsigned long long) reserved	);
	std::cout << "-------------------------------------------------" << std::endl;
	print_mem2( "Random seeds",				"int",			sizeof(int),			TOTAL_LCA_CELLS * DIRECTIONS,	seeds_size						);
	print_mem2( "Cells ids",				"unsigned int", sizeof(unsigned int),	TOTAL_LCA_CELLS,				cells_ids_size					);
	print_mem2( "D cells ids",				"unsigned int", sizeof(unsigned int),	TOTAL_LCA_CELLS * DIRECTIONS,	d_cells_ids_size				);
	print_mem2( "Cells occ",				"unsigned int", sizeof(unsigned int),	TOTAL_LCA_CELLS,				cells_occ_size					);
	print_mem2( "Agents flags",				"unsigned int", sizeof(unsigned int),	TOTAL_LCA_CELLS * DIRECTIONS,	agents_flags_size				);
	print_mem2( "Agents flags copy",		"unsigned int", sizeof(unsigned int),	TOTAL_LCA_CELLS * DIRECTIONS,	agents_flags_copy_size			);
	print_mem2( "Agents drift counter",		"int",			sizeof(int),			TOTAL_LCA_CELLS * DIRECTIONS,	agents_drift_counter_size		);
	print_mem2( "Agents drift counter copy","int",			sizeof(int),			TOTAL_LCA_CELLS * DIRECTIONS,	agents_drift_counter_copy_size	);	
	print_mem2( "Future Occupancy",			"unsigned int", sizeof(unsigned int),	TOTAL_LCA_CELLS * DIRECTIONS,	future_occ_size					);
	print_mem2( "Agents ids",				"unsigned int", sizeof(unsigned int),	TOTAL_LCA_CELLS * DIRECTIONS,	agents_ids_size					);
	print_mem2( "Agents dids",				"unsigned int", sizeof(unsigned int),	TOTAL_LCA_CELLS * DIRECTIONS,	agents_dids_size				);
	print_mem2( "Agents ids copy",			"unsigned int", sizeof(unsigned int),	TOTAL_LCA_CELLS * DIRECTIONS,	agents_ids_copy_size			);
	print_mem2( "Agents future ids",		"unsigned int", sizeof(unsigned int),	TOTAL_LCA_CELLS * DIRECTIONS,	agents_future_ids_size			);
	print_mem2( "Agents goal cell",			"unsigned int", sizeof(unsigned int),	TOTAL_LCA_CELLS * DIRECTIONS,	agents_goal_cell_size			);
	print_mem2( "Agents goal cell copy",	"unsigned int", sizeof(unsigned int),	TOTAL_LCA_CELLS * DIRECTIONS,	agents_goal_cell_copy_size		);
	print_mem2( "Agents virt cell 1",		"unsigned int", sizeof(unsigned int),	TOTAL_LCA_CELLS * DIRECTIONS,	agents_virt_cell1_size			);
	print_mem2( "Agents virt cell 1 copy",	"unsigned int", sizeof(unsigned int),	TOTAL_LCA_CELLS * DIRECTIONS,	agents_virt_cell1_copy_size		);
	print_mem2( "Agents virt cell 2",		"unsigned int", sizeof(unsigned int),	TOTAL_LCA_CELLS * DIRECTIONS,	agents_virt_cell2_size			);
	print_mem2( "Agents virt cell 2 copy",	"unsigned int", sizeof(unsigned int),	TOTAL_LCA_CELLS * DIRECTIONS,	agents_virt_cell2_copy_size		);
	print_mem2( "Agents virt cell 3",		"unsigned int", sizeof(unsigned int),	TOTAL_LCA_CELLS * DIRECTIONS,	agents_virt_cell3_size			);
	print_mem2( "Agents virt cell 3 copy",	"unsigned int", sizeof(unsigned int),	TOTAL_LCA_CELLS * DIRECTIONS,	agents_virt_cell3_copy_size		);
	print_mem2( "Agents channel",			"unsigned int", sizeof(unsigned int),	TOTAL_LCA_CELLS * DIRECTIONS,	agents_channel_size				);
	print_mem2( "Agents channel copy",		"unsigned int", sizeof(unsigned int),	TOTAL_LCA_CELLS * DIRECTIONS,	agents_channel_copy_size		);
	print_mem2( "Displace param",			"float",		sizeof(float),			TOTAL_LCA_CELLS * DIRECTIONS,	displace_param_size				);
	print_mem2( "Displace param copy",		"float",		sizeof(float),			TOTAL_LCA_CELLS * DIRECTIONS,	displace_param_copy_size		);
	print_mem2( "LCA data",					"float",		sizeof(float),			TOTAL_LCA_CELLS * MDP_CHANNELS,	lca_data_size					);
	print_mem2( "MDP density",				"float",		sizeof(float),			TOTAL_MDP_CELLS,				mdp_density_size				);
	print_mem2( "MDP speeds",				"float",		sizeof(float),			TOTAL_MDP_CELLS,				mdp_speeds_size					);
	print_mem2( "MDP LCA density",			"float",		sizeof(float),			TOTAL_LCA_CELLS,				mdp_lca_density_size			);
	print_mem2( "MDP policy",				"float",		sizeof(float),			TOTAL_MDP_CELLS * MDP_CHANNELS,	mdp_policy_size					);
	print_mem2( "Agents xy",				"float2",		sizeof(float2),			TOTAL_LCA_CELLS * DIRECTIONS,	agents_offset_xy_size			);
	print_mem2( "Agents xy copy",			"float2",		sizeof(float2),			TOTAL_LCA_CELLS * DIRECTIONS,	agents_offset_xy_copy_size		);
	std::cout << "-------------------------------------------------" << std::endl;
	print_mem( "Required memory",		required						);
	if( required > (unsigned long long)reserved )
	{
		std::cout << std::endl;
		std::cout << "Not enough memory!" << std::endl << std::endl;
		result = false;
		return;
	}
	std::cout << std::endl;

	cudaEventRecord( start, 0 );
	{
		cells_ids.resize				( TOTAL_LCA_CELLS					);
		d_cells_ids.resize				( TOTAL_LCA_CELLS * DIRECTIONS		);
		cells_row.resize				( TOTAL_LCA_CELLS					);
		cells_col.resize				( TOTAL_LCA_CELLS					);
		cells_occ.resize				( TOTAL_LCA_CELLS					);
		agents_flags.resize				( TOTAL_LCA_CELLS * DIRECTIONS		);
		agents_flags_copy.resize		( TOTAL_LCA_CELLS * DIRECTIONS		);
		agents_drift_counter.resize		( TOTAL_LCA_CELLS * DIRECTIONS		);
		agents_drift_counter_copy.resize( TOTAL_LCA_CELLS * DIRECTIONS		);
		agents_ids.resize				( TOTAL_LCA_CELLS * DIRECTIONS		);
		agents_dids.resize				( TOTAL_LCA_CELLS * DIRECTIONS		);
		agents_goal_cell.resize			( TOTAL_LCA_CELLS * DIRECTIONS		);
		agents_virt_cell1.resize		( TOTAL_LCA_CELLS * DIRECTIONS		);
		agents_virt_cell1_copy.resize	( TOTAL_LCA_CELLS * DIRECTIONS		);
		agents_virt_cell2.resize		( TOTAL_LCA_CELLS * DIRECTIONS		);
		agents_virt_cell2_copy.resize	( TOTAL_LCA_CELLS * DIRECTIONS		);
		agents_virt_cell3.resize		( TOTAL_LCA_CELLS * DIRECTIONS		);
		agents_virt_cell3_copy.resize	( TOTAL_LCA_CELLS * DIRECTIONS		);
		agents_channel.resize			( TOTAL_LCA_CELLS * DIRECTIONS		);
		agents_channel_copy.resize		( TOTAL_LCA_CELLS * DIRECTIONS		);
		displace_param.resize			( TOTAL_LCA_CELLS * DIRECTIONS		);
		agents_offset_xy.resize			( TOTAL_LCA_CELLS * DIRECTIONS		);
		agents_ids_copy.resize			( TOTAL_LCA_CELLS * DIRECTIONS		);
		agents_goal_cell_copy.resize	( TOTAL_LCA_CELLS * DIRECTIONS		);
		displace_param_copy.resize		( TOTAL_LCA_CELLS * DIRECTIONS		);
		agents_offset_xy_copy.resize	( TOTAL_LCA_CELLS * DIRECTIONS		);
		future_occ.resize				( TOTAL_LCA_CELLS					);
		h_seeds.resize					( TOTAL_LCA_CELLS * DIRECTIONS		);
		agents_future_ids.resize		( TOTAL_LCA_CELLS * DIRECTIONS		);
		mdp_density.resize				( TOTAL_MDP_CELLS					);
		mdp_lca_density.resize			( TOTAL_LCA_CELLS					);

		thrust::fill( agents_drift_counter.begin(), agents_drift_counter.end(), 0 );
		//thrust::generate ( h_seeds.begin(), h_seeds.end(), random() );
		thrust::generate ( h_seeds.begin(), h_seeds.end(), random );
		d_seeds			= h_seeds;

		agents_pos		= pos;
		displace_delta	= speed;
		mdp_speeds		= scene_speeds;
		

		lca_data.resize( TOTAL_LCA_CELLS * DIRECTIONS );
		mdp_policy.resize( TOTAL_MDP_CELLS * DIRECTIONS );
		for( unsigned int c = 0; c < MDP_CHANNELS; c++ )
		{
			thrust::copy( lca[c].begin(), lca[c].end(), lca_data.begin() + c * TOTAL_LCA_CELLS );
			thrust::copy( policy[c].begin(), policy[c].end(), mdp_policy.begin() + c * TOTAL_MDP_CELLS );
		}
	}
	cudaEventRecord( stop, 0 );
	cudaEventSynchronize( stop );
	cudaEventElapsedTime( &mem_alloc_time, start, stop );

	cudaEventRecord( start, 0 );
	{
		init_cells_and_agents(	agents_pos,
								lca_data,
								d_seeds,
								cells_ids,
								d_cells_ids,
								cells_occ,
								agents_ids,
								agents_dids,
								agents_goal_cell,
								agents_virt_cell1,
								agents_virt_cell2,
								agents_virt_cell3,
								agents_channel,
								displace_param,
								agents_offset_xy,
								HALF_SCENE_WIDTH_IN_LCA_CELLS,
								LCA_TILE_WIDTH,
								SCENE_WIDTH_IN_LCA_CELLS,
								SCENE_WIDTH_IN_MDP_CELLS,
								NUM_AGENTS,
								MDP_CHANNELS				);
	}
	cudaEventRecord( stop, 0 );
	cudaEventSynchronize( stop );
	cudaEventElapsedTime( &agents_time, start, stop );

#if defined __DEBUG && defined __PRINT_AGENTS
	std::cout << "Agents:\n";
	for( unsigned int a = 0 ; a < TOTAL_LCA_CELLS * DIRECTIONS; a++ )
	{
		if( agents_ids[a] < TOTAL_LCA_CELLS * DIRECTIONS )
		{
			float2 axy = agents_offset_xy[a];
			std::cout << "ID: "	<< std::setw( 5 ) << agents_ids[a];
			if( (a / DIRECTIONS) == agents_goal_cell[a] )
			{
				std::cout << " CURR: "	<< std::setw( 3 ) <<  "G";
			}
			else
			{
				std::cout << " CURR: "	<< std::setw( 3 ) <<  (a / DIRECTIONS);
			}
			std::cout << std::fixed <<
				" NEXT1: "	<< std::setw( 3 ) << agents_virt_cell1[a] <<
				" NEXT2: "	<< std::setw( 3 ) << agents_virt_cell2[a] <<
				" NEXT3: "	<< std::setw( 3 ) << agents_virt_cell3[a] <<
				" GOAL: "	<< std::setw( 3 ) << agents_goal_cell[a] <<
				" PARAM:"	<< std::setw( 6 ) << std::setprecision( 2 ) << displace_param[a] <<
				" DX: "		<< std::setw( 6 ) << std::setprecision( 2 ) << axy.x <<
				" DY: "		<< std::setw( 6 ) << std::setprecision( 2 ) << axy.y << std::endl;
		}
	}
	std::cout << std::endl;
#endif

#ifdef __DEBUG
	std::cout << "Initial occupancy:\n";
	for( unsigned int r = 0; r < SCENE_WIDTH_IN_LCA_CELLS; r++ )
	{
		for( unsigned int c = 0; c < SCENE_WIDTH_IN_LCA_CELLS; c++ )
		{
			std::cout << std::setw( 2 ) << cells_occ[r * SCENE_WIDTH_IN_LCA_CELLS + c];
		}
		std::cout << std::endl;
	}
	COLLISIONS = count_collisions( cells_occ );
	std::cout << "Collisions: " << COLLISIONS << std::endl;
	std::cout << std::endl;
	std::cout << "-------------------------------------------------" << std::endl << std::endl;
#endif

	total_time	= mem_alloc_time + agents_time;
	//std::cout << std::fixed	<< std::setfill( '0' ) << std::endl <<
	std::cout << std::endl <<
	"TIMES:                "																					<< std::endl <<
	"MEM ALLOC:            " << std::setw( 10 ) << std::setprecision( 6 ) << mem_alloc_time			<< " ms."	<< std::endl <<
	"AGENTS:               " << std::setw( 10 ) << std::setprecision( 6 ) << agents_time			<< " ms."	<< std::endl <<
	"----------------------------"																				<< std::endl <<
	"TOTAL:                " << std::setw( 10 ) << std::setprecision( 6 ) << total_time				<< " ms."	<< std::endl <<
	"----------------------------"																				<< std::endl <<
	std::endl;
#ifdef __DO_PAUSE
	system( "pause" );
#endif
}
//
//=======================================================================================
//
extern "C" void launch_lca_kernel( std::vector<float>&	pos,
								   std::vector<float>&	lca,
								   std::vector<float>&	dens,
								   float&				avg_racing_qw,
								   float&				avg_scatter_gather	)
{
//->VIRTUAL_ADVANCE
	/*
	cudaEventRecord( start, 0 );
	{
		thrust::copy			(	agents_ids.begin(), 
									agents_ids.end(), 
									agents_future_ids.begin()	);

		advance_agents_virtually(	cells_ids,
									future_occ,
									agents_future_ids,
									agents_ids_copy,
									agents_virt_cell1,
									agents_virt_cell2,
									agents_virt_cell3,
									agents_goal_cell,
									SCENE_WIDTH_IN_LCA_CELLS	);
	}
	cudaEventRecord( stop, 0 );
	cudaEventSynchronize( stop );
	cudaEventElapsedTime( &temp_time, start, stop );
	virtual_advance_time += temp_time;
	COLLISIONS = count_collisions( future_occ );
	*/

#if defined __DEBUG && defined __PRINT_FUTURE_OCC
	std::cout << "Future occupancy[" << COUNTER << "] (natural):" << std::endl;
	for( unsigned int r = 0; r < SCENE_WIDTH_IN_LCA_CELLS; r++ )
	{
		for( unsigned int c = 0; c < SCENE_WIDTH_IN_LCA_CELLS; c++ )
		{
			std::cout << std::setw( 2 ) << future_occ[r * SCENE_WIDTH_IN_LCA_CELLS + c];
		}
		std::cout << std::endl;
	}
	std::cout << "Collisions: " << COLLISIONS << std::endl;
	std::cout << std::endl;
#endif
//<-VIRTUAL_ADVANCE

//->COLLISION_FIX
	cudaEventRecord( start, 0 );
	{
		fix_collisions	(	lca_data,
							agents_channel,
							cells_ids,
							cells_occ,
							agents_virt_cell1,
							agents_virt_cell2,
							agents_virt_cell3,
							agents_flags,
							agents_flags_copy,
							agents_drift_counter,
							SCENE_WIDTH_IN_LCA_CELLS	);
	}
	cudaEventRecord( stop, 0 );
	cudaEventSynchronize( stop );
	cudaEventElapsedTime( &temp_time, start, stop );
	fix_coll_time = temp_time;
	avg_racing_qw = (fix_coll_time + avg_racing_qw) / 2.0f;

//<-COLLISION_FIX

//->ACTUAL_ADVANCE
	cudaEventRecord( start, 0 );
	{
		advance_agents(	cells_ids,
						cells_occ,
						agents_flags,
						agents_flags_copy,
						agents_ids,
						agents_ids_copy,
						agents_goal_cell,
						agents_goal_cell_copy,
						agents_virt_cell1,
						agents_virt_cell1_copy,
						agents_virt_cell2,
						agents_virt_cell2_copy,
						agents_virt_cell3,
						agents_virt_cell3_copy,
						agents_channel,
						agents_channel_copy,
						agents_drift_counter,
						agents_drift_counter_copy,
						displace_param,
						displace_param_copy,
						agents_offset_xy,
						agents_offset_xy_copy,
						agents_pos,
						lca_data,
						displace_delta,
						mdp_speeds,
						SCENE_WIDTH_IN_LCA_CELLS,
						SCENE_WIDTH_IN_MDP_CELLS,
						MDP_CHANNELS,
						LCA_TILE_WIDTH				);
		thrust::copy( agents_pos.begin(), agents_pos.end(), pos.begin() );
//->UPDATE_MDP_DENSITY
		update_mdp_density( mdp_lca_density,
							mdp_density,
							cells_ids,
							cells_occ,
							SCENE_WIDTH_IN_LCA_CELLS,
							SCENE_WIDTH_IN_MDP_CELLS	);
		thrust::copy( mdp_density.begin(), mdp_density.end(), dens.begin() );
//<-UPDATE_MDP_DENSITY
	}
	cudaEventRecord( stop, 0 );
	cudaEventSynchronize( stop );
	cudaEventElapsedTime( &temp_time, start, stop );
	advance_time = temp_time;
	avg_scatter_gather = (avg_scatter_gather + advance_time) / 2.0f;

#if defined __DEBUG && defined __PRINT_AGENTS
	std::cout << "agents[" << COUNTER << "]" << std::endl;
	unsigned int na = 0;

	/*
	for( unsigned int a = 0 ; a < TOTAL_LCA_CELLS * DIRECTIONS; a++ )
	{
		if( agents_ids[a] < TOTAL_LCA_CELLS * DIRECTIONS )
		{
			na++;
		}
	}
	*/

	//if( na != NUM_AGENTS )
	{
		//na = 0;
		for( unsigned int a = 0 ; a < TOTAL_LCA_CELLS * DIRECTIONS; a++ )
		{
			if( agents_ids[a] < TOTAL_LCA_CELLS * DIRECTIONS )
			{
				float2 axy			= agents_offset_xy[a];
				unsigned int idx	= agents_ids[a] * 4;
				float px			= agents_pos[idx + 0];
				float pz			= agents_pos[idx + 2];
				std::cout << "ID: "	<< std::setw( 5 ) << agents_ids[a];
				if( (a / DIRECTIONS) == agents_goal_cell[a] )
				{
					std::cout << " CELL: "	<< std::setw( 3 ) <<  "G";
				}
				else
				{
					std::cout << " CELL: "	<< std::setw( 3 ) <<  (a / DIRECTIONS);
				}
				std::cout << std::fixed <<
					" P: "	<< std::setw( 4 ) << std::setprecision( 2 ) << displace_param[a] <<
					" F: "	<< std::setw( 3 ) << std::setprecision( 0 ) << agents_flags[a] <<
					" D: "	<< std::setw( 3 ) << std::setprecision( 0 ) << agents_drift_counter[a] <<
					" N1: "	<< std::setw( 3 ) << agents_virt_cell1[a] <<
					" N2: "	<< std::setw( 3 ) << agents_virt_cell2[a] <<
					" N3: "	<< std::setw( 3 ) << agents_virt_cell3[a] <<
					" G: "	<< std::setw( 3 ) << agents_goal_cell[a] <<
					" Px: "	<< std::setw( 7 ) << std::setprecision( 2 ) << px <<
					" Py: "	<< std::setw( 7 ) << std::setprecision( 2 ) << pz <<
					" dX: "	<< std::setw( 7 ) << std::setprecision( 2 ) << axy.x <<
					" dY: "	<< std::setw( 7 ) << std::setprecision( 2 ) << axy.y << std::endl;
				na++;
			}
		}
		std::cout << std::endl;

		if( na != NUM_AGENTS )
			std::cout << "Got " << na << " agents!!!" << std::endl;
		//system( "pause" );
	}
#endif

#if defined __DEBUG && defined __PRINT_OCC
	std::cout << "occupancy[" << COUNTER << "] (fixed):" << std::endl;
	for( unsigned int r = 0; r < SCENE_WIDTH_IN_LCA_CELLS; r++ )
	{
		for( unsigned int c = 0; c < SCENE_WIDTH_IN_LCA_CELLS; c++ )
		{
			std::cout << std::setw( 2 ) << cells_occ[r * SCENE_WIDTH_IN_LCA_CELLS + c];
		}
		std::cout << std::endl;
	}
	COLLISIONS = count_collisions( cells_occ );
	if( COLLISIONS > 0 )
		std::cout << "Collisions: " << COLLISIONS << " (!)" << std::endl;
	else
		std::cout << "Collisions: " << COLLISIONS << std::endl;
	std::cout << std::endl;
	std::cout << "-------------------------------------------------" << std::endl << std::endl;

	std::cout << "mdp_density[" << COUNTER << "]:" << std::endl;
	for( unsigned int r = 0; r < SCENE_WIDTH_IN_MDP_CELLS; r++ )
	{
		for( unsigned int c = 0; c < SCENE_WIDTH_IN_MDP_CELLS; c++ )
		{
			std::cout << std::setw( 2 ) << std::setprecision( 0 ) << mdp_density[r * SCENE_WIDTH_IN_MDP_CELLS + c];
		}
		std::cout << std::endl;
	}
	std::cout << "-------------------------------------------------" << std::endl << std::endl;

#endif
//<-ACTUAL_ADVANCE

	COUNTER++;
#ifdef __DO_PAUSE
	system( "pause" );
#endif
}
//
//=======================================================================================
//
extern "C" void cleanup_lca( void )
{
	cells_ids.clear();
	d_cells_ids.clear();
	cells_row.clear();
	cells_col.clear();
	cells_occ.clear();
	agents_ids.clear();
	agents_dids.clear();
	agents_goal_cell.clear();
	agents_virt_cell1.clear();
	agents_offset_xy.clear();
	agents_ids_copy.clear();
	agents_goal_cell_copy.clear();
	agents_virt_cell1_copy.clear();
	agents_offset_xy_copy.clear();
	future_occ.clear();
	h_seeds.clear();
	agents_future_ids.clear();
	d_seeds.clear();
	agents_pos.clear();
	lca_data.clear();
	displace_delta.clear();
	mdp_density.clear();
	mdp_policy.clear();
}
//
//=======================================================================================
#endif

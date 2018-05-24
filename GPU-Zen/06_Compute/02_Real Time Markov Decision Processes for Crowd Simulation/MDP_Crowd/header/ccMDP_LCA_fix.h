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

struct fix_agent_collisions1
{
	unsigned int	scene_width_in_lca_cells;
	unsigned int	total_lca_cells;
	unsigned int	tlcd;
	unsigned int	lca_row;
	unsigned int	lca_col;
	unsigned int	lca_cell_id;

	unsigned int*	raw_occ_ptr;
	unsigned int*	raw_vc1_ptr;
	unsigned int*	raw_vc2_ptr;
	unsigned int*	raw_vc3_ptr;
	unsigned int*	raw_flags_ptr;
	unsigned int*	raw_flags_copy_ptr;
	int*			raw_drift_ptr;

	float			f_deg_inc;	
	int				swilc;

	fix_agent_collisions1(	unsigned int	_swilc,
							unsigned int*	_rop,
							unsigned int*	_rvc1p,
							unsigned int*	_rvc2p,
							unsigned int*	_rvc3p,
							unsigned int*	_rfp,
							unsigned int*	_rfcp,
							int*			_rdp	) : scene_width_in_lca_cells	(	_swilc			),
														raw_occ_ptr					(	_rop			),
														raw_vc1_ptr					(	_rvc1p			),
														raw_vc2_ptr					(	_rvc2p			),
														raw_vc3_ptr					(	_rvc3p			),
														raw_flags_ptr				(	_rfp			),
														raw_flags_copy_ptr			(	_rfcp			),
														raw_drift_ptr				(	_rdp			)
	{
		total_lca_cells			= scene_width_in_lca_cells * scene_width_in_lca_cells;
		f_deg_inc				= 360.0f / (float)DIRECTIONS;
		swilc					= (int)scene_width_in_lca_cells;
		lca_row					= 0;
		lca_col					= 0;
		lca_cell_id				= 0;
		tlcd					= total_lca_cells * DIRECTIONS;
	}

	template <typename Tuple>						//0
	__host__ __device__ void operator()( Tuple t )	//CELL_ID
	{
		lca_cell_id							= thrust::get<0>(t);
		unsigned int	neighbor_cell_id	= total_lca_cells;
		unsigned int	nda					= 0;
		unsigned int	cda					= 0;
		unsigned int	a					= 0;
		unsigned int	d					= 0;
		int				test_neighbor_col	=-1;
		int				test_neighbor_row	=-1;
		float			test_radians		= 0.0f;
		bool			pulled				= false;
		lca_row								= (int)(lca_cell_id / scene_width_in_lca_cells);
		lca_col								= (int)(lca_cell_id % scene_width_in_lca_cells);


		for( a = 0; a < DIRECTIONS; a++ )
		{
			cda = lca_cell_id * DIRECTIONS + a;
			if( raw_vc1_ptr[cda] == tlcd ) break;
			else if( raw_vc1_ptr[cda] == lca_cell_id ) raw_flags_ptr[cda] = 0;
		}


		for( d = 0; d < DIRECTIONS; d++ )	//FOR AGENTS INCOMING, FIRST CHOICE
		{
			test_radians		= DEG2RAD * (float)d * f_deg_inc;
			test_neighbor_col	= (int)lca_col + ((int)SIGNF(  cosf( test_radians ) ));
			test_neighbor_row	= (int)lca_row + ((int)SIGNF( -sinf( test_radians ) ));
			if( INBOUNDS( test_neighbor_col, swilc ) && INBOUNDS( test_neighbor_row, swilc ) )
			{
				neighbor_cell_id = (unsigned int)(test_neighbor_row * swilc + test_neighbor_col);
				for( a = 0; a < DIRECTIONS; a++ )
				{
					nda = neighbor_cell_id * DIRECTIONS + a;
					if( raw_vc1_ptr[nda] == tlcd ) break;
					else if( raw_vc1_ptr[nda] == lca_cell_id )	//INCOMING_HERE, FIRST CHOICE
					{
						if( !pulled && raw_occ_ptr[lca_cell_id] == 0 && raw_drift_ptr[nda] == 0 )
						{
							//raw_drift_ptr[nda]	= 0;				//RESET_DRIFT
							raw_flags_ptr[nda]	= 1;				//GO1_FLAG
							pulled = true;							//PULL_ONLY_ONE
						}
						else
						{
							raw_flags_ptr[nda]	= 0;				//HOLD_FLAG
						}
					}
				}
			}
		}
	}
};
//
//=======================================================================================
//
struct fix_agent_collisions2
{
	unsigned int	scene_width_in_lca_cells;
	unsigned int	total_lca_cells;
	unsigned int	tlcd;
	unsigned int	lca_row;
	unsigned int	lca_col;
	unsigned int	lca_cell_id;

	float*			raw_lca_ptr;
	unsigned int*	raw_chn_ptr;
	unsigned int*	raw_occ_ptr;
	unsigned int*	raw_vc1_ptr;
	unsigned int*	raw_vc2_ptr;
	unsigned int*	raw_vc3_ptr;
	unsigned int*	raw_flags_ptr;
	int*			raw_drift_ptr;

	float			f_deg_inc;	
	int				swilc;

	fix_agent_collisions2(	unsigned int	_swilc,
							float*			_rlp,
							unsigned int*	_rcp,
							unsigned int*	_rop,
							unsigned int*	_rvc1p,
							unsigned int*	_rvc2p,
							unsigned int*	_rvc3p,
							unsigned int*	_rfp,
							int*			_rdp	) : scene_width_in_lca_cells	(	_swilc			),
														raw_lca_ptr					(	_rlp			),
														raw_chn_ptr					(	_rcp			),
														raw_occ_ptr					(	_rop			),
														raw_vc1_ptr					(	_rvc1p			),
														raw_vc2_ptr					(	_rvc2p			),
														raw_vc3_ptr					(	_rvc3p			),
														raw_flags_ptr				(	_rfp			),
														raw_drift_ptr				(	_rdp			)
	{
		total_lca_cells			= scene_width_in_lca_cells * scene_width_in_lca_cells;
		f_deg_inc				= 360.0f / (float)DIRECTIONS;
		swilc					= (int)scene_width_in_lca_cells;
		lca_row					= 0;
		lca_col					= 0;
		lca_cell_id				= 0;
		tlcd					= total_lca_cells * DIRECTIONS;
	}

	template <typename Tuple>						//0
	__host__ __device__ void operator()( Tuple t )	//CELL_ID
	{
		lca_cell_id							= thrust::get<0>(t);
		unsigned int	neighbor_cell_id	= total_lca_cells;
		unsigned int	nda					= 0;
		unsigned int	a					= 0;
		unsigned int	d					= 0;
		unsigned int	lca_idx				= 0;
		int				test_neighbor_col	=-1;
		int				test_neighbor_row	=-1;
		float			test_radians		= 0.0f;
		bool			pulled				= false;
		lca_row								= (int)(lca_cell_id / scene_width_in_lca_cells);
		lca_col								= (int)(lca_cell_id % scene_width_in_lca_cells);

		for( d = 0; d < DIRECTIONS; d++ )	//FOR AGENTS INCOMING
		{
			test_radians		= DEG2RAD * (float)d * f_deg_inc;
			test_neighbor_col	= (int)lca_col + ((int)SIGNF(  cosf( test_radians ) ));
			test_neighbor_row	= (int)lca_row + ((int)SIGNF( -sinf( test_radians ) ));
			if( INBOUNDS( test_neighbor_col, swilc ) && INBOUNDS( test_neighbor_row, swilc ) )
			{
				neighbor_cell_id = (unsigned int)(test_neighbor_row * swilc + test_neighbor_col);
				for( a = 0; a < DIRECTIONS; a++ )
				{
					nda = neighbor_cell_id * DIRECTIONS + a;
					if( raw_vc2_ptr[nda] == tlcd ) break;
					if( raw_vc2_ptr[nda] == lca_cell_id )	//INCOMING_HERE, SECOND CHOICE
					{
						lca_idx = raw_chn_ptr[nda] * total_lca_cells + lca_cell_id;
						if( raw_occ_ptr[lca_cell_id]	== 0				&& 
							raw_flags_ptr[nda]			== 0				&&
							raw_drift_ptr[nda]			== MAX_DRIFT - 1	&&
							raw_lca_ptr[lca_idx]		!= (float)DIRECTIONS	)
						{
							raw_flags_ptr[nda] = 2;			//GO2_FLAG
							pulled = true;
							break;
						}
					}
				}
			}
			if( pulled ) break;
		}

		if( !pulled )
		{
			for( d = 0; d < DIRECTIONS; d++ )	//FOR AGENTS INCOMING
			{
				test_radians		= DEG2RAD * (float)d * f_deg_inc;
				test_neighbor_col	= (int)lca_col + ((int)SIGNF(  cosf( test_radians ) ));
				test_neighbor_row	= (int)lca_row + ((int)SIGNF( -sinf( test_radians ) ));
				if( INBOUNDS( test_neighbor_col, swilc ) && INBOUNDS( test_neighbor_row, swilc ) )
				{
					neighbor_cell_id = (unsigned int)(test_neighbor_row * swilc + test_neighbor_col);
					for( a = 0; a < DIRECTIONS; a++ )
					{
						nda = neighbor_cell_id * DIRECTIONS + a;
						if( raw_vc2_ptr[nda] == tlcd ) break;
						if( raw_vc2_ptr[nda] == lca_cell_id )	//INCOMING_HERE, SECOND CHOICE
						{
							lca_idx = raw_chn_ptr[nda] * total_lca_cells + lca_cell_id;
							if( raw_occ_ptr[lca_cell_id]	== 0				&& 
								raw_flags_ptr[nda]			== 0				&&
								raw_drift_ptr[nda]			== 0				&&
								raw_lca_ptr[lca_idx]		!= (float)DIRECTIONS	)
							{
								raw_flags_ptr[nda] = 2;			//GO2_FLAG
								raw_drift_ptr[nda] = MAX_DRIFT;
								pulled = true;
								break;
							}
						}
					}
				}
				if( pulled ) break;
			}
		}
	}
};
//
//=======================================================================================
//
struct fix_agent_collisions3
{
	unsigned int	scene_width_in_lca_cells;
	unsigned int	total_lca_cells;
	unsigned int	tlcd;
	unsigned int	lca_row;
	unsigned int	lca_col;
	unsigned int	lca_cell_id;

	float*			raw_lca_ptr;
	unsigned int*	raw_chn_ptr;
	unsigned int*	raw_occ_ptr;
	unsigned int*	raw_vc1_ptr;
	unsigned int*	raw_vc2_ptr;
	unsigned int*	raw_vc3_ptr;
	unsigned int*	raw_flags_ptr;
	int*			raw_drift_ptr;

	float			f_deg_inc;	
	int				swilc;

	fix_agent_collisions3(	unsigned int	_swilc,
							float*			_rlp,
							unsigned int*	_rcp,
							unsigned int*	_rop,
							unsigned int*	_rvc1p,
							unsigned int*	_rvc2p,
							unsigned int*	_rvc3p,
							unsigned int*	_rfp,
							int*			_rdp	) : scene_width_in_lca_cells	(	_swilc			),
														raw_lca_ptr					(	_rlp			),
														raw_chn_ptr					(	_rcp			),
														raw_occ_ptr					(	_rop			),
														raw_vc1_ptr					(	_rvc1p			),
														raw_vc2_ptr					(	_rvc2p			),
														raw_vc3_ptr					(	_rvc3p			),
														raw_flags_ptr				(	_rfp			),
														raw_drift_ptr				(	_rdp			)
	{
		total_lca_cells			= scene_width_in_lca_cells * scene_width_in_lca_cells;
		f_deg_inc				= 360.0f / (float)DIRECTIONS;
		swilc					= (int)scene_width_in_lca_cells;
		lca_row					= 0;
		lca_col					= 0;
		lca_cell_id				= 0;
		tlcd					= total_lca_cells * DIRECTIONS;
	}

	template <typename Tuple>						//0
	__host__ __device__ void operator()( Tuple t )	//CELL_ID
	{
		lca_cell_id							= thrust::get<0>(t);
		unsigned int	neighbor_cell_id	= total_lca_cells;
		unsigned int	nda					= 0;
		unsigned int	a					= 0;
		unsigned int	d					= 0;
		unsigned int	lca_idx				= 0;
		int				test_neighbor_col	=-1;
		int				test_neighbor_row	=-1;
		float			test_radians		= 0.0f;
		bool			pulled				= false;
		lca_row								= (int)(lca_cell_id / scene_width_in_lca_cells);
		lca_col								= (int)(lca_cell_id % scene_width_in_lca_cells);

		for( d = 0; d < DIRECTIONS; d++ )	//FOR AGENTS INCOMING
		{
			test_radians		= DEG2RAD * (float)d * f_deg_inc;
			test_neighbor_col	= (int)lca_col + ((int)SIGNF(  cosf( test_radians ) ));
			test_neighbor_row	= (int)lca_row + ((int)SIGNF( -sinf( test_radians ) ));
			if( INBOUNDS( test_neighbor_col, swilc ) && INBOUNDS( test_neighbor_row, swilc ) )
			{
				neighbor_cell_id = (unsigned int)(test_neighbor_row * swilc + test_neighbor_col);
				for( a = 0; a < DIRECTIONS; a++ )
				{
					nda = neighbor_cell_id * DIRECTIONS + a;
					if( raw_vc3_ptr[nda] == tlcd ) break;
					if( raw_vc3_ptr[nda] == lca_cell_id )	//INCOMING_HERE, SECOND CHOICE
					{
						lca_idx = raw_chn_ptr[nda] * total_lca_cells + lca_cell_id;
						if( raw_occ_ptr[lca_cell_id]	== 0				&& 
							raw_flags_ptr[nda]			== 0				&&
							raw_drift_ptr[nda]			== MAX_DRIFT - 1	&&
							raw_lca_ptr[lca_idx]		!= (float)DIRECTIONS	)
						{
							raw_flags_ptr[nda] = 3;			//GO3_FLAG
							pulled = true;
							break;
						}
					}
				}
			}
			if( pulled ) break;
		}

		if( !pulled )
		{
			for( d = 0; d < DIRECTIONS; d++ )	//FOR AGENTS INCOMING
			{
				test_radians		= DEG2RAD * (float)d * f_deg_inc;
				test_neighbor_col	= (int)lca_col + ((int)SIGNF(  cosf( test_radians ) ));
				test_neighbor_row	= (int)lca_row + ((int)SIGNF( -sinf( test_radians ) ));
				if( INBOUNDS( test_neighbor_col, swilc ) && INBOUNDS( test_neighbor_row, swilc ) )
				{
					neighbor_cell_id = (unsigned int)(test_neighbor_row * swilc + test_neighbor_col);
					for( a = 0; a < DIRECTIONS; a++ )
					{
						nda = neighbor_cell_id * DIRECTIONS + a;
						if( raw_vc3_ptr[nda] == tlcd ) break;
						if( raw_vc3_ptr[nda] == lca_cell_id )	//INCOMING_HERE, THIRD CHOICE
						{
							lca_idx = raw_chn_ptr[nda] * total_lca_cells + lca_cell_id;
							if( raw_occ_ptr[lca_cell_id]	== 0				&& 
								raw_flags_ptr[nda]			== 0				&&
								raw_drift_ptr[nda]			== 0				&&
								raw_lca_ptr[lca_idx]		!= (float)DIRECTIONS	)
							{
								raw_flags_ptr[nda] = 3;			//GO3_FLAG
								raw_drift_ptr[nda] = MAX_DRIFT;
								pulled = true;
								break;
							}
						}
					}
				}
				if( pulled ) break;
			}
		}
	}
};
//
//=======================================================================================
//
void fix_collisions	(	thrust::device_vector<float>&			lca,
						thrust::device_vector<unsigned int>&	agents_channel,
						thrust::device_vector<unsigned int>&	cells_ids,
						thrust::device_vector<unsigned int>&	occupancy,
						thrust::device_vector<unsigned int>&	agents_virt_cell1,
						thrust::device_vector<unsigned int>&	agents_virt_cell2,
						thrust::device_vector<unsigned int>&	agents_virt_cell3,
						thrust::device_vector<unsigned int>&	agents_flags,
						thrust::device_vector<unsigned int>&	agents_flags_copy,
						thrust::device_vector<int>&				agents_drift_counter,
						unsigned int							scene_width_in_lca_cells	)
{
	unsigned int total_lca_cells			= scene_width_in_lca_cells * scene_width_in_lca_cells;
	thrust::copy( agents_flags.begin(), agents_flags.end(), agents_flags_copy.begin() );
	thrust::fill( agents_flags.begin(), agents_flags.end(), total_lca_cells * DIRECTIONS );
	float*			raw_lca_ptr				= thrust::raw_pointer_cast( lca.data()					);
	unsigned int*	raw_chn_ptr				= thrust::raw_pointer_cast( agents_channel.data()		);
	unsigned int*	raw_occupancy_ptr		= thrust::raw_pointer_cast( occupancy.data()			);
	unsigned int*	raw_vc1_ptr				= thrust::raw_pointer_cast( agents_virt_cell1.data()	);
	unsigned int*	raw_vc2_ptr				= thrust::raw_pointer_cast( agents_virt_cell2.data()	);
	unsigned int*	raw_vc3_ptr				= thrust::raw_pointer_cast( agents_virt_cell3.data()	);
	unsigned int*	raw_flags_ptr			= thrust::raw_pointer_cast( agents_flags.data()			);
	unsigned int*	raw_flags_copy_ptr		= thrust::raw_pointer_cast( agents_flags_copy.data()	);
	int*			raw_drift_ptr			= thrust::raw_pointer_cast( agents_drift_counter.data() );
	// DCELL_ID, AGENT_ID, AGENT_GOAL_CELL, AGENT_VIRT_CELL, ISEED
	thrust::for_each(	
		thrust::make_zip_iterator(
			thrust::make_tuple(
				cells_ids.begin()
			)
		),
		thrust::make_zip_iterator(
			thrust::make_tuple(
				cells_ids.end()
			)
		),
		fix_agent_collisions1(	scene_width_in_lca_cells,
								raw_occupancy_ptr,
								raw_vc1_ptr,
								raw_vc2_ptr,
								raw_vc3_ptr,
								raw_flags_ptr,
								raw_flags_copy_ptr,
								raw_drift_ptr			)
	);
	
	thrust::for_each(	
		thrust::make_zip_iterator(
			thrust::make_tuple(
				cells_ids.begin()
			)
		),
		thrust::make_zip_iterator(
			thrust::make_tuple(
				cells_ids.end()
			)
		),
		fix_agent_collisions2(	scene_width_in_lca_cells,
								raw_lca_ptr,
								raw_chn_ptr,
								raw_occupancy_ptr,
								raw_vc1_ptr,
								raw_vc2_ptr,
								raw_vc3_ptr,
								raw_flags_ptr,
								raw_drift_ptr			)
	);

	thrust::for_each(	
		thrust::make_zip_iterator(
			thrust::make_tuple(
				cells_ids.begin()
			)
		),
		thrust::make_zip_iterator(
			thrust::make_tuple(
				cells_ids.end()
			)
		),
		fix_agent_collisions3(	scene_width_in_lca_cells,
								raw_lca_ptr,
								raw_chn_ptr,
								raw_occupancy_ptr,
								raw_vc1_ptr,
								raw_vc2_ptr,
								raw_vc3_ptr,
								raw_flags_ptr,
								raw_drift_ptr			)
	);

}
//
//=======================================================================================
//

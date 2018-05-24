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

struct advance_n_agents_virtually
{
	unsigned int	scene_width_in_lca_cells;
	unsigned int	total_lca_cells;
	unsigned int	tlcd;
	float			f_deg_inc;
	unsigned int*	raw_agents_ids_ptr;
	unsigned int*	raw_agents_ids_copy_ptr;
	unsigned int*	raw_vc1_ptr;
	unsigned int*	raw_vc2_ptr;
	unsigned int*	raw_vc3_ptr;
	unsigned int*	raw_goal_cells_ptr;

	advance_n_agents_virtually(	unsigned int	_swilc,
								unsigned int*	_raip,
								unsigned int*	_raicp,
								unsigned int*	_rvc1p,
								unsigned int*	_rvc2p,
								unsigned int*	_rvc3p,
								unsigned int*	_rgcp	) : scene_width_in_lca_cells(	_swilc	),
															raw_agents_ids_ptr		(	_raip	),
															raw_agents_ids_copy_ptr	(	_raicp	),
															raw_vc1_ptr				(	_rvc1p	),
															raw_vc2_ptr				(	_rvc2p	),
															raw_vc3_ptr				(	_rvc3p	),
															raw_goal_cells_ptr		(	_rgcp	)
	{
		total_lca_cells = scene_width_in_lca_cells * scene_width_in_lca_cells;
		tlcd			= total_lca_cells * DIRECTIONS;
		f_deg_inc		= 360.0f / (float)DIRECTIONS;
	}

	template <typename Tuple>						//0			   1
	__host__ __device__ void operator()( Tuple t )	//LCA_CELL_ID, OCCUPANCY
	{
		unsigned int	lca_cell_id				= thrust::get<0>(t);
		unsigned int	neighbor_lca_cell_id	= total_lca_cells;
		unsigned int	offset					= 0;
		unsigned int	occupancy				= 0;
		unsigned int	cda						= 0;
		unsigned int	nda						= 0;
		unsigned int	cdo						= 0;
		unsigned int	d						= 0;
		unsigned int	a						= 0;
		int				lca_row					= (int)(lca_cell_id / scene_width_in_lca_cells);
		int				lca_col					= (int)(lca_cell_id % scene_width_in_lca_cells);
		int				test_neighbor_lca_col	= -1;
		int				test_neighbor_lca_row	= -1;
		int				swilc					= (int)scene_width_in_lca_cells;
		float			test_radians			= 0.0f;

		for( a = 0; a < DIRECTIONS; a++ )	//STAYING
		{
			cda = lca_cell_id * DIRECTIONS + a;
			if( raw_vc1_ptr[cda] == tlcd ) break;
			if( raw_vc1_ptr[cda] == lca_cell_id )
			{
				cdo = lca_cell_id * DIRECTIONS + offset;
				raw_agents_ids_ptr[cdo] = raw_agents_ids_copy_ptr[cda];
				if( lca_cell_id != raw_goal_cells_ptr[cda] ) occupancy++;
				offset++;
			}
		}

		for( d = 0; d < DIRECTIONS; d++ )	//INCOMING
		{
			test_radians			= DEG2RAD * ((float)d * f_deg_inc);
			test_neighbor_lca_col	= lca_col + ((int)SIGNF(  cosf( test_radians ) ));
			test_neighbor_lca_row	= lca_row + ((int)SIGNF( -sinf( test_radians ) ));
			if( INBOUNDS( test_neighbor_lca_col, swilc ) && INBOUNDS( test_neighbor_lca_row, swilc ) )
			{
				neighbor_lca_cell_id = (unsigned int)(test_neighbor_lca_row * swilc + test_neighbor_lca_col);
				for( a = 0; a < DIRECTIONS; a++ )
				{
					nda = neighbor_lca_cell_id * DIRECTIONS + a;
					if( raw_vc1_ptr[nda] == tlcd ) break;
					if( raw_vc1_ptr[nda] == lca_cell_id )
					{
						cdo						= lca_cell_id * DIRECTIONS + offset;
						raw_agents_ids_ptr[cdo] = raw_agents_ids_copy_ptr[nda];
						if( lca_cell_id != raw_goal_cells_ptr[nda] ) occupancy++;
						offset++;
					}
				}
			}
		}
		thrust::get<1>(t) = occupancy;
	}
};
//
//=======================================================================================
//
void advance_agents_virtually(	thrust::device_vector<unsigned int>&	lca_cells_ids,
								thrust::device_vector<unsigned int>&	occupancy,
								thrust::device_vector<unsigned int>&	agents_ids,
								thrust::device_vector<unsigned int>&	agents_temp_ids,
								thrust::device_vector<unsigned int>&	agents_virt_cell1,
								thrust::device_vector<unsigned int>&	agents_virt_cell2,
								thrust::device_vector<unsigned int>&	agents_virt_cell3,
								thrust::device_vector<unsigned int>&	agents_goal_cell,
								unsigned int							scene_width_in_lca_cells	)

{
	unsigned int total_lca_cells			= scene_width_in_lca_cells * scene_width_in_lca_cells;
	thrust::copy( agents_ids.begin(), agents_ids.end(), agents_temp_ids.begin() );
	thrust::fill( agents_ids.begin(), agents_ids.end(), total_lca_cells * DIRECTIONS );
	unsigned int* raw_agents_ids_ptr		= thrust::raw_pointer_cast( agents_ids.data()			);
	unsigned int* raw_agents_ids_copy_ptr	= thrust::raw_pointer_cast( agents_temp_ids.data()		);
	unsigned int* raw_vc1_ptr				= thrust::raw_pointer_cast( agents_virt_cell1.data()	);
	unsigned int* raw_vc2_ptr				= thrust::raw_pointer_cast( agents_virt_cell2.data()	);
	unsigned int* raw_vc3_ptr				= thrust::raw_pointer_cast( agents_virt_cell3.data()	);
	unsigned int* raw_goal_cells_ptr		= thrust::raw_pointer_cast( agents_goal_cell.data()		);
	thrust::for_each(
		thrust::make_zip_iterator(
			thrust::make_tuple(
				lca_cells_ids.begin(),
				occupancy.begin()
			)
		),
		thrust::make_zip_iterator(
			thrust::make_tuple(
				lca_cells_ids.end(),
				occupancy.end()
			)
		),
		advance_n_agents_virtually(	scene_width_in_lca_cells, 
									raw_agents_ids_ptr,
									raw_agents_ids_copy_ptr,
									raw_vc1_ptr,
									raw_vc2_ptr,
									raw_vc3_ptr,
									raw_goal_cells_ptr		)
	);
}
//
//=======================================================================================

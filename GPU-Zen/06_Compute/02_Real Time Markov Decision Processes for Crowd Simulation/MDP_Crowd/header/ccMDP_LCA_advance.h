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
struct advance_n_agents2
{
	unsigned int	scene_width_in_lca_cells;
	unsigned int	scene_width_in_mdp_cells;
	unsigned int	lca_mdp_width_ratio;
	unsigned int	lca_cells_per_mdp_cell;
	unsigned int	total_lca_cells;
	unsigned int	tlcd;									//TOTAL_LCA_CELLS * DIRECTIONS
	unsigned int	lca_row;
	unsigned int	lca_col;
	unsigned int	lca_cell_id;
	unsigned int	channel;

	unsigned int*	raw_agents_ids_ptr;
	unsigned int*	raw_agents_ids_copy_ptr;
	unsigned int*	raw_goal_cells_ptr;
	unsigned int*	raw_goal_cells_copy_ptr;
	unsigned int*	raw_vc1_ptr;
	unsigned int*	raw_vc1_copy_ptr;
	unsigned int*	raw_vc2_ptr;
	unsigned int*	raw_vc2_copy_ptr;
	unsigned int*	raw_vc3_ptr;
	unsigned int*	raw_vc3_copy_ptr;
	unsigned int*	raw_chn_ptr;
	unsigned int*	raw_chn_copy_ptr;
	unsigned int*	raw_flags_ptr;
	unsigned int*	raw_flags_copy_ptr;

	int*			raw_drift_ptr;
	int*			raw_drift_copy_ptr;
	
	float*			raw_disp_param_ptr;
	float*			raw_disp_param_copy_ptr;
	float*			raw_disp_delta_ptr;
	float*			raw_mdp_speeds_ptr;

	float2*			raw_agents_xy_ptr;
	float2*			raw_agents_xy_copy_ptr;

	float*			raw_agents_pos_ptr;
	float*			raw_lca_ptr;

	float			lca_cell_width;
	float			f_hcw;
	float			f_hsw;
	float			f_deg_inc;
	
	int				swilc;
	int				swimc;
	int				gx;
	int				gz;

	advance_n_agents2(	float			_lca_cell_width,
						unsigned int	_swilc,
						unsigned int	_swimc,
						unsigned int*	_raip,
						unsigned int*	_raicp,
						unsigned int*	_rgcp,
						unsigned int*	_rgccp,
						unsigned int*	_rvc1p,
						unsigned int*	_rvc1cp,
						unsigned int*	_rvc2p,
						unsigned int*	_rvc2cp,
						unsigned int*	_rvc3p,
						unsigned int*	_rvc3cp,
						unsigned int*	_rchnp,
						unsigned int*	_rchncp,
						unsigned int*	_rfp,
						unsigned int*	_rfcp,
						int*			_rdp,
						int*			_rdcp,
						float*			_rdpp,
						float*			_rdpcp,
						float2*			_raxyp,
						float2*			_raxycp,
						float*			_rapp,
						float*			_rlcap,
						float*			_rddp,
						float*			_rmsp	) : lca_cell_width				(	_lca_cell_width	),
													scene_width_in_lca_cells	(	_swilc			),
													scene_width_in_mdp_cells	(	_swimc			),
													raw_agents_ids_ptr			(	_raip			),
													raw_agents_ids_copy_ptr		(	_raicp			),
													raw_goal_cells_ptr			(	_rgcp			),
													raw_goal_cells_copy_ptr		(	_rgccp			),
													raw_vc1_ptr					(	_rvc1p			),
													raw_vc1_copy_ptr			(	_rvc1cp			),
													raw_vc2_ptr					(	_rvc2p			),
													raw_vc2_copy_ptr			(	_rvc2cp			),
													raw_vc3_ptr					(	_rvc3p			),
													raw_vc3_copy_ptr			(	_rvc3cp			),
													raw_chn_ptr					(	_rchnp			),
													raw_chn_copy_ptr			(	_rchncp			),
													raw_flags_ptr				(	_rfp			),
													raw_flags_copy_ptr			(	_rfcp			),
													raw_drift_ptr				(	_rdp			),
													raw_drift_copy_ptr			(	_rdcp			),
													raw_disp_param_ptr			(	_rdpp			),
													raw_disp_param_copy_ptr		(	_rdpcp			),
													raw_agents_xy_ptr			(	_raxyp			),
													raw_agents_xy_copy_ptr		(	_raxycp			),
													raw_agents_pos_ptr			(	_rapp			),
													raw_lca_ptr					(	_rlcap			),
													raw_disp_delta_ptr			(	_rddp			),
													raw_mdp_speeds_ptr			(	_rmsp			)
	{
		lca_mdp_width_ratio		= scene_width_in_lca_cells / scene_width_in_mdp_cells;
		lca_cells_per_mdp_cell	= lca_mdp_width_ratio * lca_mdp_width_ratio;
		total_lca_cells			= scene_width_in_lca_cells * scene_width_in_lca_cells;
		tlcd					= total_lca_cells * DIRECTIONS;
		f_deg_inc				= 360.0f / (float)DIRECTIONS;
		swilc					= (int)scene_width_in_lca_cells;
		swimc					= (int)scene_width_in_mdp_cells;
		f_hsw					= ((float)scene_width_in_lca_cells * lca_cell_width) / 2.0f;
		f_hcw					= lca_cell_width / 2.0f;
		gx						=-1;
		gz						=-1;
		lca_row					= 0;
		lca_col					= 0;
		lca_cell_id				= 0;
		channel					= 0;
	}

	template <typename Tuple>						//0        1
	__host__ __device__ void operator()( Tuple t )	//CELL_ID, OCCUPANCY
	{
		lca_cell_id							= thrust::get<0>(t);
		unsigned int	neighbor_cell_id	= total_lca_cells;
		unsigned int	offset				= 0;
		unsigned int	cda					= 0;
		unsigned int	nda					= 0;
		unsigned int	cd					= lca_cell_id * DIRECTIONS;
		unsigned int	cdo					= cd + offset;
		unsigned int	a					= 0;
		unsigned int	d					= 0;
		unsigned int	vcell				= 0;
		unsigned int	flag				= 0;
		int				test_neighbor_col	=-1;
		int				test_neighbor_row	=-1;
		float			test_radians		= 0.0f;
		float			param				= 0.0f;
		lca_row								= lca_cell_id / scene_width_in_lca_cells;
		lca_col								= lca_cell_id % scene_width_in_lca_cells;

		for( a = 0; a < DIRECTIONS; a++ )	//FOR_LOCAL_AGENTS
		{
			cda		= cd + a;
			param	= raw_disp_param_copy_ptr[cda];
			flag	= raw_flags_copy_ptr[cda];
			vcell	= tlcd;
			if( flag == 1 )			vcell = raw_vc1_copy_ptr[cda];
			else if( flag == 2 )	vcell = raw_vc2_copy_ptr[cda];
			else if( flag == 3 )	vcell = raw_vc3_copy_ptr[cda];

			if( raw_agents_ids_copy_ptr[cda] == tlcd )	break;
			if( flag == 0 )		//STAYING
			{
				raw_agents_ids_ptr[cdo]		= raw_agents_ids_copy_ptr[cda];
				raw_agents_xy_ptr[cdo]		= raw_agents_xy_copy_ptr[cda];
				raw_disp_param_ptr[cdo]		= param;
				raw_drift_ptr[cdo]			= raw_drift_copy_ptr[cda];
				raw_flags_ptr[cdo]			= raw_flags_copy_ptr[cda];
				raw_chn_ptr[cdo]			= raw_chn_copy_ptr[cda];

				raw_goal_cells_ptr[cdo]		= raw_goal_cells_copy_ptr[cda];
				raw_vc1_ptr[cdo]			= raw_vc1_copy_ptr[cda];
				raw_vc2_ptr[cdo]			= raw_vc2_copy_ptr[cda];
				raw_vc3_ptr[cdo]			= raw_vc3_copy_ptr[cda];

				// raw_agents_pos_ptr   remains the same.
				offset++;
				cdo = cd + offset;
			}
			else if( param < 1.0f )		// IN_TRANSIT (SCATTER)
			{
				if( raw_drift_copy_ptr[cda] == MAX_DRIFT )		// JUST_CHANGED_NEXT_CELL
				{
					unsigned int	aid				= raw_agents_ids_copy_ptr[cda] * 4;
					
					/*
					bool			walks_over_me	= false;
					unsigned int	agent_lca_col	= (unsigned int)(floor((raw_agents_pos_ptr[aid + 0] + f_hsw) / lca_cell_width));
					unsigned int	agent_lca_row	= (unsigned int)(floor((raw_agents_pos_ptr[aid + 2] + f_hsw) / lca_cell_width));
					if( agent_lca_col == lca_col && agent_lca_row == lca_row ) walks_over_me = true;
					

					/*
					bool			closer_to_me	= false;
					float			apx				= (raw_agents_pos_ptr[aid + 0] + f_hsw);
					float			apz				= (raw_agents_pos_ptr[aid + 2] + f_hsw);
					float			c1px			= ((float)lca_col * lca_cell_width) + f_hcw;
					float			c1pz			= ((float)lca_row * lca_cell_width) + f_hcw;
					unsigned int	lca2_row		= vcell / scene_width_in_lca_cells;
					unsigned int	lca2_col		= vcell % scene_width_in_lca_cells;
					float			c2px			= ((float)lca2_col * lca_cell_width) + f_hcw;
					float			c2pz			= ((float)lca2_row * lca_cell_width) + f_hcw;
					float d12						= (apx-c1px)*(apx-c1px) + (apz-c1pz)*(apz-c1pz);
					float d22						= (apx-c2px)*(apx-c2px) + (apz-c2pz)*(apz-c2pz);
					if( d12 < d22 ) closer_to_me	= true;
					*/

					//if( walks_over_me )
					{
						unsigned int mdp_row		= lca_row / lca_mdp_width_ratio;
						unsigned int mdp_col		= lca_col / lca_mdp_width_ratio;
						unsigned int mdp_idx		= mdp_row * scene_width_in_mdp_cells + mdp_col;
						float mdp_speed				= raw_mdp_speeds_ptr[mdp_idx];

						raw_agents_ids_ptr[cdo]		= raw_agents_ids_copy_ptr[cda];
						raw_goal_cells_ptr[cdo]		= raw_goal_cells_copy_ptr[cda];
						raw_vc1_ptr[cdo]			= raw_vc1_copy_ptr[cda];
						raw_vc2_ptr[cdo]			= raw_vc2_copy_ptr[cda];
						raw_vc3_ptr[cdo]			= raw_vc3_copy_ptr[cda];
						raw_drift_ptr[cdo]			= MAX_DRIFT - 1;
						raw_drift_copy_ptr[cda]		= MAX_DRIFT - 1;
						raw_flags_ptr[cdo]			= raw_flags_copy_ptr[cda];
						channel						= raw_chn_copy_ptr[cda];
						raw_chn_ptr[cdo]			= channel;

						float new_param				= 0.0f;
						new_param					= mdp_speed * raw_disp_delta_ptr[raw_agents_ids_copy_ptr[cda]];
						raw_disp_param_ptr[cdo]		= new_param;

						float			apx			= ((float)lca_col * lca_cell_width);
						float			apz			= ((float)lca_row * lca_cell_width);
						float			adx			= raw_agents_pos_ptr[aid + 0] + f_hsw - apx;
						float			adz			= raw_agents_pos_ptr[aid + 2] + f_hsw - apz;
						raw_agents_xy_ptr[cdo].x	= adx;
						raw_agents_xy_ptr[cdo].y	= adz;

						float			ax1			= (((float)lca_col * lca_cell_width) + adx) - f_hsw;
						float			az1			= (((float)lca_row * lca_cell_width) + adz) - f_hsw;
						unsigned int	lca_row2	= vcell / scene_width_in_lca_cells;
						unsigned int	lca_col2	= vcell % scene_width_in_lca_cells;
						float			ax2			= (((float)lca_col2 * lca_cell_width) + adx) - f_hsw;
						float			az2			= (((float)lca_row2 * lca_cell_width) + adz) - f_hsw;
						// P(t) = A + t(B - A):
						float			dX			= ax2 - ax1;
						float			dZ			= az2 - az1;
						float			ax			= ax1 + new_param * dX;
						float			az			= az1 + new_param * dZ;
						// ESTIMATE ORIENTATION:
						float			dirX		= SIGNF( dX );
						float			dirZ		= SIGNF( dZ );
						float			radAngle	= atan2f( dirZ, dirX );
						float			degAngle	= 90.0f - (RAD2DEG * radAngle);

						raw_agents_pos_ptr[aid + 0]	= ax;
						raw_agents_pos_ptr[aid + 2] = az;
						raw_agents_pos_ptr[aid + 3] = degAngle * DEG2RAD;

						offset++;
						cdo = cd + offset;
					}
				}
				else				// KEEP_ADVANCING
				{
					unsigned int mdp_row		= lca_row / lca_mdp_width_ratio;
					unsigned int mdp_col		= lca_col / lca_mdp_width_ratio;
					unsigned int mdp_idx		= mdp_row * scene_width_in_mdp_cells + mdp_col;
					float mdp_speed				= raw_mdp_speeds_ptr[mdp_idx];

					raw_agents_ids_ptr[cdo]		= raw_agents_ids_copy_ptr[cda];
					raw_agents_xy_ptr[cdo]		= raw_agents_xy_copy_ptr[cda];
					raw_goal_cells_ptr[cdo]		= raw_goal_cells_copy_ptr[cda];
					raw_vc1_ptr[cdo]			= raw_vc1_copy_ptr[cda];
					raw_vc2_ptr[cdo]			= raw_vc2_copy_ptr[cda];
					raw_vc3_ptr[cdo]			= raw_vc3_copy_ptr[cda];
					raw_chn_ptr[cdo]			= raw_chn_copy_ptr[cda];
					raw_drift_ptr[cdo]			= raw_drift_copy_ptr[cda];
					raw_flags_ptr[cdo]			= raw_flags_copy_ptr[cda];

					float new_param				= 0.0f;
					new_param					= param + mdp_speed * raw_disp_delta_ptr[raw_agents_ids_copy_ptr[cda]];
					raw_disp_param_ptr[cdo]		= new_param;

					float			adx			= raw_agents_xy_copy_ptr[cda].x;
					float			adz			= raw_agents_xy_copy_ptr[cda].y;
					float			ax1			= (((float)lca_col * lca_cell_width) + adx) - f_hsw;
					float			az1			= (((float)lca_row * lca_cell_width) + adz) - f_hsw;
					unsigned int	lca_row2	= vcell / scene_width_in_lca_cells;
					unsigned int	lca_col2	= vcell % scene_width_in_lca_cells;
					float			ax2			= (((float)lca_col2 * lca_cell_width) + adx) - f_hsw;
					float			az2			= (((float)lca_row2 * lca_cell_width) + adz) - f_hsw;
					// P(t) = A + t(B - A):
					float			dX			= ax2 - ax1;
					float			dZ			= az2 - az1;
					float			ax			= ax1 + new_param * dX;
					float			az			= az1 + new_param * dZ;
					// ESTIMATE ORIENTATION:
					float			dirX		= SIGNF( dX );
					float			dirZ		= SIGNF( dZ );
					float			radAngle	= atan2f( dirZ, dirX );
					float			degAngle	= 90.0f - (RAD2DEG * radAngle);

					unsigned int	aid			= raw_agents_ids_copy_ptr[cda] * 4;
					raw_agents_pos_ptr[aid + 0]	= ax;
					raw_agents_pos_ptr[aid + 2] = az;
					raw_agents_pos_ptr[aid + 3] = degAngle * DEG2RAD;

					offset++;
					cdo = cd + offset;
				}
			}
		}

		for( d = 0; d < DIRECTIONS; d++ )	//FOR_NEIGHBORING_AGENTS (GATHER)
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
					if( raw_agents_ids_copy_ptr[nda] == tlcd )	break;

					param	= raw_disp_param_copy_ptr[nda];
					flag	= raw_flags_copy_ptr[nda];
					vcell	= tlcd;
					if( flag == 1 )			vcell = raw_vc1_copy_ptr[nda];
					else if( flag == 2 )	vcell = raw_vc2_copy_ptr[nda];
					else if( flag == 3 )	vcell = raw_vc3_copy_ptr[nda];

					unsigned int	aid				= raw_agents_ids_copy_ptr[nda] * 4;
					//float			apx				= (raw_agents_pos_ptr[aid + 0] + f_hsw);
					//float			apz				= (raw_agents_pos_ptr[aid + 2] + f_hsw);

					/*
					bool			walks_over_me	= false;
					unsigned int	agent_lca_col	= (unsigned int)(floor(apx / lca_cell_width));
					unsigned int	agent_lca_row	= (unsigned int)(floor(apz / lca_cell_width));
					if( agent_lca_col == lca_col && agent_lca_row == lca_row ) walks_over_me = true;
					*/

					/*
					bool			closer_to_me	= false;
					float			c1px			= ((float)agent_lca_col * lca_cell_width) + f_hcw;
					float			c1pz			= ((float)agent_lca_row * lca_cell_width) + f_hcw;
					float			c2px			= ((float)lca_col * lca_cell_width) + f_hcw;
					float			c2pz			= ((float)lca_row * lca_cell_width) + f_hcw;
					float			d12				= (apx-c1px)*(apx-c1px) + (apz-c1pz)*(apz-c1pz);
					float			d22				= (apx-c2px)*(apx-c2px) + (apz-c2pz)*(apz-c2pz);
					if( d22 <= d12 ) closer_to_me	= true;
					*/

					if( vcell == lca_cell_id && flag > 0 && param >= 1.0f )	//INCOMING_HERE, PARAM_EXPIRED
					{
						raw_agents_ids_ptr[cdo]		= raw_agents_ids_copy_ptr[nda];
						raw_disp_param_ptr[cdo]		= 0.0f;
						channel						= raw_chn_copy_ptr[nda];
						raw_chn_ptr[cdo]			= channel;
						raw_flags_ptr[cdo]			= raw_flags_copy_ptr[nda];

						float			apx			= ((float)lca_col * lca_cell_width);
						float			apz			= ((float)lca_row * lca_cell_width);
						float			adx			= raw_agents_pos_ptr[aid + 0]  + f_hsw - apx;
						float			adz			= raw_agents_pos_ptr[aid + 2]  + f_hsw - apz;
						raw_agents_xy_ptr[cdo].x	= adx;
						raw_agents_xy_ptr[cdo].y	= adz;

						raw_drift_ptr[cdo] = 0;
						//GET_NEW_GOAL:
						if( raw_lca_ptr[channel * total_lca_cells + lca_cell_id] < DIRECTIONS )		//THIS_IS_GOAL
						{
							nearest_exit_from_goal();
						}
						else
						{
							nearest_exit();
						}

						if( gx > -1 && gz > -1 )
						{
							raw_goal_cells_ptr[cdo]	= (unsigned int)(gz * swilc + gx);
						}
						else
						{
							gx						= (int)(raw_goal_cells_copy_ptr[nda] % scene_width_in_lca_cells);
							gz						= (int)(raw_goal_cells_copy_ptr[nda] / scene_width_in_lca_cells);
							raw_goal_cells_ptr[cdo] = raw_goal_cells_copy_ptr[nda];
						}
						//GET_NEW_ALTERNATIVES:
						int				diffX		= (gx - (int)lca_col);
						int				ncx			= diffX;
						if( ncx != 0 )	ncx			= diffX / abs( diffX );
						int				diffZ		= (gz - (int)lca_row);
						int				ncz			= diffZ;
						if( ncz != 0 )	ncz			= diffZ / abs( diffZ );
						unsigned int	ntx			= lca_col + (unsigned int)ncx;
						unsigned int	ntz			= lca_row + (unsigned int)ncz;
						raw_vc1_ptr[cdo]			= ntz * scene_width_in_lca_cells + ntx;

						unsigned int	niA			= tlcd;
						unsigned int	niB			= tlcd;
						if( abs(ncx) > 0 && abs(ncz) > 0 )
						{
							if( INBOUNDS( (int)ntx, swilc ) )	niA	= lca_row * scene_width_in_lca_cells + ntx;
							if( INBOUNDS( (int)ntz, swilc ) )	niB	= ntz * scene_width_in_lca_cells + lca_col;
						}
						else if( abs(ncx) > 0 && INBOUNDS( (int)ntx, swilc ) )
						{
							if( (int)lca_row > 0 )				niA	= ((int)lca_row - 1) * scene_width_in_lca_cells + ntx;
							if( ((int)lca_row + 1) < swilc )	niB	= ((int)lca_row + 1) * scene_width_in_lca_cells + ntx;
						}
						else if( INBOUNDS( (int)ntz, swilc ) )
						{
							if( (int)lca_col > 0 )				niA	= ntz * scene_width_in_lca_cells + ((int)lca_col - 1);
							if( ((int)lca_col + 1) < swilc )	niB	= ntz * scene_width_in_lca_cells + ((int)lca_col + 1);
						}
						raw_vc2_ptr[cdo]			= niA;
						raw_vc3_ptr[cdo]			= niB;

						offset++;
						cdo = cd + offset;
					}
					/*
					else if( raw_drift_copy_ptr[nda] == MAX_DRIFT && walks_over_me )
					{
						raw_agents_ids_ptr[cdo]		= raw_agents_ids_copy_ptr[nda];
						raw_disp_param_ptr[cdo]		= 0.0f;
						channel						= raw_chn_copy_ptr[nda];
						raw_chn_ptr[cdo]			= channel;
						raw_flags_ptr[cdo]			= raw_flags_copy_ptr[nda];

						float			apx			= ((float)lca_col * lca_cell_width);
						float			apz			= ((float)lca_row * lca_cell_width);
						float			adx			= raw_agents_pos_ptr[aid + 0]  + f_hsw - apx;
						float			adz			= raw_agents_pos_ptr[aid + 2]  + f_hsw - apz;
						raw_agents_xy_ptr[cdo].x	= adx;
						raw_agents_xy_ptr[cdo].y	= adz;

						raw_drift_ptr[cdo]		= MAX_DRIFT - 1;
						raw_goal_cells_ptr[cdo]	= raw_goal_cells_copy_ptr[nda];
						raw_vc1_ptr[cdo]		= raw_vc1_copy_ptr[nda];
						raw_vc2_ptr[cdo]		= raw_vc2_copy_ptr[nda];
						raw_vc3_ptr[cdo]		= raw_vc3_copy_ptr[nda];

						offset++;
						cdo = cd + offset;
					}
					*/
				}
			}
		}

		thrust::get<1>(t) = offset;
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
					if( raw_lca_ptr[channel * total_lca_cells +i] < (float)DIRECTIONS )
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
struct advance_n_agents
{
	unsigned int	scene_width_in_lca_cells;
	unsigned int	scene_width_in_mdp_cells;
	unsigned int	lca_mdp_width_ratio;
	unsigned int	lca_cells_per_mdp_cell;
	unsigned int	total_lca_cells;
	unsigned int	tlcd;									//TOTAL_LCA_CELLS * DIRECTIONS
	unsigned int	lca_row;
	unsigned int	lca_col;
	unsigned int	lca_cell_id;

	unsigned int*	raw_agents_ids_ptr;
	unsigned int*	raw_agents_ids_copy_ptr;
	unsigned int*	raw_goal_cells_ptr;
	unsigned int*	raw_goal_cells_copy_ptr;
	unsigned int*	raw_vc1_ptr;
	unsigned int*	raw_vc1_copy_ptr;
	unsigned int*	raw_vc2_ptr;
	unsigned int*	raw_vc2_copy_ptr;
	unsigned int*	raw_vc3_ptr;
	unsigned int*	raw_vc3_copy_ptr;
	float*			raw_disp_param_ptr;
	float*			raw_disp_param_copy_ptr;
	float*			raw_disp_delta_ptr;
	unsigned int*	raw_flags_ptr;

	float2*			raw_agents_xy_ptr;
	float2*			raw_agents_xy_copy_ptr;

	float*			raw_agents_pos_ptr;
	float*			raw_lca_ptr;

	float			lca_cell_width;
	float			f_hsw;
	float			f_deg_inc;
	
	int				swilc;
	int				swimc;
	int				gx;
	int				gz;

	advance_n_agents(	float			_lca_cell_width,
						unsigned int	_swilc,
						unsigned int	_swimc,
						unsigned int*	_raip,
						unsigned int*	_raicp,
						unsigned int*	_rgcp,
						unsigned int*	_rgccp,
						unsigned int*	_rvc1p,
						unsigned int*	_rvc1cp,
						unsigned int*	_rvc2p,
						unsigned int*	_rvc2cp,
						unsigned int*	_rvc3p,
						unsigned int*	_rvc3cp,
						float*			_rdpp,
						float*			_rdpcp,
						unsigned int*	_rfp,
						float2*			_raxyp,
						float2*			_raxycp,
						float*			_rapp,
						float*			_rlcap,
						float*			_rddp	) : lca_cell_width				(	_lca_cell_width	),
													scene_width_in_lca_cells	(	_swilc			),
													scene_width_in_mdp_cells	(	_swimc			),
													raw_agents_ids_ptr			(	_raip			),
													raw_agents_ids_copy_ptr		(	_raicp			),
													raw_goal_cells_ptr			(	_rgcp			),
													raw_goal_cells_copy_ptr		(	_rgccp			),
													raw_vc1_ptr					(	_rvc1p			),
													raw_vc1_copy_ptr			(	_rvc1cp			),
													raw_vc2_ptr					(	_rvc2p			),
													raw_vc2_copy_ptr			(	_rvc2cp			),
													raw_vc3_ptr					(	_rvc3p			),
													raw_vc3_copy_ptr			(	_rvc3cp			),
													raw_disp_param_ptr			(	_rdpp			),
													raw_disp_param_copy_ptr		(	_rdpcp			),
													raw_flags_ptr				(	_rfp			),
													raw_agents_xy_ptr			(	_raxyp			),
													raw_agents_xy_copy_ptr		(	_raxycp			),
													raw_agents_pos_ptr			(	_rapp			),
													raw_lca_ptr					(	_rlcap			),
													raw_disp_delta_ptr			(	_rddp			)
	{
		lca_mdp_width_ratio		= scene_width_in_lca_cells / scene_width_in_mdp_cells;
		lca_cells_per_mdp_cell	= lca_mdp_width_ratio * lca_mdp_width_ratio;
		total_lca_cells			= scene_width_in_lca_cells * scene_width_in_lca_cells;
		tlcd					= total_lca_cells * DIRECTIONS;
		f_deg_inc				= 360.0f / (float)DIRECTIONS;
		swilc					= (int)scene_width_in_lca_cells;
		swimc					= (int)scene_width_in_mdp_cells;
		f_hsw					= ((float)scene_width_in_lca_cells * lca_cell_width) / 2.0f;
		gx						=-1;
		gz						=-1;
		lca_row					= 0;
		lca_col					= 0;
		lca_cell_id				= 0;
	}

	template <typename Tuple>						//0        1
	__host__ __device__ void operator()( Tuple t )	//CELL_ID, OCCUPANCY
	{
		lca_cell_id							= thrust::get<0>(t);
		unsigned int	neighbor_cell_id	= total_lca_cells;
		unsigned int	offset				= 0;
		unsigned int	cda					= 0;
		unsigned int	nda					= 0;
		unsigned int	cdo					= 0;
		unsigned int	cd					= lca_cell_id * DIRECTIONS;
		unsigned int	a					= 0;
		unsigned int	d					= 0;
		int				test_neighbor_col	=-1;
		int				test_neighbor_row	=-1;
		float			test_radians		= 0.0f;
		lca_row								= lca_cell_id / scene_width_in_lca_cells;
		lca_col								= lca_cell_id % scene_width_in_lca_cells;

		for( a = 0; a < DIRECTIONS; a++ )	//FOR AGENTS STAYING
		{
			cda = cd + a;
			if( raw_vc1_copy_ptr[cda] == tlcd )	break;

			if( raw_vc1_copy_ptr[cda] == lca_cell_id || raw_flags_ptr[cda] == 0 )	//ACTUALLY_STAYING
			{
				cdo							= cd + offset;
				raw_agents_ids_ptr[cdo]		= raw_agents_ids_copy_ptr[cda];
				raw_agents_xy_ptr[cdo]		= raw_agents_xy_copy_ptr[cda];
				raw_disp_param_ptr[cdo]		= raw_disp_param_copy_ptr[cda];

				if( lca_cell_id == raw_goal_cells_copy_ptr[cda] )	//THIS_IS_GOAL
				{
					nearest_exit();
					raw_goal_cells_ptr[cdo]	= (unsigned int)(gz * swilc + gx);
				}
				else
				{
					gx							= (int)(raw_goal_cells_copy_ptr[cda] % scene_width_in_lca_cells);
					gz							= (int)(raw_goal_cells_copy_ptr[cda] / scene_width_in_lca_cells);
					raw_goal_cells_ptr[cdo]		= raw_goal_cells_copy_ptr[cda];
				}

				int				diffX		= (gx - (int)lca_col);
				int				ncx			= diffX;
				if( ncx != 0 )	ncx			= diffX / abs( diffX );
				int				diffZ		= (gz - (int)lca_row);
				int				ncz			= diffZ;
				if( ncz != 0 )	ncz			= diffZ / abs( diffZ );
				unsigned int	ntx			= lca_col + (unsigned int)ncx;
				unsigned int	ntz			= lca_row + (unsigned int)ncz;
				raw_vc1_ptr[cdo]			= ntz * scene_width_in_lca_cells + ntx;

				unsigned int	niA			= tlcd;
				unsigned int	niB			= tlcd;
				if( abs(ncx) > 0 && abs(ncz) > 0 )
				{
					if( INBOUNDS( (int)ntx, swilc ) )	niA	= lca_row * scene_width_in_lca_cells + ntx;
					if( INBOUNDS( (int)ntz, swilc ) )	niB	= ntz * scene_width_in_lca_cells + lca_col;
				}
				else if( abs(ncx) > 0 && INBOUNDS( (int)ntx, swilc ) )
				{
					if( (int)lca_row > 0 )				niA	= ((int)lca_row - 1) * scene_width_in_lca_cells + ntx;
					if( ((int)lca_row + 1) < swilc )	niB	= ((int)lca_row + 1) * scene_width_in_lca_cells + ntx;
				}
				else if( INBOUNDS( (int)ntz, swilc ) )
				{
					if( (int)lca_col > 0 )				niA	= ntz * scene_width_in_lca_cells + ((int)lca_col - 1);
					if( ((int)lca_col + 1) < swilc )	niB	= ntz * scene_width_in_lca_cells + ((int)lca_col + 1);
				}
				raw_vc2_ptr[cdo]			= niA;
				raw_vc3_ptr[cdo]			= niB;


				float			adx			= raw_agents_xy_copy_ptr[cda].x;
				float			adz			= raw_agents_xy_copy_ptr[cda].y;
				float			ax			= (((float)lca_col * lca_cell_width) + adx) - f_hsw;
				float			az			= (((float)lca_row * lca_cell_width) + adz) - f_hsw;
				unsigned int	aid			= raw_agents_ids_copy_ptr[cda] * 4;
				raw_agents_pos_ptr[aid + 0]	= ax;
				raw_agents_pos_ptr[aid + 2] = az;

				offset++;
			}

			else if( raw_flags_ptr[cda] > 0 )	//IN TRANSIT
			{
				if( raw_disp_param_copy_ptr[cda] < 1.0f )
				{
					cdo							= cd + offset;
					raw_agents_ids_ptr[cdo]		= raw_agents_ids_copy_ptr[cda];
					raw_agents_xy_ptr[cdo]		= raw_agents_xy_copy_ptr[cda];
					raw_goal_cells_ptr[cdo]		= raw_goal_cells_copy_ptr[cda];
					raw_vc1_ptr[cdo]			= raw_vc1_copy_ptr[cda];
					raw_vc2_ptr[cdo]			= raw_vc2_copy_ptr[cda];
					raw_vc3_ptr[cdo]			= raw_vc3_copy_ptr[cda];
					raw_disp_param_ptr[cdo]		= raw_disp_param_copy_ptr[cda] + raw_disp_delta_ptr[raw_agents_ids_copy_ptr[cda]];;

					unsigned int	vcell		= 0;
					if( raw_flags_ptr[cda] == 1 )		vcell = raw_vc1_copy_ptr[cda];
					else if( raw_flags_ptr[cda] == 2 )	vcell = raw_vc2_copy_ptr[cda];
					else								vcell = raw_vc3_copy_ptr[cda];

					float			adx			= raw_agents_xy_copy_ptr[cda].x;
					float			adz			= raw_agents_xy_copy_ptr[cda].y;

					float			ax1			= (((float)lca_col * lca_cell_width) + adx) - f_hsw;
					float			az1			= (((float)lca_row * lca_cell_width) + adz) - f_hsw;

					unsigned int	lca_row2	= vcell / scene_width_in_lca_cells;
					unsigned int	lca_col2	= vcell % scene_width_in_lca_cells;
					float			ax2			= (((float)lca_col2 * lca_cell_width) + adx) - f_hsw;
					float			az2			= (((float)lca_row2 * lca_cell_width) + adz) - f_hsw;

					// P(t) = A + t(B - A):
					float			dX			= ax2 - ax1;
					float			dZ			= az2 - az1;
					float			ax			= ax1 + raw_disp_param_ptr[cdo] * dX;
					float			az			= az1 + raw_disp_param_ptr[cdo] * dZ;

					// ESTIMATE ORIENTATION:
					float			dirX		= SIGNF( dX );
					float			dirZ		= SIGNF( dZ );
					float			radAngle	= atan2f( dirZ, dirX );
					float			degAngle	= 90.0f - (RAD2DEG * radAngle);

					unsigned int	aid			= raw_agents_ids_copy_ptr[cda] * 4;
					raw_agents_pos_ptr[aid + 0]	= ax;
					raw_agents_pos_ptr[aid + 2] = az;
					raw_agents_pos_ptr[aid + 3] = degAngle * DEG2RAD;

					offset++;
				}
			}

		}

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

					unsigned int vcell = 0;
					if( raw_flags_ptr[nda] == 1 )		vcell = raw_vc1_copy_ptr[nda];
					else if( raw_flags_ptr[nda] == 2 )	vcell = raw_vc2_copy_ptr[nda];
					else if( raw_flags_ptr[nda] == 3 )	vcell = raw_vc3_copy_ptr[nda];

					if( vcell == tlcd )	break;

					if( vcell == lca_cell_id && raw_flags_ptr[nda] > 0 )	//INCOMING_HERE
					{
						float param = raw_disp_param_copy_ptr[nda];
						cdo			= cd + offset;

						if( param >= 1.0f )
						{
							raw_agents_ids_ptr[cdo]		= raw_agents_ids_copy_ptr[nda];
							raw_agents_xy_ptr[cdo]		= raw_agents_xy_copy_ptr[nda];
							raw_disp_param_ptr[cdo]		= 0.0f;

							if( vcell == raw_goal_cells_copy_ptr[nda] )		//NEXT_IS_GOAL
							{
								nearest_exit();
								raw_goal_cells_ptr[cdo]	= (unsigned int)(gz * swilc + gx);
							}
							else
							{
								gx						= (int)(raw_goal_cells_copy_ptr[nda] % scene_width_in_lca_cells);
								gz						= (int)(raw_goal_cells_copy_ptr[nda] / scene_width_in_lca_cells);
								raw_goal_cells_ptr[cdo]	= raw_goal_cells_copy_ptr[nda];
							}

							int				diffX		= (gx - (int)lca_col);
							int				ncx			= diffX;
							if( ncx != 0 )	ncx			= diffX / abs( diffX );
							int				diffZ		= (gz - (int)lca_row);
							int				ncz			= diffZ;
							if( ncz != 0 )	ncz			= diffZ / abs( diffZ );
							unsigned int	ntx			= lca_col + (unsigned int)ncx;
							unsigned int	ntz			= lca_row + (unsigned int)ncz;
							raw_vc1_ptr[cdo]			= ntz * scene_width_in_lca_cells + ntx;

							unsigned int	niA			= tlcd;
							unsigned int	niB			= tlcd;
							if( abs(ncx) > 0 && abs(ncz) > 0 )
							{
								if( INBOUNDS( (int)ntx, swilc ) )	niA	= lca_row * scene_width_in_lca_cells + ntx;
								if( INBOUNDS( (int)ntz, swilc ) )	niB	= ntz * scene_width_in_lca_cells + lca_col;
							}
							else if( abs(ncx) > 0 && INBOUNDS( (int)ntx, swilc ) )
							{
								if( (int)lca_row > 0 )				niA	= ((int)lca_row - 1) * scene_width_in_lca_cells + ntx;
								if( ((int)lca_row + 1) < swilc )	niB	= ((int)lca_row + 1) * scene_width_in_lca_cells + ntx;
							}
							else if( INBOUNDS( (int)ntz, swilc ) )
							{
								if( (int)lca_col > 0 )				niA	= ntz * scene_width_in_lca_cells + ((int)lca_col - 1);
								if( ((int)lca_col + 1) < swilc )	niB	= ntz * scene_width_in_lca_cells + ((int)lca_col + 1);
							}
							raw_vc2_ptr[cdo]			= niA;
							raw_vc3_ptr[cdo]			= niB;

							float			adx			= raw_agents_xy_copy_ptr[nda].x;
							float			adz			= raw_agents_xy_copy_ptr[nda].y;
							float			ax			= (((float)lca_col * lca_cell_width) + adx) - f_hsw;
							float			az			= (((float)lca_row * lca_cell_width) + adz) - f_hsw;
							unsigned int	aid			= raw_agents_ids_copy_ptr[nda] * 4;
							raw_agents_pos_ptr[aid + 0]	= ax;
							raw_agents_pos_ptr[aid + 2] = az;

							offset++;
						}
					}
				}
			}
		}
		thrust::get<1>(t) = offset;
	}

	inline __host__ __device__ void nearest_exit( void )
	{
		float	test_radians				= DEG2RAD * (raw_lca_ptr[lca_cell_id] * f_deg_inc + 135.0f);
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
					if( raw_lca_ptr[i] < (float)DIRECTIONS )
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
					else if( raw_lca_ptr[i] == 9.0f )
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

};
//
//=======================================================================================
//
void advance_agents(	thrust::device_vector<unsigned int>&	cells_ids,
						thrust::device_vector<unsigned int>&	occupancy,
						thrust::device_vector<unsigned int>&	agents_flags,
						thrust::device_vector<unsigned int>&	agents_flags_copy,
						thrust::device_vector<unsigned int>&	agents_ids,
						thrust::device_vector<unsigned int>&	agents_ids_copy,
						thrust::device_vector<unsigned int>&	agents_goal_cell,
						thrust::device_vector<unsigned int>&	agents_goal_cell_copy,
						thrust::device_vector<unsigned int>&	agents_virt_cell1,
						thrust::device_vector<unsigned int>&	agents_virt_cell1_copy,
						thrust::device_vector<unsigned int>&	agents_virt_cell2,
						thrust::device_vector<unsigned int>&	agents_virt_cell2_copy,
						thrust::device_vector<unsigned int>&	agents_virt_cell3,
						thrust::device_vector<unsigned int>&	agents_virt_cell3_copy,
						thrust::device_vector<unsigned int>&	agents_channel,
						thrust::device_vector<unsigned int>&	agents_channel_copy,
						thrust::device_vector<int>&				agents_drift_counter,
						thrust::device_vector<int>&				agents_drift_counter_copy,
						thrust::device_vector<float>&			displace_param,
						thrust::device_vector<float>&			displace_param_copy,
						thrust::device_vector<float2>&			agents_xy,
						thrust::device_vector<float2>&			agents_xy_copy,
						thrust::device_vector<float>&			agents_pos,
						thrust::device_vector<float>&			lca,
						thrust::device_vector<float>&			displace_delta,
						thrust::device_vector<float>&			mdp_speeds,
						unsigned int							scene_width_in_lca_cells,
						unsigned int							scene_width_in_mdp_cells,
						unsigned int							mdp_channels,
						float									cell_width					)
{
	unsigned int total_lca_cells = scene_width_in_lca_cells * scene_width_in_lca_cells;
	
	thrust::copy( agents_ids.begin(), agents_ids.end(), agents_ids_copy.begin() );
	thrust::fill( agents_ids.begin(), agents_ids.end(), total_lca_cells * DIRECTIONS );

	thrust::copy( agents_flags.begin(), agents_flags.end(), agents_flags_copy.begin() );
	thrust::fill( agents_flags.begin(), agents_flags.end(), total_lca_cells * DIRECTIONS );
	
	thrust::copy( agents_goal_cell.begin(), agents_goal_cell.end(), agents_goal_cell_copy.begin() );
	thrust::fill( agents_goal_cell.begin(), agents_goal_cell.end(), total_lca_cells * DIRECTIONS );
	
	thrust::copy( agents_virt_cell1.begin(), agents_virt_cell1.end(), agents_virt_cell1_copy.begin() );
	thrust::fill( agents_virt_cell1.begin(), agents_virt_cell1.end(), total_lca_cells * DIRECTIONS );

	thrust::copy( agents_virt_cell2.begin(), agents_virt_cell2.end(), agents_virt_cell2_copy.begin() );
	thrust::fill( agents_virt_cell2.begin(), agents_virt_cell2.end(), total_lca_cells * DIRECTIONS );

	thrust::copy( agents_virt_cell3.begin(), agents_virt_cell3.end(), agents_virt_cell3_copy.begin() );
	thrust::fill( agents_virt_cell3.begin(), agents_virt_cell3.end(), total_lca_cells * DIRECTIONS );

	thrust::copy( agents_channel.begin(), agents_channel.end(), agents_channel_copy.begin() );
	thrust::fill( agents_channel.begin(), agents_channel.end(), mdp_channels );

	thrust::copy( agents_drift_counter.begin(), agents_drift_counter.end(), agents_drift_counter_copy.begin() );
	thrust::fill( agents_drift_counter.begin(), agents_drift_counter.end(), 0 );

	thrust::copy( displace_param.begin(), displace_param.end(), displace_param_copy.begin() );
	thrust::fill( displace_param.begin(), displace_param.end(), 0.0f );
	
	thrust::copy( agents_xy.begin(), agents_xy.end(), agents_xy_copy.begin() );
	thrust::fill( agents_xy.begin(), agents_xy.end(), make_float2( 0.0f, 0.0f ) );
	
	unsigned int*	raw_agents_ids_ptr		= thrust::raw_pointer_cast( agents_ids.data()				);
	unsigned int*	raw_agents_ids_copy_ptr	= thrust::raw_pointer_cast( agents_ids_copy.data()			);
	unsigned int*	raw_goal_cells_ptr		= thrust::raw_pointer_cast( agents_goal_cell.data()			);
	unsigned int*	raw_goal_cells_copy_ptr	= thrust::raw_pointer_cast( agents_goal_cell_copy.data()	);
	unsigned int*	raw_vc1_ptr				= thrust::raw_pointer_cast( agents_virt_cell1.data()		);
	unsigned int*	raw_vc1_copy_ptr		= thrust::raw_pointer_cast( agents_virt_cell1_copy.data()	);
	unsigned int*	raw_vc2_ptr				= thrust::raw_pointer_cast( agents_virt_cell2.data()		);
	unsigned int*	raw_vc2_copy_ptr		= thrust::raw_pointer_cast( agents_virt_cell2_copy.data()	);
	unsigned int*	raw_vc3_ptr				= thrust::raw_pointer_cast( agents_virt_cell3.data()		);
	unsigned int*	raw_vc3_copy_ptr		= thrust::raw_pointer_cast( agents_virt_cell3_copy.data()	);
	unsigned int*	raw_chn_ptr				= thrust::raw_pointer_cast( agents_channel.data()			);
	unsigned int*	raw_chn_copy_ptr		= thrust::raw_pointer_cast( agents_channel_copy.data()		);
	unsigned int*	raw_flags_ptr			= thrust::raw_pointer_cast( agents_flags.data()				);
	unsigned int*	raw_flags_copy_ptr		= thrust::raw_pointer_cast( agents_flags_copy.data()		);
	
	int*			raw_drift_ptr			= thrust::raw_pointer_cast( agents_drift_counter.data()		);
	int*			raw_drift_copy_ptr		= thrust::raw_pointer_cast( agents_drift_counter_copy.data());
	
	float*			raw_disp_param_ptr		= thrust::raw_pointer_cast( displace_param.data()			);
	float*			raw_disp_param_copy_ptr	= thrust::raw_pointer_cast( displace_param_copy.data()		);
	
	float2*			raw_agents_xy_ptr		= thrust::raw_pointer_cast( agents_xy.data()				);
	float2*			raw_agents_xy_copy_ptr	= thrust::raw_pointer_cast( agents_xy_copy.data()			);
	
	float*			raw_agents_pos_ptr		= thrust::raw_pointer_cast( agents_pos.data()				);
	float*			raw_lca_ptr				= thrust::raw_pointer_cast( lca.data()						);
	float*			raw_disp_delta_ptr		= thrust::raw_pointer_cast( displace_delta.data()			);
	float*			raw_mdp_speeds_ptr		= thrust::raw_pointer_cast( mdp_speeds.data()				);


	thrust::for_each(
		thrust::make_zip_iterator(
			thrust::make_tuple(
				cells_ids.begin(),
				occupancy.begin()
			)
		),
		thrust::make_zip_iterator(
			thrust::make_tuple(
				cells_ids.end(),
				occupancy.end()
			)
		),
		advance_n_agents2(	cell_width,
							scene_width_in_lca_cells,
							scene_width_in_mdp_cells,
							raw_agents_ids_ptr,
							raw_agents_ids_copy_ptr,
							raw_goal_cells_ptr,
							raw_goal_cells_copy_ptr,
							raw_vc1_ptr,
							raw_vc1_copy_ptr,
							raw_vc2_ptr,
							raw_vc2_copy_ptr,
							raw_vc3_ptr,
							raw_vc3_copy_ptr,
							raw_chn_ptr,
							raw_chn_copy_ptr,
							raw_flags_ptr,
							raw_flags_copy_ptr,
							raw_drift_ptr,
							raw_drift_copy_ptr,
							raw_disp_param_ptr,
							raw_disp_param_copy_ptr,
							raw_agents_xy_ptr,
							raw_agents_xy_copy_ptr,
							raw_agents_pos_ptr,
							raw_lca_ptr,
							raw_disp_delta_ptr,
							raw_mdp_speeds_ptr		)
	);

/*
	thrust::for_each(
		thrust::make_zip_iterator(
			thrust::make_tuple(
				cells_ids.begin(),
				occupancy.begin()
			)
		),
		thrust::make_zip_iterator(
			thrust::make_tuple(
				cells_ids.end(),
				occupancy.end()
			)
		),
		advance_n_agents(	cell_width,
							scene_width_in_lca_cells,
							scene_width_in_mdp_cells,
							raw_agents_ids_ptr,
							raw_agents_ids_copy_ptr,
							raw_goal_cells_ptr,
							raw_goal_cells_copy_ptr,
							raw_vc1_ptr,
							raw_vc1_copy_ptr,
							raw_vc2_ptr,
							raw_vc2_copy_ptr,
							raw_vc3_ptr,
							raw_vc3_copy_ptr,
							raw_disp_param_ptr,
							raw_disp_param_copy_ptr,
							raw_flags_ptr,
							raw_agents_xy_ptr,
							raw_agents_xy_copy_ptr,
							raw_agents_pos_ptr,
							raw_lca_ptr,
							raw_disp_delta_ptr		)
	);
*/
}
//
//=======================================================================================
//

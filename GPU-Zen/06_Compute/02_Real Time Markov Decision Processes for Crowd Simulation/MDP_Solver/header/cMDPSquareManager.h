
/*

Copyright 2014 Sergio Ruiz, Benjamin Hernandez

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


*/


#pragma once


#include <fstream>
#include <sstream>
#include <vector>
#include <string>

#include "ccMDP_GPU.h"
#include "ccMDP_GPU_modular.h"

using namespace std;

//=======================================================================================

#ifndef __MDP_SQUARE_MANAGER
#define __MDP_SQUARE_MANAGER

class MDPSquareManager
{
public:
							MDPSquareManager		(	void								);
							~MDPSquareManager		(	void								);

	bool					solve_from_csv			(	string&				csvFilename,
														vector<float>&		policy			);

	bool 					solve_from_csv			(	string& 			csvFilename,
														vector<float>& 		policy,
														int 				table_method 	);

	void					get_policy				(	vector<float>&		policy			);
	int						getRows					(	void								);
	int						getColumns				(	void								);
	int						getIterations			(	void								);
	bool					getConvergence			(	void								);

	bool					in_bounds				(	int					r,
														int					c,
														int					q				);
	int						mdp_get_index			(	int					row,
														int					column,
														int					iteration		);
	int						mdp_get_qindex			(	int					row,
														int					column,
														int					iteration,
														int					q				);
	int						mdp_get_index			(	int					row,
														int					column,
														int					iteration,
														int					direction		);
	int						mdp_get_qindex			(	int					row,
														int					column,
														int					iteration,
														int					direction,
														int					q				);

//->FOR_INT_VECTORS
	int						mdp_get_value			(	int					row,
														int					column,
														int					iteration,
														vector<int>&		mdp_vector		);
	int						mdp_get_value			(	int					row,
														int					column,
														int					iteration,
														int					direction,
														vector<int>&		mdp_vector		);
	int						mdp_get_qvalue			(	int					row,
														int					column,
														int					iteration,
														int					direction,
														int					q,
														vector<int>&		mdp_vector		);
//<-FOR_INT_VECTORS

//->FOR_FLOAT_VECTORS
	float					mdp_get_value			(	int					row,
														int					column,
														int					iteration,
														vector<float>&		mdp_vector		);
	float					mdp_get_value			(	int					row,
														int					column,
														int					iteration,
														int					direction,
														vector<float>&		mdp_vector		);
	float					mdp_get_qvalue			(	int					row,
														int					column,
														int					iteration,
														int					direction,
														int					q,
														vector<float>&		mdp_vector		);
//<-FOR_FLOAT_VECTORS

	bool					mdp_is_wall				(	int					row,
														int					column			);
	bool					mdp_is_exit				(	int					row,
														int					column			);
	void					split					(	const string&		str,
														const string&		delimiters,
														vector<string>& 	tokens			);
	vector<float>			read_matrix				(	string				file_name		);
	void					write_policy			(	void								);
	void					print_rewards			(	void								);
	void					print_mdp				(	void								);
	void					init_mdp				(	void								);
	void					getRewards				(	vector<float>&		mdp				);



//->SINGLE_ITERATION_PROCESS
	void					init_structures_on_host	(	vector<float>&		mdp_topology	);
	void					init_perms_on_device	(	void								);
	void					upload_to_device		(	void								);
	void					iterate_on_device		(	void								);
	void					download_to_host		(	void								);
//<-SINGLE_ITERATION_PROCESS



private:

	int						cells;
	int						rows;
	int						columns;
	const int				NQ;
	float					discount;
	bool					convergence;
	float					iteratingTime;
	int						iterations;

	vector<int>				reachable;
	vector<float>			Q;								// NQ-sized
	vector<float>			V;
	vector<int>				P;
	vector<int>				is_wall;
	vector<float>			rewards;
	vector<float>			host_probability1;
	vector<float>			host_probability2;
	vector<float>			host_permutations;
	vector<int>				host_vicinity;
	vector<int>				host_vicinity8;
	vector<float>			iterationTimes;
// declare sieves and shifts for convolution:
	vector<float>			host_dir_rw_sieve;
	vector<vector <float> >	host_dir_rw_shifts;
	vector<float>			host_dir_pv_sieve;
	vector<vector <float> >	host_dir_pv_shifts;
	vector<int>				host_dir_ib_sieve;
	vector<vector <int> >	host_dir_ib_shifts;
	vector<int>				host_dir_iw_sieve;
	vector< vector<int> >	host_dir_iw_shifts;
	vector<float>			host_dir_rw;
	vector<float>			host_dir_rw8;
	vector<float>			host_dir_pv;
	vector<float>			host_dir_pv8;
	vector<int>				host_dir_ib;
	vector<int>				host_dir_iw;
	vector<int>				host_dir_ib_iw;
	vector<int>				host_dir_ib_iw8;
	vector<int>				host_dir_ib_iw8_inv;
};

#endif

//=======================================================================================

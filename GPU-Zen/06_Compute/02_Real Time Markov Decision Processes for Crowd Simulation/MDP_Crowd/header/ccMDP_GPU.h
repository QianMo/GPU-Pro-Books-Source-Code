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

#pragma once

#include <vector>

// mdp_gpu_device.h: Declares C++ <-> CUDA Interoperation prototypes

/******************************************************************************************************************************************************
	Inits Permutation Tables on Device:

	 -------------------- --------------------- ------------------------------------------------------------------------------------------------------
	|    PARAM NAME		 |	 (INITIAL) SIZE	   |                                            DESCRIPTION												  |
	 -------------------- --------------------- ------------------------------------------------------------------------------------------------------
	rows				 |	1				   | number of MDP matrix horizontal lines.
	columns				 |	1				   | number of MDP matrix vertical lines.
	NQ					 |	1				   | number of options to choose from (directions, states, and so on).
	host_dir_ib_iwNQ	 |	rows*columns*NQ*NQ | for every MDP cell, in-bounds & not-forbidden values (0-1). Every NQ-chunk repeated NQ times.
	host_dir_ib_iwNQ_inv |	rows*columns*NQ*NQ | inverse for host_dir_ib_iwNQ.
	host_probability1	 |	rows*columns	   | for every MDP cell, the probability for the most likely option to be chosen.
	host_probability2	 |	rows*columns	   | for every MDP cell, the probability for the least likely option to be chosen.
	host_permutations	 |	0				   | empty vector that will hold the final permutation tables usable by the mdp_iterate_on_device function.

******************************************************************************************************************************************************/

void mdp_init_permutations_on_device	(	const int				rows,
											const int				columns,
											const int				NQ,
											std::vector<int>&		host_dir_ib_iwNQ,
											std::vector<int>&		host_dir_ib_iwNQ_inv,
											std::vector<float>&		host_probability1,
											std::vector<float>&		host_probability2,
											std::vector<float>&		host_permutations		);

/******************************************************************************************************************************************************
Inits Permutation Tables on Device 2:

------------------------ ---------------------- ------------------------------------------------------------------------------------------------------
|    PARAM NAME			|	 (INITIAL) SIZE	   |                                            DESCRIPTION												  |
------------------------ ---------------------- ------------------------------------------------------------------------------------------------------
rows					|	1				   | number of MDP matrix horizontal lines.
columns					|	1				   | number of MDP matrix vertical lines.
NQ						|	1				   | number of options to choose from (directions, states, and so on).
probability1			|	1				   | the probability for the most likely option to be chosen.
probability2			|	1				   | the probability for the least likely option to be chosen.
host_dir_ib_iwNQ		|	rows*columns*NQ*NQ | for every MDP cell, in-bounds & not-forbidden values (0-1). Every NQ-chunk repeated NQ times.
host_dir_ib_iwNQ_trans	|	rows*columns*NQ*NQ | transposed (over secondary diagonal) for host_dir_ib_iwNQ.
host_permutations		|	0				   | empty vector that will hold the final permutation tables usable by the mdp_iterate_on_device function.

******************************************************************************************************************************************************/

void mdp_init_permutations_on_device2(		const int				rows,
											const int				columns,
											const int				NQ,
											const float				probability1,
											const float				probability2,
											std::vector<int>&		host_dir_ib_iw,
											std::vector<int>&		host_dir_ib_iw_trans,
											std::vector<float>&		host_permutations		);

/******************************************************************************************************************************************************
	Iterates MDP values on Device until convergence is achieved or the safety limit is reached:

	 -------------------- --------------------- ------------------------------------------------------------------------------------------------------
	|    PARAM NAME		 |	 (INITIAL) SIZE	   |                                            DESCRIPTION												  |
	 -------------------- --------------------- ------------------------------------------------------------------------------------------------------
	rows				 |  1				   | number of MDP matrix horizontal lines.
	columns				 |  1				   | number of MDP matrix vertical lines.
	NQ					 |  1				   | number of options to choose from (directions, states, and so on).
	discount			 |  1				   | discount factor (GAMMA) applicable to this MDP.
	host_vicinityNQ		 |  rows*columns*NQ*NQ | for every MDP cell, holds the index for the cell of each neighbor. Each value is repeated NQ times.
	host_dir_rwNQ		 |  rows*columns*NQ*NQ | for every MDP cell, holds the reward for the cell of each neighbor, distributed amongst NQ values.
	P					 |  rows*columns	   | for every MDP cell, holds the value of the best policy. Grows by (rows*columns) on every iteration.
	Q					 |  rows*columns*NQ	   | for every MDP cell, holds the value for every option available at the current iteration.
	V					 |  rows*columns+1	   | for every MDP cell, holds the best Q amongst the NQ-Qvalues available.
	host_dir_pvNQ		 |  rows*columns*NQ*NQ | for every MDP cell, holds the previous iteration's V value of every neighbouring cell.
	host_permutations	 |  rows*columns*NQ*NQ | Vector holding the permutation tables generated by the 'mdp_init_permutations_on_device' function.
	iteratingTime		 |  1				   | returns the time elapsed while iterating.
	iterationTimes		 |  0				   | save time for each iteration.
	convergence			 |  1				   | returns the convergence confirmation.

******************************************************************************************************************************************************/

int mdp_iterate_on_device		(	const int				rows,
									const int				columns,
									const int				NQ,
									float					discount,
									std::vector<int>&		host_vicinityNQ,
									std::vector<float>&		host_dir_rw,
									std::vector<int>&		P,
									std::vector<float>&		Q,
									std::vector<float>&		V,
									std::vector<float>&		host_dir_pv,
									std::vector<float>&		host_permutations,
									float&					totalTime,
									std::vector<float>&		iterationTimes,
									bool&					convergence			);

/*****************************************************************************************************************************************************/

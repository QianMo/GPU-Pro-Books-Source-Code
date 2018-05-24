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

void mmdp_init_permutations_on_device	(	const int				_rows,
											const int				_columns,
											const int				_NQ,
											std::vector<int>&		host_dir_ib_iwNQ,
											std::vector<int>&		host_dir_ib_iwNQ_inv,
											std::vector<float>&		host_probability1,
											std::vector<float>&		host_probability2,
											std::vector<float>&		host_permutations		);

void mmdp_init_permutations_on_device2	(	const int				rows,
											const int				columns,
											const int				NQ,
											const float				probability1,
											const float				probability2,
											std::vector<int>&		host_dir_ib_iw,
											std::vector<int>&		host_dir_ib_iw_trans,
											std::vector<float>&		host_permutations		);

void mmdp_upload_to_device				(	std::vector<int>&		P,
											std::vector<float>&		Q,
											std::vector<float>&		V,
											std::vector<int>&		host_vicinityNQ,
											std::vector<float>&		host_dir_rwNQ,
											std::vector<float>&		host_dir_pvNQ,
											std::vector<float>&		host_permutations		);

int mmdp_iterate_on_device				(	float					discount,
											bool&					convergence				);

void mmdp_download_to_host				(	std::vector<int>&		P,
											std::vector<float>&		V						);

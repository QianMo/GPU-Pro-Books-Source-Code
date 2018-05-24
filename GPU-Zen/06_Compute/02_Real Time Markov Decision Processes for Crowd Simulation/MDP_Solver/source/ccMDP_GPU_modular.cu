
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


#include "ccMDP_GPU_modular.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/count.h>
#include <thrust/copy.h>
#include <thrust/tuple.h>
#include <thrust/iterator/zip_iterator.h>

// k, NQ, mdp_size, mdp_sizeNQ, mdp_sizeNQxNQ, rows, columns
#define	I_K				0
#define I_NQ			1
#define I_MDP_SIZE		2
#define I_C_SIZE		3
#define I_MDP_SIZENQ	4
#define I_MDP_SIZENQxNQ	5
#define I_ROWS			6
#define	I_COLUMNS		7

//=======================================================================================
//

//->SHORT-HAND_FOR_THRUST_OPERATORS
thrust::multiplies<float>						op_mult_float;
thrust::plus<float>								op_plus_float;
thrust::minus<int>								op_minus_int;
thrust::divides<int>							op_div_int;
//<-SHORT-HAND_FOR_THRUST_OPERATORS
//->SHORT-HAND_FOR_THRUST_ITERATORS
thrust::device_vector<float>::iterator			iterT;
thrust::device_vector<float>::iterator			iterQ;
thrust::device_vector<float>::iterator			iterV;
thrust::device_vector<int>::iterator			iterS;
thrust::device_vector<int>::iterator			iterP;
//<-SHORT-HAND_FOR_THRUST_ITERATORS
//->VARIABLES_FOR_THRUST_ZIP_ITERATOR
typedef thrust::device_vector<int>::iterator	IntIter;
typedef thrust::device_vector<float>::iterator	FloatIter;
typedef thrust::tuple<FloatIter, IntIter>		IteratorTuple;
typedef thrust::zip_iterator<IteratorTuple>		ZipIterator;
thrust::maximum< thrust::tuple<float,int> >		zi_binary_op;		// Binary Operator for key-reduction.
thrust::equal_to<int>							zi_binary_pred;		// Binary Predicate for key-reduction.
//<-VARIABLES_FOR_THRUST_ZIP_ITERATOR
//->GLOBAL_VECTORS
thrust::device_vector<float>					dev_permutations;
thrust::device_vector<int>						dev_vicinityNQ;
thrust::device_vector<float>					dev_dir_rwNQ;
thrust::device_vector<float>					dev_dir_pvNQ;
thrust::device_vector<int>						dev_P;
thrust::device_vector<float>					dev_Q;
thrust::device_vector<float>					dev_V;
//<-GLOBAL_VECTORS
thrust::device_vector<int>						vars;				// k, NQ, mdp_size, mdp_sizeNQ, mdp_sizeNQxNQ, rows, columns

//
//=======================================================================================
//
void mmdp_init_permutations_on_device(	const int				_rows,
										const int				_columns,
										const int				_NQ,
										std::vector<int>&		host_dir_ib_iwNQ,
										std::vector<int>&		host_dir_ib_iwNQ_inv,
										std::vector<float>&		host_probability1,
										std::vector<float>&		host_probability2,
										std::vector<float>&		host_permutations	)
{
//->INIT_VARIABLES
	host_permutations.clear();
	vars.clear();
	vars.push_back( 0 );
	vars.push_back(_NQ );
	vars.push_back( _rows * _columns );
	vars.push_back( _rows * _columns );
	vars.push_back( _rows * _columns * _NQ );
	vars.push_back( _rows * _columns * _NQ * _NQ );
	vars.push_back( _rows );
	vars.push_back( _columns );

	float											T1				= 0.0f;
	float											T2				= 0.0f;
	int												i				= 0;
	int												r				= 0;
	int												c				= 0;
	int												indexNQxNQ		= 0;
	int												NQ2				= vars[I_NQ] * vars[I_NQ];
	thrust::device_vector<int>						dev_dir_ib_iwNQ		( vars[I_ROWS] * vars[I_COLUMNS] * NQ2	);
	thrust::device_vector<int>						dev_dir_ib_iwNQ_inv	( vars[I_ROWS] * vars[I_COLUMNS] * NQ2	);
	thrust::device_vector<float>					temp_table			( NQ2					);
	thrust::device_vector<float>					diagonal_1			( NQ2					);
	thrust::device_vector<float>					diagonal_1_inv		( NQ2					);
	thrust::device_vector<float>					diagonal_T1			( NQ2					);
	thrust::device_vector<float>					full_T1				( NQ2					);
//->TRANSFER_DATA_TO_DEVICE
	thrust::copy(	host_dir_ib_iwNQ.begin(),
					host_dir_ib_iwNQ.end(),
					dev_dir_ib_iwNQ.begin()			);
	thrust::copy(	host_dir_ib_iwNQ_inv.begin(),
					host_dir_ib_iwNQ_inv.end(),
					dev_dir_ib_iwNQ_inv.begin()		);
//<-TRANSFER_DATA_TO_DEVICE
//->PREPARE_DIAGONAL_MATRICES
	for( r = 0; r < vars[I_NQ]; r++ )
	{
		for( c = 0; c < vars[I_NQ]; c++ )
		{
			i = r * vars[I_NQ] + c;
			if( r == c )
			{
				diagonal_1[i]		= 1.0f;
				diagonal_1_inv[i]	= 0.0f;
			}
			else
			{
				diagonal_1[i]		= 0.0f;
				diagonal_1_inv[i]	= 1.0f;
			}
		}
	}
//<-PREPARE_DIAGONAL_MATRICES
	host_permutations.resize( vars[I_ROWS] * vars[I_COLUMNS] * NQ2 );
//<-INIT_VARIABLES
	for( r = 0; r < vars[I_ROWS]; r++ )
	{
		for( c = 0; c < vars[I_COLUMNS]; c++ )
		{
			i				= r * vars[I_COLUMNS];
			T1				= host_probability1[ i + c ];
			// Obtain a diagonal matrix with T1:
			thrust::fill		(	full_T1.begin(),
									full_T1.end(),
									T1											);
			thrust::transform	(	full_T1.begin(),
									full_T1.end(),
									diagonal_1.begin(),
									diagonal_T1.begin(),
									op_mult_float								);
			T2				= host_probability2[ i + c ];
			indexNQxNQ		= (i * NQ2) + (c * NQ2);	// Index of the permutation table.
			// 1.A. Fill table with T2:
			thrust::fill		(	temp_table.begin(),
									temp_table.end(),
									T2											);
			// 1.B. Fill diagonal with T1:
			thrust::transform	(	temp_table.begin(),
									temp_table.end(),
									diagonal_1_inv.begin(),
									temp_table.begin(),
									op_mult_float								);
			thrust::transform	(	temp_table.begin(),
									temp_table.end(),
									diagonal_T1.begin(),
									temp_table.begin(),
									op_plus_float								);
			// At this time, "temp_table" has a diagonal filled with T1, every other cell holds T2.
			// 2.A. Clean table (horizontal):
			thrust::transform	(	temp_table.begin(),
									temp_table.end(),
									dev_dir_ib_iwNQ.begin() + indexNQxNQ,
									temp_table.begin(),
									op_mult_float								);
			// 2.B. Clean table (vertical):
			thrust::transform	(	temp_table.begin(),
									temp_table.end(),
									dev_dir_ib_iwNQ_inv.begin() + indexNQxNQ,
									temp_table.begin(),
									op_mult_float								);
			// 3. Transfer back to host (i.e. append 'temp_table' to 'host_permutations'):
			thrust::copy		(	temp_table.begin(),
									temp_table.end(),
									host_permutations.begin() + indexNQxNQ		);
		}
	}
}
//
//=======================================================================================
//
void mmdp_upload_to_device( 	std::vector<int>&		P,
								std::vector<float>&		Q,
								std::vector<float>&		V,
								std::vector<int>&		host_vicinityNQ,
								std::vector<float>&		host_dir_rwNQ,
								std::vector<float>&		host_dir_pvNQ,
								std::vector<float>&		host_permutations	)
{
	dev_P.clear();
	dev_Q.clear();
	dev_V.clear();
	dev_vicinityNQ.clear();
	dev_dir_rwNQ.clear();
	dev_dir_pvNQ.clear();
	dev_permutations.clear();

	dev_P.resize( P.size() );
	dev_Q.resize( Q.size() );
	dev_V.resize( V.size() );
	dev_vicinityNQ.resize( host_vicinityNQ.size() );
	dev_dir_rwNQ.resize( host_dir_rwNQ.size() );
	dev_dir_pvNQ.resize( host_dir_pvNQ.size() );
	dev_permutations.resize( host_permutations.size() );

	thrust::copy( P.begin(),					P.end(),					dev_P.begin()				);
	thrust::copy( Q.begin(),					Q.end(),					dev_Q.begin()				);
	thrust::copy( V.begin(),					V.end(),					dev_V.begin()				);
	thrust::copy( host_vicinityNQ.begin(),		host_vicinityNQ.end(),		dev_vicinityNQ.begin()		);
	thrust::copy( host_dir_rwNQ.begin(),		host_dir_rwNQ.end(),		dev_dir_rwNQ.begin()		);
	thrust::copy( host_dir_pvNQ.begin(),		host_dir_pvNQ.end(),		dev_dir_pvNQ.begin()		);
	thrust::copy( host_permutations.begin(),	host_permutations.end(),	dev_permutations.begin()	);
}
//
//=======================================================================================
//
int	mmdp_iterate_on_device(	float					discount,
							bool&					convergence		)
{
//->INIT_VARIABLES
	int												mdp_size		= vars[I_MDP_SIZE];
	int												mdp_sizeNQxNQ	= vars[I_MDP_SIZENQxNQ];
	int												mdp_sizeNQ		= vars[I_MDP_SIZENQ];
	int												NQ				= vars[I_NQ];
	int												i1				= 0;
	int												i2				= 0;
	thrust::device_vector<int>						dev_result			( mdp_size						);
	thrust::device_vector<float>					dev_prob_tables		( mdp_sizeNQxNQ					);
	thrust::device_vector<float>					dev_discountNQxNQ	( mdp_sizeNQxNQ					);
	thrust::device_vector<int>						seq_indicesQ		( mdp_sizeNQ					);		// Helper for key-reduction.
	thrust::device_vector<int>						indicesNQxNQ		( mdp_sizeNQxNQ					);		// Helper for key-reduction.
	thrust::device_vector<int>						indicesNQ			( mdp_sizeNQ					);		// Helper for key-reduction.
	thrust::device_vector<int>						indices_outNQxNQ	( dev_Q.size()					);		// Helper for key-reduction.
	thrust::device_vector<int>						indices_outNQ		( mdp_size						);		// Helper for key-reduction.
	thrust::device_vector<int>						nqs_NQ				( mdp_sizeNQ					);		// Helper for filling "indicesNQ"
	thrust::device_vector<int>						nqs_NQxNQ			( mdp_sizeNQxNQ					);		// Helper for filling "indicesNQxNQ"
//->FILL_KEY_REDUCTION_HELPERS
	thrust::fill		(	dev_discountNQxNQ.begin(),	dev_discountNQxNQ.end(),	discount			);
	thrust::fill		(	nqs_NQxNQ.begin(),			nqs_NQxNQ.end(),			NQ					);
	thrust::fill		(	nqs_NQ.begin(),				nqs_NQ.end(),				NQ					);
	thrust::sequence	(	indicesNQxNQ.begin(),		indicesNQxNQ.end()								);
	thrust::sequence	(	indicesNQ.begin(),			indicesNQ.end()									);
	thrust::transform	(	indicesNQxNQ.begin(),
							indicesNQxNQ.end(),
							nqs_NQxNQ.begin(),
							indicesNQxNQ.begin(),
							op_div_int																	);
	thrust::transform	(	indicesNQ.begin(),
							indicesNQ.end(),
							nqs_NQ.begin(),
							indicesNQ.begin(),
							op_div_int																	);
	for( iterS = seq_indicesQ.begin(); iterS < seq_indicesQ.end(); iterS += NQ  )
	{
		thrust::sequence( iterS, iterS + NQ );
	}
//<-FILL_KEY_REDUCTION_HELPERS
//<-INIT_VARIABLES

//->PERMUTATIONS_VS_PREV_VALUES_MULTIPLICATION
	// With these 3 operations most of the MDP is actually solved (and really fast!):
	thrust::transform(	dev_permutations.begin(),
						dev_permutations.end(),
						dev_dir_pvNQ.begin(),
						dev_prob_tables.begin(),
						op_mult_float				);
	thrust::transform(	dev_prob_tables.begin(),
						dev_prob_tables.end(),
						dev_discountNQxNQ.begin(),
						dev_prob_tables.begin(),
						op_mult_float				);
	thrust::transform(	dev_prob_tables.begin(),
						dev_prob_tables.end(),
						dev_dir_rwNQ.begin(),
						dev_prob_tables.begin(),
						op_plus_float				);
//<-PERMUTATIONS_VS_PREV_VALUES_MULTIPLICATION
//->OBTAINING_QS
	thrust::reduce_by_key	(	indicesNQxNQ.begin(),
								indicesNQxNQ.end(),
								dev_prob_tables.begin(),
								indices_outNQxNQ.begin(),
								dev_Q.begin()				);
	thrust::replace			(	dev_Q.begin(),
								dev_Q.end(),
								0.0f,
								-100.0f						);
//<-OBTAINING_QS
//->OBTAINING_BEST_QS
	dev_P.resize( vars[I_C_SIZE] + mdp_size );
	ZipIterator firstIn  = thrust::make_zip_iterator( thrust::make_tuple(	dev_Q.begin(),
																			seq_indicesQ.begin()			) );
	ZipIterator firstOut = thrust::make_zip_iterator( thrust::make_tuple(	dev_V.begin(),
																			dev_P.begin() + vars[I_C_SIZE]	) );
	thrust::reduce_by_key(	indicesNQ.begin(),
							indicesNQ.end(),
							firstIn,
							indices_outNQ.begin(),
							firstOut,
							zi_binary_pred,
							zi_binary_op			);
	vars[I_C_SIZE] += mdp_size;
//<-OBTAINING_BEST_QS
//->CHECK_CONVERGENCE
	i1 = mdp_size * (vars[I_K] - 1);
	i2 = mdp_size * vars[I_K];
	thrust::transform(	dev_P.begin() + i1,
						dev_P.begin() + i1 + mdp_size,
						dev_P.begin() + i2,
						dev_result.begin(),
						op_minus_int				);
	if( thrust::count( dev_result.begin(), dev_result.end(), 0 ) == mdp_size )
	{
		convergence = true;
	}
	else
	{
//->UPDATE_DEV_DIR_PV
		// Perform the update with a map-value operation:
		thrust::gather( dev_vicinityNQ.begin(),	// map_begins
						dev_vicinityNQ.end(),	// map_ends
						dev_V.begin(),			// values
						dev_dir_pvNQ.begin()	);
//<-UPDATE_DEV_DIR_PV
		vars[I_K]++;
	}
//<-CHECK_CONVERGENCE
	return vars[I_K];
}
//
//=======================================================================================
//
void mmdp_download_to_host(	std::vector<int>&		P,
							std::vector<float>&		V	)
{
	P.resize( dev_P.size() );
	thrust::copy( dev_P.begin(), dev_P.end(), P.begin() );
	thrust::copy( dev_V.begin(), dev_V.end(), V.begin() );
}
//
//=======================================================================================

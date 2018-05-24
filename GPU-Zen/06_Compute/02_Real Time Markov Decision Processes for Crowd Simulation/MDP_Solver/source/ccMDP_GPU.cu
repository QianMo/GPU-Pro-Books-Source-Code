
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


#define DEBUG_TIME 1		//DISPLAY_OPERATION_TIME
//#define SHOW_ITERS 1		//DISPLAY_ITERATIONS
//#define SHOW_PERMS 1		//DISPLAY_PERMUTATION_TABLES

#define ITER_SAFETY	1001	//LIMIT_FOR_ITERATIONS

#include "ccMDP_GPU.h"

#include <thrust/version.h>
#include <cuda.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/binary_search.h>
#include <thrust/sort.h>
#include <thrust/count.h>
#include <thrust/copy.h>
#include <thrust/tuple.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/execution_policy.h>
#include <thrust/system_error.h>
#include <thrust/system/cuda/execution_policy.h>
#include "ccThrustUtil.h"

struct CopyFunctor : thrust::unary_function<int, int>
{
	CopyFunctor(){}

	__device__ int operator()(int i)
	{
		return i;
	}
};

struct false_prob : thrust::unary_function<float, float>
{
	const float p;
	false_prob(float _p) : p(_p) {}
	__device__ float operator()(int reachable) const
	{
		float frc = (float)reachable - 1.0f;
		if( frc > 0.0f )
		{
			return p / frc;
		}
		else
		{
			return 0.0f;
		}
	}
};

struct true_prob : thrust::unary_function<float, float>
{
	const float p;
	true_prob(float _p) : p(_p) {}
	__device__ float operator()(int reachable) const
	{
		if( reachable > 1 )
		{
			return p;
		}
		else
		{
			return 1.0f;
		}
	}
};

struct index_diag : thrust::unary_function<int, int>
{
	const int NQ;
	index_diag(int _nq) : NQ(_nq) {}
	__device__ float operator()(int index) const
	{
		int col = index % NQ;
		int row = ((index - col) / NQ) % NQ;
		if( row == col )
		{
			return 1;
		}
		else
		{
			return 0;
		}
	}
};

/**************************************************************************************************************************************************
	Inits Permutation Tables on Device:

	PARAM NAME				(INITIAL) SIZE		DESCRIPTION
	rows					1					number of MDP matrix horizontal lines.
	columns					1					number of MDP matrix vertical lines.
	NQ						1					number of options to choose from (directions, states, and so on).
	host_dir_ib_iwNQ		rows*columns*NQ*NQ	for every MDP cell, in-bounds & not-forbidden values (0-1). Every NQ-chunk repeated NQ times.
	host_dir_ib_iwNQ_inv	rows*columns*NQ*NQ	inverse for host_dir_ib_iwNQ.
	host_probability1		rows*columns		for every MDP cell, the probability for the most likely option to be chosen.
	host_probability2		rows*columns		for every MDP cell, the probability for the least likely option to be chosen.
	host_permutations		0					empty vector that will hold the final permutation tables usable by the mdp_iterate_on_device function.
**************************************************************************************************************************************************/

void mdp_init_permutations_on_device(	const int				rows,
										const int				columns,
										const int				NQ,
										std::vector<int>&		host_dir_ib_iwNQ,
										std::vector<int>&		host_dir_ib_iwNQ_inv,
										std::vector<float>&		host_probability1,
										std::vector<float>&		host_probability2,
										std::vector<float>&		host_permutations
								    )
{
	//->INIT_VARIABLES
#ifdef DEBUG_TIME
	cudaEvent_t										start;
	cudaEvent_t										stop;
	float											elapsedTime		= 0.0f;
#endif
	float											T1				= 0.0f;
	float											T2				= 0.0f;
	int												i				= 0;
	int												r				= 0;
	int												c				= 0;
	int												indexNQxNQ		= 0;
	int												NQ2				= NQ * NQ;
	//->SHORT-HAND_FOR_THRUST_OPERATORS
	thrust::multiplies<float>						op_mult_float;
	thrust::plus<float>								op_plus_float;
	//<-SHORT-HAND_FOR_THRUST_OPERATORS
	thrust::device_vector<int>						dev_dir_ib_iwNQ		( rows * columns * NQ2	);
	thrust::device_vector<int>						dev_dir_ib_iwNQ_inv	( rows * columns * NQ2	);
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
	for( r = 0; r < NQ; r++ )
	{
		for( c = 0; c < NQ; c++ )
		{
			i = r * NQ + c;
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
	host_permutations.resize( rows * columns * NQ2 );
	//<-INIT_VARIABLES
//--------------------------------------------------------------------------------------------------------------
#ifdef DEBUG_TIME
	cudaEventCreate( &start );
	cudaEventCreate( &stop  );
	cudaEventRecord( start, 0 );
#endif

	for( r = 0; r < rows; r++ )
	{
		for( c = 0; c < columns; c++ )
		{
			i				= r * columns;
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
#ifdef SHOW_PERMS
			int row = 0;
			int col = 0;
			thrust::host_vector<float> temp_table2 = temp_table;
			printf( "PERM_TABLE[%i][%i]:\n", r, c );
			for( row = 0; row < NQ; row++ )
			{
				for( col = 0; col < NQ; col++ )
				{
					printf( "%.3f ", temp_table2[ row * NQ + col ] );
				}
				printf( "\n" );
			}
			printf( "\n" );
#endif
		}
	}
//--------------------------------------------------------------------------------------------------------------
#ifdef DEBUG_TIME
	cudaEventRecord( stop, 0 );
	cudaEventSynchronize( stop );
	cudaEventElapsedTime( &elapsedTime, start, stop );
	printf( "PERM_TABLES_CALC_TIME:  %010.6f(s)\n", elapsedTime / 1000.0f );
	//->CLEAN_UP
	cudaEventDestroy( start );
	cudaEventDestroy( stop  );
	//<-CLEAN_UP
#endif
}

/**************************************************************************************************************************************************
Inits Permutation Tables on Device 2:

PARAM NAME				(INITIAL) SIZE		DESCRIPTION
rows					1					number of MDP matrix horizontal lines.
columns					1					number of MDP matrix vertical lines.
NQ						1					number of options to choose from (directions, states, and so on).
host_dir_ib_iwNQ		rows*columns*NQ*NQ	for every MDP cell, in-bounds & not-forbidden values (0-1). Every NQ-chunk repeated NQ times.
host_dir_ib_iwNQ_inv	rows*columns*NQ*NQ	inverse for host_dir_ib_iwNQ.
host_probability1		rows*columns		for every MDP cell, the probability for the most likely option to be chosen.
host_probability2		rows*columns		for every MDP cell, the probability for the least likely option to be chosen.
host_permutations		0					empty vector that will hold the final permutation tables usable by the mdp_iterate_on_device function.
**************************************************************************************************************************************************/

void mdp_init_permutations_on_device2	(	const int				rows,
											const int				columns,
											const int				NQ,
											const float				probability1,
											const float				probability2,
											std::vector<int>&		host_dir_ib_iw,
											std::vector<int>&		host_dir_ib_iw_trans,
											std::vector<float>&		host_permutations
										)
{

	//->INIT_VARIABLES
#ifdef DEBUG_TIME
	cudaEvent_t										start;
	cudaEvent_t										stop;
	float											elapsedTime = 0.0f;
#endif
	
	// AUXILIARY STRUCTURES:
	thrust::minus<float>							op_minus_float;
	thrust::plus<float>								op_plus_float;
	thrust::multiplies<float>						op_mult_float;
	thrust::divides<int>							op_div_int;
	thrust::device_vector<int>::iterator			iterS;
	typedef thrust::device_vector<int>::iterator	Int_Iterator;
	int												NQ2			=	NQ * NQ;
	float											prob_t		=	probability1;
	float											prob_f		=	probability2;

	thrust::device_vector<int>						indices_out				( rows * columns		);
	thrust::device_vector<int>						nqs_NQ					( rows * columns * NQ	);
	thrust::device_vector<int>						indices_NQ				( rows * columns * NQ	);
	thrust::device_vector<int>						indices_out_NQ			( rows * columns * NQ	);
	thrust::device_vector<int>						seq_indices_NQ			( rows * columns * NQ	);
	thrust::device_vector<int>						reachable_temp			( rows * columns * NQ	);

	thrust::device_vector<int>						nqs_NQ2					( rows * columns * NQ2	);
	thrust::device_vector<int>						ones					( rows * columns * NQ2	);
	thrust::device_vector<int>						seq_indices				( rows * columns * NQ2	);
	thrust::device_vector<int>						indices_NQ2				( rows * columns * NQ2	);
	
	thrust::device_vector<int>						in_bounds_is_wall		( rows * columns * NQ2	);
	thrust::device_vector<int>						in_bounds_is_wall_trans	( rows * columns * NQ2	);
	thrust::device_vector<int>						reachable				( rows * columns * NQ2	);

	thrust::device_vector<float>					permutations			( rows * columns * NQ2	);
	thrust::device_vector<float>					temp_table				( rows * columns * NQ2	);
	thrust::device_vector<float>					diagonal				( rows * columns * NQ2	);
	thrust::device_vector<float>					diagonal_neg			( rows * columns * NQ2	);
	thrust::device_vector<float>					diagonal_T1				( rows * columns * NQ2	);
	thrust::device_vector<float>					full_T1					( rows * columns * NQ2	);

#ifdef DEBUG_TIME
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
#endif

	host_permutations.resize( rows * columns * NQ2 );
	thrust::fill			( ones.begin(),					ones.end(),					1																					);
	thrust::fill			( nqs_NQ2.begin(),				nqs_NQ2.end(),				NQ																					);
	thrust::fill			( nqs_NQ.begin(),				nqs_NQ.end(),				NQ																					);
	thrust::sequence		( seq_indices.begin(),			seq_indices.end()																								);
	thrust::sequence		( indices_NQ.begin(),			indices_NQ.end()																								);
	thrust::transform		( seq_indices.begin(),			seq_indices.end(),			nqs_NQ2.begin(),					indices_NQ2.begin(),	op_div_int				);
	thrust::transform		( indices_NQ.begin(),			indices_NQ.end(),			nqs_NQ.begin(),						indices_NQ.begin(),		op_div_int				);
	
	for( iterS = seq_indices_NQ.begin(); iterS < seq_indices_NQ.end(); iterS += NQ )
	{
		thrust::sequence( iterS, iterS + NQ );
	}

	// COLLECT STATE BOUNDARY DATA (COMPUTED IN HOST, COPIED TO DEVICE):
	thrust::copy			( host_dir_ib_iw.begin(),		host_dir_ib_iw.end(),		in_bounds_is_wall.begin()															);
	thrust::copy			( host_dir_ib_iw_trans.begin(), host_dir_ib_iw_trans.end(),	in_bounds_is_wall_trans.begin()														);
	
	// PREPARE PERMUTATIONS MATRIX:
	thrust::reduce_by_key	( indices_NQ2.begin(),			indices_NQ2.end(),			in_bounds_is_wall.begin(),			indices_out_NQ.begin(),	reachable_temp.begin()	);
	repeated_range<Int_Iterator> reachable_range( reachable_temp.begin(), reachable_temp.end(), NQ );
	thrust::copy			( reachable_range.begin(),		reachable_range.end(),		reachable.begin()																	);
	thrust::transform		( reachable.begin(),			reachable.end(),			temp_table.begin(),					false_prob(prob_f)								);
	thrust::transform		( seq_indices.begin(),			seq_indices.end(),			diagonal.begin(),					index_diag(NQ)									);
	thrust::transform		( ones.begin(),					ones.end(),					diagonal.begin(),					diagonal_neg.begin(),	op_minus_float			);
	thrust::transform		( reachable.begin(),			reachable.end(),			full_T1.begin(),					true_prob(prob_t)								);
	thrust::transform		( full_T1.begin(),				full_T1.end(),				diagonal.begin(),					diagonal_T1.begin(),	op_mult_float			);
	thrust::transform		( temp_table.begin(),			temp_table.end(),			diagonal_neg.begin(),				temp_table.begin(),		op_mult_float			);
	thrust::transform		( temp_table.begin(),			temp_table.end(),			diagonal_T1.begin(),				permutations.begin(),	op_plus_float			);
	thrust::transform		( permutations.begin(),			permutations.end(),			in_bounds_is_wall.begin(),			permutations.begin(),	op_mult_float			);		// CLEAN HORIZONTAL
	thrust::transform		( permutations.begin(),			permutations.end(),			in_bounds_is_wall_trans.begin(),	permutations.begin(),	op_mult_float			);		// CLEAN VERTICAL

	// TRANSFER BACK TO DEVICE:
	thrust::copy			( permutations.begin(),			permutations.end(),			host_permutations.begin()															);

#ifdef DEBUG_TIME
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("PERM_TABLES_CALC_TIME:  %010.6f(s)\n", elapsedTime / 1000.0f);
	//->CLEAN_UP
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	//<-CLEAN_UP
#endif

}


/**************************************************************************************************************************************************
	Iterates MDP values on Device until convergence is achieved or the safety limit is reached:

	PARAM NAME			(INITIAL)SIZE		DESCRIPTION
	rows				1					number of MDP matrix horizontal lines.
	columns				1					number of MDP matrix vertical lines.
	NQ					1					number of options to choose from (directions, states, and so on).
	discount			1					discount factor (GAMMA) applicable to this MDP.
	host_vicinityNQ		rows*columns*NQ*NQ	for every MDP cell, holds the index for the cell of each neighbor. Each value is repeated NQ times.
	host_dir_rwNQ		rows*columns*NQ*NQ	for every MDP cell, holds the reward for the cell of each neighbor, distributed amongst NQ values.
	P					rows*columns		for every MDP cell, holds the value of the best policy. Grows by (rows*columns) on every iteration.
	Q					rows*columns*NQ		for every MDP cell, holds the value for every option available at the current iteration.
	V					rows*columns+1		for every MDP cell, holds the best Q amongst the NQ-Qvalues available.
	host_dir_pvNQ		rows*columns*NQ*NQ	for every MDP cell, holds the previous iteration's V value of every neighbouring cell.
	host_permutations	rows*columns*NQ*NQ	Vector holding the permutation tables generated by the 'mdp_init_permutations_on_device' function.
	iteratingTime		1					returns the time elapsed while iterating.
	iterationTimes		0					save time for each iteration.
	convergence			1					returns the convergence confirmation.
**************************************************************************************************************************************************/

int	 mdp_iterate_on_device(	const int				rows,
							const int				columns,
							const int				NQ,
							float					discount,
							std::vector<int>&		host_vicinityNQ,
							std::vector<float>&		host_dir_rwNQ,
							std::vector<int>&		P,
							std::vector<float>&		Q,
							std::vector<float>&		V,
							std::vector<float>&		host_dir_pvNQ,
							std::vector<float>&		host_permutations,
							float&					totalTime,
							std::vector<float>&		iterationTimes,
							bool&					convergence
						  )
{

	int major = THRUST_MAJOR_VERSION;
	int minor = THRUST_MINOR_VERSION;
	printf("Thrust v%i.%i\n", major, minor);
	size_t freeMem, totalMem;
	cuMemGetInfo(&freeMem, &totalMem);
	printf("Free memory %lu, Total memory %lu\n", freeMem, totalMem);

	//->INIT_VARIABLES
	cudaEvent_t										start;
	cudaEvent_t										stop;
	cudaEventCreate( &start );
	cudaEventCreate( &stop  );
#ifdef DEBUG_TIME
	cudaEvent_t										start1;
	cudaEvent_t										stop1;
	cudaEventCreate( &start1 );
	cudaEventCreate( &stop1 );
#endif
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
	int												k				= 0;
	int												mdp_size		= rows * columns;
	int												mdp_sizeNQ		= mdp_size * NQ;
	int												mdp_sizeNQxNQ	= mdp_size * NQ * NQ;
	int												csize			= mdp_size;	// Temp size of dev_P
	float											elapsedTime		= 0.0f;
#ifdef DEBUG_TIME
	float											elapsedTime1	= 0.0f;
#endif
	thrust::device_vector<int>						dev_result			( P.size()						);
	thrust::device_vector<int>						dev_curr_P			( P.size()						);
	thrust::device_vector<int>						dev_prev_P			( P.size()						);
	thrust::device_vector<float>					dev_Q				( Q.size()						);
	thrust::device_vector<float>					dev_V				( V.size()						);
	thrust::device_vector<int>						dev_vicinityNQ		( host_vicinityNQ.size()		);
	thrust::device_vector<float>					dev_dir_rwNQ		( host_dir_rwNQ.size()			);
	thrust::device_vector<float>					dev_dir_pvNQ		( host_dir_pvNQ.size()			);
	thrust::device_vector<float>					dev_permutations	( host_permutations.size()		);
	thrust::device_vector<float>					dev_prob_tables		( mdp_sizeNQxNQ					);
	thrust::device_vector<float>					dev_discountNQxNQ	( mdp_sizeNQxNQ					);
	thrust::device_vector<int>						seq_indicesQ		( mdp_sizeNQ					);		// Helper for key-reduction.
	thrust::device_vector<int>						indicesNQxNQ		( mdp_sizeNQxNQ					);		// Helper for key-reduction.
	thrust::device_vector<int>						indicesNQ			( mdp_sizeNQ					);		// Helper for key-reduction.
	thrust::device_vector<int>						indices_outNQxNQ	( Q.size()						);		// Helper for key-reduction.
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
//--------------------------------------------------------------------------------------------------------------
	//->TRANSFER_ALL_DATA_TO_DEVICE
	//thrust::copy( P.begin(),					P.end(),					dev_curr_P.begin()			);
	//thrust::copy( P.begin(),			        P.end(),					dev_prev_P.begin()			);
	thrust::fill( dev_prev_P.begin(),			dev_prev_P.end(),			0							);
	thrust::copy( Q.begin(),					Q.end(),					dev_Q.begin()				);
	thrust::copy( V.begin(),					V.end(),					dev_V.begin()				);
	thrust::copy( host_vicinityNQ.begin(),		host_vicinityNQ.end(),		dev_vicinityNQ.begin()		);
	thrust::copy( host_dir_rwNQ.begin(),		host_dir_rwNQ.end(),		dev_dir_rwNQ.begin()		);
	thrust::copy( host_dir_pvNQ.begin(),		host_dir_pvNQ.end(),		dev_dir_pvNQ.begin()		);
	thrust::copy( host_permutations.begin(),	host_permutations.end(),	dev_permutations.begin()	);
	totalTime		= 0.0f;
	convergence		= false;

	//<-TRANSFER_ALL_DATA_TO_DEVICE
//--------------------------------------------------------------------------------------------------------------
	//->WORK_ON_DEVICE
	while( !convergence && k < ITER_SAFETY )
	{
		cudaEventRecord( start, 0 );
//--------------------------------------------------------------------------------------------------------------
		//->PERMUTATIONS_VS_PREV_VALUES_MULTIPLICATION
#ifdef DEBUG_TIME
		cudaEventRecord( start1, 0 );
#endif
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
#ifdef DEBUG_TIME
		cudaEventRecord( stop1, 0 );
		cudaEventSynchronize( stop1 );
		cudaEventElapsedTime( &elapsedTime1, start1, stop1 );
		//printf( "[%03i] MAD_TIME:     %010.6f(s)\n", k, elapsedTime1 / 1000.0f );
#endif
		//<-PERMUTATIONS_VS_PREV_VALUES_MULTIPLICATION
//--------------------------------------------------------------------------------------------------------------
		//->OBTAINING_QS
#ifdef DEBUG_TIME
		cudaEventRecord( start1, 0 );
#endif
		thrust::reduce_by_key	(	indicesNQxNQ.begin(),
									indicesNQxNQ.end(),
									dev_prob_tables.begin(),
									indices_outNQxNQ.begin(),
									dev_Q.begin()				);
		thrust::replace			(	dev_Q.begin(),
									dev_Q.end(),
									0.0f,
									-100.0f						);
#ifdef DEBUG_TIME
		cudaEventRecord( stop1, 0 );
		cudaEventSynchronize( stop1 );
		cudaEventElapsedTime( &elapsedTime1, start1, stop1 );
		//printf( "[%03i]+Q_CALC_TIME:  %010.6f(s)\n", k, elapsedTime1 / 1000.0f );
#endif
		//<-OBTAINING_QS
//--------------------------------------------------------------------------------------------------------------
		//->OBTAINING_BEST_QS
#ifdef DEBUG_TIME
		cudaEventRecord( start1, 0 );
#endif
		// Technique			= key-reduction over zip-iterator <float,int>
		// zip_iterator<float>  = devQ:			the just obtained Q-values.
		// zip_iterator<int>    = seq_indicesQ: 0,1,2,3,...,N-1,0123...		Holds the bestP number.
		// key_reduction values = indices8:	    0000000011111111...			Holds the keys for reduction.

		ZipIterator firstIn = thrust::make_zip_iterator(thrust::make_tuple(dev_Q.begin(), seq_indicesQ.begin()));
		ZipIterator firstOut = thrust::make_zip_iterator(thrust::make_tuple(dev_V.begin(), dev_curr_P.begin()));

		try{
			thrust::reduce_by_key(indicesNQ.begin(),
				indicesNQ.end(),
				firstIn,
				indices_outNQ.begin(),
				firstOut,
				zi_binary_pred,
				zi_binary_op);
		}
		catch (thrust::system_error &e)
		{
			printf("ERROR: %s\n", e.what());
		}

		csize += mdp_size;
#ifdef DEBUG_TIME
		cudaEventRecord( stop1, 0 );
		cudaEventSynchronize( stop1 );
		cudaEventElapsedTime( &elapsedTime1, start1, stop1 );
		//printf( "[%03i]+BEST_Q_TIME:  %010.6f(s)\n", k, elapsedTime1 / 1000.0f );
#endif
		//<-OBTAINING_BEST_QS
//--------------------------------------------------------------------------------------------------------------
		//->CHECK_CONVERGENCE
#ifdef SHOW_ITERS
		printf( "\n" );
		thrust::host_vector<int> tempP( mdp_size );
		thrust::host_vector<float> tempV( mdp_size );
		thrust::copy( dev_curr_P.begin(), dev_curr_P.end(), tempP.begin() );
		thrust::copy( dev_V.begin(), dev_V.begin() + mdp_size, tempV.begin() );
		for( int r = (rows - 1); r >= 0; r-- )
		{
			for( int c = 0; c < columns; c++ )
			{
				int index = r * columns + c;
				printf( "(%07.3f) %i", tempV[ index ], tempP[ index ] );
				if( (c + 1) < columns )
				{
					printf( "\t" );
				}
			}
			printf( "\n" );
		}
		printf( "\n" );
#endif

		//printf( "[%03i] Before thrust::transform...i1=%03i\n", k, i1 );
		// Subtract current policy from previous policy.
		// If convergence is achieved, the 'dev_result' vector has only zeroes:
		thrust::transform(dev_curr_P.begin(), dev_curr_P.end(), dev_prev_P.begin(), dev_result.begin(), op_minus_int);



		try
		{
			//thrust::copy(dev_curr_P.begin(), dev_curr_P.end(), dev_prev_P.begin());
			thrust::transform(dev_curr_P.begin(), dev_curr_P.end(), dev_prev_P.begin(), CopyFunctor());
		}
		catch (thrust::system_error &e)
		{
			printf("ERROR: %s\n", e.what());
		}

		//printf( "[%03i] Before thrust::count...\n", k );
		// Count the zeroes in 'dev_result' to determine if convergence is achieved:
		int numZeroes = thrust::count(dev_result.begin(), dev_result.end(), 0);
		if( numZeroes == mdp_size )
		{
			cudaEventRecord( stop, 0 );
			cudaEventSynchronize( stop );
			cudaEventElapsedTime( &elapsedTime, start, stop );
			totalTime += elapsedTime;
			iterationTimes.push_back( elapsedTime );
			convergence = true;
		}
		else
		{
			//->UPDATE_DEV_DIR_PV
#ifdef DEBUG_TIME
			cudaEventRecord( start1, 0 );
#endif
			//printf( "[%03i] Before thrust::gather...\n", k );
			// Perform the update with a map-value operation:
			thrust::gather( dev_vicinityNQ.begin(),	// map_begins
							dev_vicinityNQ.end(),	// map_ends
							dev_V.begin(),			// values
							dev_dir_pvNQ.begin()	);
#ifdef DEBUG_TIME
			cudaEventRecord( stop1, 0 );
			cudaEventSynchronize( stop1 );
			cudaEventElapsedTime( &elapsedTime1, start1, stop1 );
			//printf( "[%03i]+GATHER_TIME:  %010.6f(s)\n", k, elapsedTime1 / 1000.0f );
#endif
			//<-UPDATE_DEV_DIR_PV
			cudaEventRecord( stop, 0 );
			cudaEventSynchronize( stop );
			cudaEventElapsedTime( &elapsedTime, start, stop );
			totalTime += elapsedTime;
			iterationTimes.push_back( elapsedTime );
			//printf( "." );
			k++;
		}
		//<-CHECK_CONVERGENCE
//--------------------------------------------------------------------------------------------------------------
	}
	//<-WORK_ON_DEVICE
//--------------------------------------------------------------------------------------------------------------
#ifdef DEBUG_TIME
		printf( "ITERATING_TIME:     %010.6f(s)\n", totalTime / 1000.0f );
#endif
	printf( "\n" );
	//->TRANSFER_BACK_TO_HOST
	//P.resize( dev_P.size() );
	//thrust::copy( dev_P.begin(), dev_P.end(), P.begin() );
	thrust::copy(dev_curr_P.begin(), dev_curr_P.end(), P.begin());
	thrust::copy( dev_V.begin(), dev_V.end(), V.begin() );
	//<-TRANSFER_BACK_TO_HOST
	//->CLEAN_UP
	cudaEventDestroy( start );
	cudaEventDestroy( stop  );
#ifdef DEBUG_TIME
	cudaEventDestroy( start1 );
	cudaEventDestroy( stop1  );
#endif
	//<-CLEAN_UP
	return k;
}

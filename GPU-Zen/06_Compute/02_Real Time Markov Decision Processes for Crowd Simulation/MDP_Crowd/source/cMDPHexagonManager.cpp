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
#include "cMDPHexagonManager.h"
#ifdef __unix
	#include <sys/time.h>
#endif

//=======================================================================================
//
MDPHexagonManager::MDPHexagonManager( LogManager* log_manager_ ) : NQ( 6 )
{
	log_manager			= log_manager_;
	cells				= 0;
	rows				= 0;
	columns				= 0;
	discount			= 0.98f;
	convergence			= false;
	iteratingTime		= 0.0f;
	iterations			= 0;
}
//
//=======================================================================================
//
MDPHexagonManager::~MDPHexagonManager( void )
{

}
//
//=======================================================================================
//
bool MDPHexagonManager::solve_from_csv( string& csvFilename, vector<float>& policy )
{
	cells				= 0;
	rows				= 0;
	columns				= 0;
	discount			= 0.98f;
	convergence			= false;
	iteratingTime		= 0.0f;

	log_manager->log( LogManager::MDP, "HEXAGON_MDP::SOLVING \"%s\"...", csvFilename.c_str() );

	string csv( csvFilename );
	rewards = read_matrix( csv );
	//print_rewards();

//->INIT_MDP_ON_HOST
	float iet_cpu, iet_gpu;
#if defined _WIN32
	DWORD global_tickAtStart = GetTickCount();							// start global_timer
	DWORD init_tickAtStart = GetTickCount();							// start init_timer
#elif defined __unix
	timeval global_start, start, stop, result_cpu, result_gpu, result_ite, global_res;
	gettimeofday( &global_start, NULL );
	gettimeofday( &start, NULL );
#endif
	init_mdp();
#if defined _WIN32
	DWORD init_elapsedTicks = GetTickCount() - init_tickAtStart;		// stop init_timer
	iet_cpu = (float)init_elapsedTicks;
#elif defined __unix
	gettimeofday( &stop, NULL );
	timersub( &stop, &start, &result_cpu );
	iet_cpu = (float)result_cpu.tv_sec*1000.0f+(float)result_cpu.tv_usec/1000.0f;
#endif
//<-INIT_MDP_ON_HOST

//->INIT_MDP_ON_GPU
#ifdef _WIN32
	init_tickAtStart = GetTickCount();									// start init_timer
#elif defined __unix
	gettimeofday( &start, NULL );
#endif
	//->INIT_PERMUTATION_TABLES_ON_GPU
	mdp_init_permutations_on_device(	rows,
										columns,
										NQ,
										host_dir_ib_iw6,
										host_dir_ib_iw6_inv,
										host_probability1,
										host_probability2,
										host_permutations	);
	//<-INIT_PERMUTATION_TABLES_ON_GPU
#ifdef _WIN32
	init_elapsedTicks = GetTickCount() - init_tickAtStart;				// stop init_timer
	iet_gpu = (float)init_elapsedTicks;
#elif defined __unix
	gettimeofday( &stop, NULL );
	timersub( &stop, &start, &result_gpu );
	iet_gpu = (float)result_gpu.tv_sec*1000.0f+(float)result_gpu.tv_usec/1000.0f;
#endif
//<-INIT_MDP_ON_GPU

	float iit, get;
//->SOLVE_MDP_ON_GPU
#ifdef _WIN32
	DWORD iter_tickAtStart = GetTickCount();							// start iter_timer
#elif defined __unix
	gettimeofday( &start, NULL );
#endif
	iterations = mdp_iterate_on_device(	rows,
										columns,
										NQ,
										discount,
										host_vicinity6,
										host_dir_rw6,
										P,
										Q,
										V,
										host_dir_pv6,
										host_permutations,
										iteratingTime,
										iterationTimes,
										convergence );
#ifdef _WIN32
	DWORD iter_elapsedTicks = GetTickCount() - iter_tickAtStart;		// stop iter_timer
	iit = (float)iter_elapsedTicks;

	DWORD global_elapsedTicks = GetTickCount() - global_tickAtStart;	// stop global_timer
	get = (float)global_elapsedTicks;
#elif defined __unix
	gettimeofday( &stop, NULL );
	timersub( &stop, &start, &result_ite );
	iit = (float)result_ite.tv_sec*1000.0f+(float)result_ite.tv_usec/1000.0f;

	timersub( &stop, &global_start, &global_res );
	get = (float)global_res.tv_sec*1000.0f+(float)global_res.tv_usec/1000.0f;
#endif
//<-SOLVE_MDP_ON_GPU

//->SHOW_RESULTS
	//print_mdp();
	//write_policy();
	float sum = 0.0f;
	for( int i = 0; i < (int)iterationTimes.size(); i++ )
	{
		sum += iterationTimes[i];
	}
	float mean = (sum / (float)iterationTimes.size()) / 1000.0f;
	log_manager->log( LogManager::MDP, "%010.6fs  MEAN_ITERATION_TIME.", mean 				);
	log_manager->log( LogManager::MDP, "%010.6fs  INIT_TIME@CPU.", 		iet_cpu / 1000.0f 	);
	log_manager->log( LogManager::MDP, "%010.6fs  INIT_TIME@GPU.", 		iet_gpu / 1000.0f 	);
	log_manager->log( LogManager::MDP, "%010.6fs  ITERATING_TIME.", 	iit / 1000.0f 		);
	log_manager->log( LogManager::MDP, "%010.6fs  TOTAL_TIME_ELAPSED.", get / 1000.0f 		);
//<-SHOW_RESULTS
	get_policy( policy );
	return convergence;
}
//
//=======================================================================================
//
int MDPHexagonManager::getRows( void )
{
	return rows;
}
//
//=======================================================================================
//
int MDPHexagonManager::getColumns( void )
{
	return columns;
}
//
//=======================================================================================
//
int MDPHexagonManager::getIterations( void )
{
	return iterations;
}
//
//=======================================================================================
//
bool MDPHexagonManager::getConvergence( void )
{
	return convergence;
}
//
//=======================================================================================
//
bool MDPHexagonManager::is_even( int number )
{
	return (number % 2 == 0);
}
//
//=======================================================================================
//
bool MDPHexagonManager::is_odd( int number )
{
	return (number % 2 == 1);
}
//
//=======================================================================================
//
bool MDPHexagonManager::in_bounds( int r, int c, int q )
{
	bool result = false;
	switch( q )
	{
		case 0:	// UP-LEFT
			if( is_odd( c ) )
			{
				result = true;
			}
			else
			{
				if( (r+1) < rows && c > 0 )
				{
					result = true;
				}
			}
			break;
		case 1:	// DOWN-LEFT
			if( is_odd( c ) )
			{
				if( r > 0 )
				{
					result = true;
				}
			}
			else
			{
				if( c > 0 )
				{
					result = true;
				}
			}
			break;
		case 2:	// DOWN
			if( r > 0 )
			{
				result = true;
			}
			break;
		case 3:	// DOWN-RIGHT
			if( is_odd( c ) )
			{
				if( r > 0 && (c+1) < columns )
				{
					result = true;
				}
			}
			else
			{
				if( (c+1) < columns )
				{
					result = true;
				}
			}
			break;
		case 4:	// UP-RIGHT
			if( is_odd( c ) )
			{
				if( (c+1) < columns )
				{
					result = true;
				}
			}
			else
			{
				if( (r+1) < rows && (c+1) < columns )
				{
					result = true;
				}
			}
			break;
		case 5:	// UP
			if( (r+1) < rows )
			{
				result = true;
			}
			break;
	}
	return result;
}
//
//=======================================================================================
//
int MDPHexagonManager::mdp_get_index( int row, int column, int iteration )
{
	int index = (rows*columns*iteration)+(row*columns+column);
	return index;
}
//
//=======================================================================================
//
int MDPHexagonManager::mdp_get_qindex( int row, int column, int iteration, int q )
{
	int index = (rows*columns*NQ*iteration)+((row*columns*NQ)+(column*NQ)+q);
	return index;
}
//
//=======================================================================================
//
int MDPHexagonManager::mdp_get_index( int row, int column, int iteration, int direction )
{
	int r1 = row;
	int c1 = column;
	switch( direction )
	{
		case 0:	//UP_LEFT
			if( is_odd( column ) )
			{
				c1 = column - 1;
			}
			else
			{
				if( (row+1) < rows && column > 0 )
				{
					r1 = row + 1;
					c1 = column - 1;
				}
			}
			break;
		case 1:	//DOWN_LEFT
			if( is_odd( column ) )
			{
				if( row > 0 )
				{
					r1 = row - 1;
					c1 = column - 1;
				}
			}
			else
			{
				if( column > 0 )
				{
					c1 = column - 1;
				}
			}
			break;
		case 2:	//DOWN
			if( row > 0 )
			{
				r1 = row - 1;
			}
			break;
		case 3:	//DOWN_RIGHT
			if( is_odd( column ) )
			{
				if( row > 0 && (column+1) < columns )
				{
					r1 = row - 1;
					c1 = column + 1;
				}
			}
			else
			{
				if( (column+1) < columns )
				{
					c1 = column + 1;
				}
			}
			break;
		case 4:	//UP_RIGHT
			if( is_odd( column ) )
			{
				if( (column+1) < columns )
				{
					c1 = column + 1;
				}
			}
			else
			{
				if( (row+1) < rows && (column+1) < columns )
				{
					r1 = row + 1;
					c1 = column + 1;
				}
			}
			break;
		case 5:	//UP
			if( (row+1) < rows )
			{
				r1 = row + 1;
			}
			break;
	}
	int index = mdp_get_index( r1, c1, iteration );
	return index;
}
//
//=======================================================================================
//
int MDPHexagonManager::mdp_get_qindex( int row, int column, int iteration, int direction, int q )
{
	int r1 = row;
	int c1 = column;
	switch( direction )
	{
		case 0:	//UP_LEFT
			if( is_odd( column ) )
			{
				c1 = column - 1;
			}
			else
			{
				if( (row+1) < rows && column > 0 )
				{
					r1 = row + 1;
					c1 = column - 1;
				}
			}
			break;
		case 1:	//DOWN_LEFT
			if( is_odd( column ) )
			{
				if( row > 0 )
				{
					r1 = row - 1;
					c1 = column - 1;
				}
			}
			else
			{
				if( column > 0 )
				{
					c1 = column - 1;
				}
			}
			break;
		case 2:	//DOWN
			if( row > 0 )
			{
				r1 = row - 1;
			}
			break;
		case 3:	//DOWN_RIGHT
			if( is_odd( column ) )
			{
				if( row > 0 && (column+1) < columns )
				{
					r1 = row - 1;
					c1 = column + 1;
				}
			}
			else
			{
				if( (column+1) < columns )
				{
					c1 = column + 1;
				}
			}
			break;
		case 4:	//UP_RIGHT
			if( is_odd( column ) )
			{
				if( (column+1) < columns )
				{
					c1 = column + 1;
				}
			}
			else
			{
				if( (row+1) < rows && (column+1) < columns )
				{
					r1 = row + 1;
					c1 = column + 1;
				}
			}
			break;
		case 5:	//UP
			if( (row+1) < rows )
			{
				r1 = row + 1;
			}
			break;
	}
	int index = mdp_get_qindex( r1, c1, iteration, q );
	return index;
}
//
//=======================================================================================
//
//->FOR_INT_VECTORS
int MDPHexagonManager::mdp_get_value( int row, int column, int iteration, vector<int>& mdp_vector )
{
	//return mdp_vector[mdp_get_index( row, column, iteration )];
	return mdp_vector[mdp_get_index(row, column, 0)];
}
//
//=======================================================================================
//
int MDPHexagonManager::mdp_get_value( int row, int column, int iteration, int direction, vector<int>& mdp_vector )
{
	return mdp_vector[mdp_get_index( row, column, iteration, direction )];
}
int MDPHexagonManager::mdp_get_qvalue( int row, int column, int iteration, int direction, int q, vector<int>& mdp_vector )
{
	return mdp_vector[mdp_get_qindex( row, column, iteration, direction, q )];
}
//<-FOR_INT_VECTORS
//
//=======================================================================================
//
//->FOR_FLOAT_VECTORS
float MDPHexagonManager::mdp_get_value( int row, int column, int iteration, vector<float>& mdp_vector )
{
	return mdp_vector[mdp_get_index( row, column, iteration )];
}
//
//=======================================================================================
//
float MDPHexagonManager::mdp_get_value( int row, int column, int iteration, int direction, vector<float>& mdp_vector )
{
	return mdp_vector[mdp_get_index( row, column, iteration, direction )];
}
//
//=======================================================================================
//
float MDPHexagonManager::mdp_get_qvalue( int row, int column, int iteration, int direction, int q, vector<float>& mdp_vector )
{
	return mdp_vector[mdp_get_qindex( row, column, iteration, direction, q )];
}
//<-FOR_FLOAT_VECTORS
//
//=======================================================================================
//
bool MDPHexagonManager::mdp_is_wall( int row, int column )
{
	if( mdp_get_value( row, column, 0, rewards ) == -10.0f )
	{
		return true;
	}
	else
	{
		return false;
	}
}
//
//=======================================================================================
//
bool MDPHexagonManager::mdp_is_exit( int row, int column )
{
	if( mdp_get_value( row, column, 0, rewards ) > 0.0f )
	{
		return true;
	}
	else
	{
		return false;
	}
}
//
//=======================================================================================
//
void MDPHexagonManager::split( const string& str, const string& delimiters, vector<string>& tokens )
{
    // Skip delimiters at beginning.
    string::size_type lastPos = str.find_first_not_of( delimiters, 0 );
    // Find first "non-delimiter".
    string::size_type pos     = str.find_first_of( delimiters, lastPos );

    while ( string::npos != pos || string::npos != lastPos )
    {
        // Found a token, add it to the vector.
		string tok( str.substr( lastPos, pos - lastPos ) );
        tokens.push_back( tok );
        // Skip delimiters.  Note the "not_of"
        lastPos = str.find_first_not_of( delimiters, pos );
        // Find next "non-delimiter"
        pos = str.find_first_of( delimiters, lastPos );
    }
}
//
//=======================================================================================
//
vector<float> MDPHexagonManager::read_matrix( string file_name )
{
	vector<vector<float>> matrix_2d;
	vector<float> matrix;
	ifstream thefile( file_name.c_str() );
	string line;
    while( getline( thefile, line ) )
    {
		columns = 0;
		vector<string> parts;
		split( line, string(","), parts );
		vector<float> vals;
		for( unsigned int v = 0; v < parts.size(); v++ )
		{
			istringstream iss1( parts[v].c_str() );
			float VAL;
			iss1 >> VAL;
			vals.push_back( VAL );
			cells++;
			columns++;
		}
		matrix_2d.push_back( vals );
		rows++;
    }
	thefile.close();
	for( int row = (rows-1); row >= 0; row-- )
	{
		for( int column = 0; column < columns; column++ )
		{
			matrix.push_back( matrix_2d[row][column] );
		}
	}
	return matrix;
}
//
//=======================================================================================
//
void MDPHexagonManager::write_policy( void )
{
	string fname( "policy_" );
	string num;
	stringstream snum;
	snum << iterations;

	fname.append( snum.str() );
	fname.append( ".csv" );
	ofstream out( fname.c_str() );
	for( int r = (rows-1); r >= 0; r-- )
	{
		for( int c = 0; c < columns; c++ )
		{
			out << P[mdp_get_index(r,c,iterations)];
			if( (c+1) < columns )
			{
				out << ",";
			}
		}
		out << "\n";
	}
	out.close();
}
//
//=======================================================================================
//
void MDPHexagonManager::print_rewards( void )
{
	printf( "\nINPUT_MATRIX[%i][%i]\n", rows, columns );
	for( int i = (rows-1); i >= 0; i-- )
	{
		for( int j = 0; j < columns; j++ )
		{
			float v = mdp_get_value(i,j,0,rewards);
			if( v == -0.04f )		// SPACE
			{
				printf( "   " );
			}
			else if( v == -10.0f )	// WALL
			{
				printf( "%c%c%c", 219,219,219 );
			}
			else if( v == 1.0 )		// EXIT
			{
				printf( "%c%c%c", 177,177,177 );
			}
			else
			{
				printf( "%03.2f ", v );
			}
		}
		printf( "\n" );
	}
	printf( "\n" );
}
//
//=======================================================================================
//
void MDPHexagonManager::print_mdp( void )
{
	printf( "POLICY[k=%i][convergence=", iterations );
	if( convergence )
	{
		printf( "TRUE" );
	}
	else
	{
		printf( "FALSE" );
	}
	printf( "]:\n" );

	for( int r = (rows-1); r >= 0; r-- )
	{
		for( int c = 0; c < columns; c++ )
		{
			if( mdp_is_wall(r,c) )
			{
				printf( "%c%c%c", 219,219,219,219 );
			}
			else
			{
				int p = mdp_get_value(r,c,iterations,P);
				if( p == mdp_get_value(r,c,iterations-1,P) )
				{
					switch ( p )
					{
						case 0:		// UL
							printf( " %c ", 218 );
							break;
						case 1:		// DL
							printf( " %c ", 192 );
							break;
						case 2:		// DD
							printf( " %c ", 118 );
							break;
						case 3:		// DR
							printf( " %c ", 217 );
							break;
						case 4:		// UR
							printf( " %c ", 191 );
							break;
						case 5:		// UU
							printf( " %c ", 94 );
							break;
						default:	// ??
							printf( " %i ", p );
							break;
					}
				}
				else
				{
					switch ( p )
					{
						case 0:		// UL
							printf( "[%c]", 218 );
							break;
						case 1:		// DL
							printf( "[%c]", 192 );
							break;
						case 2:		// DD
							printf( "[%c]", 118 );
							break;
						case 3:		// DR
							printf( "[%c]", 217 );
							break;
						case 4:		// UR
							printf( "[%c]", 191 );
							break;
						case 5:		// UU
							printf( "[%c]", 94 );
							break;
						default:	// ??
							printf( "[%i]", p );
							break;
					}
				}
			}
		}
		printf( "\n" );
	}
}
//
//=======================================================================================
//
void MDPHexagonManager::get_policy( vector<float>& policy )
{
	policy.clear();
	//printf( "\n\nFILLING_POLICY:\n" );
	for( int r = (rows-1); r >= 0; r-- )
	{
		for( int c = 0; c < columns; c++ )
		{
			if( mdp_is_wall(r,c) )
			{
				policy.push_back( (float)NQ );
				//printf( "%.0f\t", 8.0f );
			}
			else if( mdp_is_exit(r,c) )
			{
				policy.push_back( (float)(NQ+1) );
			}
			else
			{
				int p = mdp_get_value(r,c,iterations,P);
				policy.push_back( (float)p );
				//printf( "%.0f\t", (float)p );
			}
		}
		//printf( "\n" );
	}
	//printf( "\n" );
}
//
//=======================================================================================
//
void MDPHexagonManager::init_mdp( void )
{
	reachable.clear();
	Q.clear();
	V.clear();
	P.clear();
	host_probability1.clear();
	host_probability2.clear();
	host_permutations.clear();
	host_vicinity.clear();
	host_vicinity6.clear();
	iterationTimes.clear();
	host_dir_rw_sieve.clear();
	host_dir_rw_shifts.clear();
	host_dir_pv_sieve.clear();
	host_dir_pv_shifts.clear();
	host_dir_ib_sieve.clear();
	host_dir_ib_shifts.clear();
	host_dir_iw_sieve.clear();
	host_dir_iw_shifts.clear();
	host_dir_rw.clear();
	host_dir_rw6.clear();
	host_dir_pv.clear();
	host_dir_pv6.clear();
	host_dir_ib.clear();
	host_dir_iw.clear();
	host_dir_ib_iw.clear();
	host_dir_ib_iw6.clear();
	host_dir_ib_iw6_inv.clear();

	int sieve_rows = rows + 2;
	int sieve_columns = columns + 2;
	//->FILL_THE_SIEVES
	for( int c = 0; c < sieve_columns; c++ )
	{
		host_dir_rw_sieve.push_back( 0.0f );
		host_dir_pv_sieve.push_back( 0.0f );
		host_dir_ib_sieve.push_back( 0 );
		host_dir_iw_sieve.push_back( 0 );
	}
	for( int r = 0; r < rows; r++ )
	{
		host_dir_rw_sieve.push_back( 0.0f );
		host_dir_pv_sieve.push_back( 0.0f );
		host_dir_ib_sieve.push_back( 0 );
		host_dir_iw_sieve.push_back( 0 );
		for( int i = (r*columns); i < (r*columns)+columns; i++ )
		{
			host_dir_rw_sieve.push_back( rewards[i] );
			host_dir_pv_sieve.push_back( rewards[i] );
			host_dir_ib_sieve.push_back( 1 );
			if( rewards[i] == -10.0f )
			{
				host_dir_iw_sieve.push_back( 1 );
			}
			else
			{
				host_dir_iw_sieve.push_back( 0 );
			}
		}
		host_dir_rw_sieve.push_back( 0.0f );
		host_dir_pv_sieve.push_back( 0.0f );
		host_dir_ib_sieve.push_back( 0 );
		host_dir_iw_sieve.push_back( 0 );
	}
	for( int c = 0; c < sieve_columns; c++ )
	{
		host_dir_rw_sieve.push_back( 0.0f );
		host_dir_pv_sieve.push_back( 0.0f );
		host_dir_ib_sieve.push_back( 0 );
		host_dir_iw_sieve.push_back( 0 );
	}
	//<-FILL_THE_SIEVES

	//->CREATE_THE_SHIFTS
	for( int s = 0; s < NQ; s++ )
	{
		vector<float> rw_shift;
		host_dir_rw_shifts.push_back( rw_shift );
		vector<float> pv_shift;
		host_dir_pv_shifts.push_back( pv_shift );
		vector<int> ib_shift;
		host_dir_ib_shifts.push_back( ib_shift );
		vector<int> iw_shift;
		host_dir_iw_shifts.push_back( iw_shift );
	}
	//<-CREATE_THE_SHIFTS

	//->FILL_THE_SHIFTS
	for( int row = 0; row < sieve_rows; row++ )
	{
		for( int column = 0; column < sieve_columns; column++ )
		{
			int		index = row*sieve_columns+column;
			float	rw_val = host_dir_rw_sieve[index];
			float	pv_val = host_dir_pv_sieve[index];
			int		ib_val = host_dir_ib_sieve[index];
			int		iw_val = host_dir_iw_sieve[index];

			int		shifted_index	= (row+1)*sieve_columns+column;
			float	shifted_rw_val	= 0.0f;
			float	shifted_pv_val	= 0.0f;
			int		shifted_ib_val	= 0;
			int		shifted_iw_val	= 0;

			if( (row+1) < sieve_rows )
			{
				shifted_rw_val = host_dir_rw_sieve[shifted_index];
				shifted_pv_val = host_dir_pv_sieve[shifted_index];
				shifted_ib_val = host_dir_ib_sieve[shifted_index];
				shifted_iw_val = host_dir_iw_sieve[shifted_index];
			}

			if( row > 0 && row < (sieve_rows - 1) && column < (sieve_columns - 2) )		// UL candidate
			{
				if( is_even( column ) )
				{
					host_dir_rw_shifts[0].push_back( shifted_rw_val );
					host_dir_pv_shifts[0].push_back( shifted_pv_val );
					host_dir_ib_shifts[0].push_back( shifted_ib_val );
					host_dir_iw_shifts[0].push_back( shifted_iw_val );
				}
				else
				{
					host_dir_rw_shifts[0].push_back( rw_val );
					host_dir_pv_shifts[0].push_back( pv_val );
					host_dir_ib_shifts[0].push_back( ib_val );
					host_dir_iw_shifts[0].push_back( iw_val );
				}
			}
			if( row < (sieve_rows - 2) && column < (sieve_columns - 2) )				// DL candidate
			{
				if( is_even( column ) )
				{
					host_dir_rw_shifts[1].push_back( shifted_rw_val );
					host_dir_pv_shifts[1].push_back( shifted_pv_val );
					host_dir_ib_shifts[1].push_back( shifted_ib_val );
					host_dir_iw_shifts[1].push_back( shifted_iw_val );
				}
				else
				{
					host_dir_rw_shifts[1].push_back( rw_val );
					host_dir_pv_shifts[1].push_back( pv_val );
					host_dir_ib_shifts[1].push_back( ib_val );
					host_dir_iw_shifts[1].push_back( iw_val );
				}
			}
			if( row < (sieve_rows - 2) && column > 0 && column < (sieve_columns - 1) )	// DD candidate
			{
				host_dir_rw_shifts[2].push_back( rw_val );
				host_dir_pv_shifts[2].push_back( pv_val );
				host_dir_ib_shifts[2].push_back( ib_val );
				host_dir_iw_shifts[2].push_back( iw_val );
			}
			if( row < (sieve_rows - 2) && column > 1 )									// DR candidate
			{
				if( is_even( column ) )
				{
					host_dir_rw_shifts[3].push_back( shifted_rw_val );
					host_dir_pv_shifts[3].push_back( shifted_pv_val );
					host_dir_ib_shifts[3].push_back( shifted_ib_val );
					host_dir_iw_shifts[3].push_back( shifted_iw_val );
				}
				else
				{
					host_dir_rw_shifts[3].push_back( rw_val );
					host_dir_pv_shifts[3].push_back( pv_val );
					host_dir_ib_shifts[3].push_back( ib_val );
					host_dir_iw_shifts[3].push_back( iw_val );
				}
			}
			if( row > 0 && row < (sieve_rows - 1) && column > 1 )						// UR candidate
			{
				if( is_even( column ) )
				{
					host_dir_rw_shifts[4].push_back( shifted_rw_val );
					host_dir_pv_shifts[4].push_back( shifted_pv_val );
					host_dir_ib_shifts[4].push_back( shifted_ib_val );
					host_dir_iw_shifts[4].push_back( shifted_iw_val );
				}
				else
				{
					host_dir_rw_shifts[4].push_back( rw_val );
					host_dir_pv_shifts[4].push_back( pv_val );
					host_dir_ib_shifts[4].push_back( ib_val );
					host_dir_iw_shifts[4].push_back( iw_val );
				}
			}
			if( row > 1 && column > 0 && column < (sieve_columns - 1) )					// UU candidate
			{
				host_dir_rw_shifts[5].push_back( rw_val );
				host_dir_pv_shifts[5].push_back( pv_val );
				host_dir_ib_shifts[5].push_back( ib_val );
				host_dir_iw_shifts[5].push_back( iw_val );
			}
		}
	}
	//<-FILL_THE_SHIFTS

	//->FILL_USABLE_VECTORS
	//printf( "VICINITY:\n" );
	for( int r = 0; r < rows; r++ )
	{
		for( int c = 0; c < columns; c++ )
		{
			int index = (r*columns)+c;
			int index6  = (r*columns*NQ)+(c*NQ);
			for( int q = 0; q < NQ; q++ )
			{
				host_dir_rw.push_back( host_dir_rw_shifts[q][index] );
				float rw6 = host_dir_rw_shifts[q][index] / (float)NQ;
				for( int w = 0; w < NQ; w++ )
				{
					host_dir_rw6.push_back( rw6 );
				}
				host_dir_pv.push_back( host_dir_pv_shifts[q][index] );
				host_dir_ib.push_back( host_dir_ib_shifts[q][index] );
				host_dir_iw.push_back( host_dir_iw_shifts[q][index] );
				host_dir_ib_iw.push_back( host_dir_ib_shifts[q][index] - host_dir_iw_shifts[q][index] );
				Q.push_back( 0 );
				if( in_bounds( r, c, q ) )
				{
					host_vicinity.push_back( mdp_get_index( r, c, 0, q ) );
				}
				else
				{
					host_vicinity.push_back( rows*columns );
				}
				//printf( "%3i,", host_vicinity[ host_vicinity.size() - 1 ] );
			}
			//printf( "\t" );
			for( int i = 0; i < NQ; i++ )
			{
				for( int j = index6; j < (index6+NQ); j++ )
				{
					host_vicinity6.push_back( host_vicinity[j] );
					host_dir_pv6.push_back( host_dir_pv[j] );
				}
			}
			V.push_back( rewards[index] );
			P.push_back( -1 );
		}
		//printf( "\n" );
	}
	//printf( "\n" );
	int reach_count = 0;
	V.push_back( 0.0f );

	//printf( "REACHABLE:\n" );
	for( int r = 0; r < rows; r++ )
	{
		for( int c = 0; c < columns; c++ )
		{
			for( int q = 0; q < NQ; q++ )
			{
				int qindex = (r*columns*NQ)+(c*NQ)+q;
				reach_count += host_dir_ib_iw[qindex];
			}
			//printf( "%i\t", reach_count );
			reachable.push_back( reach_count );
			float frc = (float)reach_count - 1.0f;
			if( frc > 0 )
			{
				host_probability1.push_back( 0.8f );
				host_probability2.push_back( 0.2f / frc );
			}
			else
			{
				host_probability1.push_back( 1.0f );
				host_probability2.push_back( 0.0f );
			}
			reach_count = 0;
		}
		//printf( "\n" );
	}
	//printf( "\n" );

	for( int i = 0; i < (int)host_dir_ib_iw.size(); i += NQ )
	{
		vector<int> temp;
		vector<int> ib_iw_table;
		vector<int> ib_iw_table_inv;
		for( int j = 0; j < NQ; j++ )
		{
			temp.push_back( host_dir_ib_iw[ i + j ] );
		}
		ib_iw_table.clear();
		for( int m = 0; m < NQ; m++ )
		{
			for( int n = 0; n < NQ; n++ )
			{
				host_dir_ib_iw6.push_back( temp[n] );
				ib_iw_table.push_back( temp[n] );
			}
		}
		ib_iw_table_inv.clear();
		ib_iw_table_inv.resize( ib_iw_table.size() );
		for( int r = 0; r < NQ; r++ )
		{
			for( int c = 0; c < NQ; c++ )
			{
				int true_index = r*NQ+c;
				int inv_index = c*NQ+r;
				ib_iw_table_inv[inv_index] = ib_iw_table[true_index];
			}
		}
		for( int c = 0; c < (int)ib_iw_table_inv.size(); c++ )
		{
			host_dir_ib_iw6_inv.push_back( ib_iw_table_inv[c] );
		}
	}
	//<-FILL_USABLE_VECTORS
}
//
//=======================================================================================
//
void MDPHexagonManager::getRewards( vector<float>& mdp )
{
	mdp.clear();
	for( int r = (int)rows-1; r >= 0; r-- )
	{
		for( int c = 0; c < (int)columns; c++ )
		{
			mdp.push_back( rewards[r*columns+c] );
		}
	}
}
//
//=======================================================================================
//
void MDPHexagonManager::init_structures_on_host( vector<float>& mdp_topology )
{
	convergence			= false;
	iteratingTime		= 0.0f;

	rewards.clear();
	for( int r = (int)rows-1; r >= 0; r-- )
	{
		for( int c = 0; c < (int)columns; c++ )
		{
			rewards.push_back( mdp_topology[r*columns+c] );
		}
	}
	init_mdp();
}
//
//=======================================================================================
//
void MDPHexagonManager::init_perms_on_device( void )
{
	/*
	mmdp_init_permutations_on_device(	rows,
										columns,
										NQ,
										host_dir_ib_iw6,
										host_dir_ib_iw6_inv,
										host_probability1,
										host_probability2,
										host_permutations	);
	*/
	mmdp_init_permutations_on_device2(	rows,
										columns,
										NQ,
										0.8f,
										0.2f,
										host_dir_ib_iw6,
										host_dir_ib_iw6_inv,
										host_permutations	);
}
//
//=======================================================================================
//
void MDPHexagonManager::upload_to_device( void )
{
	mmdp_upload_to_device(	P,
							Q,
							V,
							host_vicinity6,
							host_dir_rw6,
							host_dir_pv6,
							host_permutations );
}
//
//=======================================================================================
//
void MDPHexagonManager::iterate_on_device( void )
{
	iterations = mmdp_iterate_on_device(	discount,
											convergence		);
}
//
//=======================================================================================
//
void MDPHexagonManager::download_to_host( void )
{
	mmdp_download_to_host( P, V );
}
//
//=======================================================================================

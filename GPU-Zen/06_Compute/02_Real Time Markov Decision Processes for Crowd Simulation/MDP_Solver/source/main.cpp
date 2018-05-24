
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


#include <iostream>
#include <vector>
#include <stdlib.h>
#include <string.h>
#include "cMDPSquareManager.h"



//
//=======================================================================================
//
int main (int argc, const char* argv[])
{

	vector<float> rewards;
	vector<float> policy;
	MDPSquareManager *mdpHelper;


	if (argc < 2 )
	{
		std::cout << "\nUsage: MDP_SINGLE_GPU [rewards.csv] table_method \n";
		std::cout << "\n";
		std::cout << "\tFILE_0\tshould be in csv format\n\n";
		std::cout << "\ttable_method\t 0 - old method / 1 - faster method \n\n";
		return 0;
	}
	string filename (argv[1]);
	mdpHelper = new MDPSquareManager();

	// 0 - old buggy method  1 - faster method
	mdpHelper->solve_from_csv ( filename, policy, 1 );
	//mdpHelper->print_mdp ();
	delete mdpHelper;

	std::cout << "argc: " << argc << " argv " << argv[0] << std::endl;
	std::cout << "MDP at Single GPU \n";

	return 0;
}

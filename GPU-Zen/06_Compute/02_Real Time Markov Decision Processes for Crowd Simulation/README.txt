

1. INTRODUCTION
----------------
This demo contains two projects. MDP_Solver directory includes the source code of the parallel MDP solver implementation in CUDA Thrust.
MDP_Crowd directory includes the source code coupling the MDP_Solver and our Crowd Engine. It only contains one character asset and five
different scenarios. Each scenario includes its corresponding optimal poly file stored in a CSV file.

2. DEPENDENCIES
---------------

MDP_Solver

CUDA SDK tested with version 7.0, 7.5 and 8.0

MDP_Crowd 

CUDA SDK tested with version 7.0, 7.5 and 8.0

OpenGL 3.x or higher

Assimp version 2.0
http://www.assimp.org/


GLEW
http://glew.sourceforge.net/

FreeGLUT
http://freeglut.sourceforge.net/

DevIL (IL, ILU and ILUT)
http://openil.sourceforge.net/


3. COMPILATION
---------------

MDP_Solver

MDP_Solver is compatible with Windows and Linux platforms. We recommed to use Visual Studio for Windows compilation; you should need to create
a CUDA Application and include source and header files to the project. Do not forget to add the library dependencies as needed.


For Linux, a makefile is included in the Release directory, which was automatically created by Nvidia Nsight CUDA 8, however it should be 
modified so the source code and dependecies paths corresponds to your system paths. Nvidia Nsight generates the next files inside Release directory
that you should modify accordingly.

makefile		
objects.mk		
./source/subdir.mk	

You can import this project in Nvidia Nsight if your system  has CUDA 8 installed.


MDP_Crowd 

MDP_Crowd is compatible with windows and Linux platforms. We recommed to use Visual Studio for Windows compilation; you should need to create
a CUDA Application and include source and header files to the project. Do not forget to add the library dependencies as needed.


For Linux, a makefile is included in the Release directory, which was automatically created by Nvidia Nsight CUDA 8, however it should be 
modified so the source code and dependecies paths corresponds to your system paths. Nvidia Nsight generates the next files inside Release directory
that you should modify accordingly.

makefile		
objects.mk		
./source/subdir.mk	

You can import this project in Nvidia Nsight if your system  has CUDA 8 installed.


4. USAGE
---------

MDP_Solver

If you want to inspect the performance of the parallel MDP Solver in your system, we have provided five CSV files containing rewards:

ccm_200x200.csv    	- University Campus
maze_100x100.csv	- A maze
o1p2_40x40.csv		- A office floor 
eiffel_100x100.csv	- Tower Eiffel area

After compulation, you should type:

MDP_Solver [reward.csv]

Example in Linux:

$ ./MDP_Solver eiffel_100x100.csv
SQUARE_MDP::SOLVING eiffel_100x100.csv ...
PERM_TABLES_CALC_TIME:  000.104102(s)
Thrust v1.8
Free memory 5946408960, Total memory 6373441536
ITERATING_TIME:     000.601300(s)

0.00306785s MEAN_ITERATION_TIME.
0.049333s INIT_TIME@CPU.
0.732456s INIT_TIME@GPU.
0.684692s ITERATING_TIME.
1.46648s TOTAL_TIME_ELAPSED.
argc: 2 argv ./MDP_Solver
MDP at Single GPU 

As a result, the program generates a file named policy_*.csv storing numbers from 0 to 7 which represents the actions (move foward, backward, left, right, etc.) 
that an agent should follow. 


MDP_Crowd 

i. Changing the scenario

If you had enough spare time to compile our code and seems to be working flawlessly in your system; maybe you should take a dive into the code and switch between
scnerarios following the next steps:


1. Open source/cGlobals.h 
2. Go to Line 126

SCENARIO_TYPE		scenario_type			= ST_EIFFEL;

Change ST_EIFFEL for ST_O1P2, ST_TOWN, ST_CCM values. They are defined in cMacros.h

ii. If you want to try the Maze scenario, do the following:

1. Open the header/cMacros.h
2. Comment line 75 

#define DRAW_SCENARIO

3. Uncomment line 86 

#define DRAW_OBSTACLES

4. Open source/cGlobals.h
5. Go to Line 126

Change scenario_type variable for ST_MAZE

Important. If you want to switch again the scenarios, does not forget to uncomment line 75 from header/cMacros.h, if line 86 from header/cMacros.h remains 
uncommented you'll see a lot of boxes in the scenarios.

iii. Changing the crowd

You can  change the size of the crowd: 

1. Open the file Crowd.xml 
2. Go  to line  70 and modify the width and height variable as desired. We recomend using numbers multiples of 2


iv. Navigation

Now that you have decided to dive, take a look at header/cPeripherals.h, you'll find mouse and keyevents to interact with the engine.











<<==== Per-pixel lists for single pass A-buffer ====>>

This is the source code for our GPUPro article "Per-pixel lists for single pass A-buffer".
We recommend that you first visit http://www.antexel.com/research/gpupro5 for updates.

NOTES:

- The implementations are all located in gpu/asppll/implementations.fp

- Precompiled binaries are available in gpu/bin
  (Compiled with Visual C++ 2010)
  For instance run "benchmark.exe <dll>" for a quick test.
  
- CMake is required to generate project files
  
- The code requires LibSL and its dependencies to be installed:
	http://webloria.loria.fr/~slefebvr/libsl/
  (Visual C++ 2010 binaries are available)

- The code has been written under Windows using Visual C++ 2010.
  Support for other platform/compilers is not tested.     

- Requirements:
  * NVidia GPU with OpenGL 4.2 support.
  * ATI hardware is currently unsupported due to lack of 64 bits atomics in GLSL.
  
- Permission to reuse is granted **provided that the original authors are clearly 
  cited in both the software manual, software splash-screen, whitepapers and 
  publications**.
  
- Software provided as is, without any warranty nor liability of any sort.
  
==== GPU implementation ====

The code focuses on providing a reference implementation of three techniques and their
variants. For each technique/variant a DLL is compiled implementing the A-buffer
functions. This DLL can then be used for benchmarking or rendering. Our code focuses on
benchmarking instead of applications.

A total of 6 DLLs are compiled in gpu/bin/:
- Per-pixel linked lists, with two allocation schemes (naive/paged)
  => postlin-naive.dll (naive: single counter allocation)
  => postlin-paged.dll
- Always sorted linked lists, with two allocation schemes (naive/paged)
  => prelin-naive.dll
  => prelin-paged.dll
  => prelin-cas32-naive.dll
  => prelin-cas32-paged.dll
- Open addressing lists, either sorted with bubble sort as a post process, 
  or always sorted (implementing a [HA-buffer])
  => postopen.dll
  => preopen.dll

All these DLLs share the same code, located in gpu/asppll/. The method and variants
are selected through pre-processor defines, which are set in the CMake script.
  
The executable 'benchmark.exe' implements a benchmarking tool for comparing the
techniques. It runs a number of tests and outputs a SQLite database. The database is
parsed by a Python script to generate the performance figures.

The executable 'seethrough.exe' loads and displays a 3D scene with transparency.
For instance, we used the 'lost_empire' model by Morgan McGuire available at:
http://graphics.cs.williams.edu/data/meshes.xml
Unpack in the same directory as the executable and launch with
seethrough.exe -m lost_empire.obj <dll>
where dll is one of the A-buffer technique, such as prelin-naive.dll
For this particular model we had to flip the textures vertically (lost_empire-RGB[A].png)

--

S. Lefebvre, S. Hornus - INRIA - 2013

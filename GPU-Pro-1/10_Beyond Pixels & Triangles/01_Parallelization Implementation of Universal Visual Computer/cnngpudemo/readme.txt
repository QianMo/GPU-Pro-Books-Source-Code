This package is a demostration program of GPU CNN simulation.  
In the root folder, there are three files and four folders.

//////////////////////////////////////////////////////////////

The files are readme.txt, run.bat and screen.png.

readme.txt is the breif description of this package.

screen.png is a screen captured during the execution of our
demo.

To execute the demo, we can simply double click the batch file
run.bat.

At the beginning of the demo execution, an sample image
"rds.png" and a filter "cnn_reaction_diffusion.txt" will be 
loaded.  To begin the simulation, we can press 'space' key
or use the right-click menu. We can also toggle the simulation 
by pressing 'space' key any time during the simulation.

At the center of the screen, we will see a 3D model of
the SOM codebook which is going to be unfolded guadually
during the training.  At the bottom of screen are the souce 
image, the codebook and the codewords.

Menu commands for loading another image, loading 
another filter and saving the output image can 
be found in the right click menu.

Notice that: the speed of simulation is delibrately 
slowed-down (by using Vsync) so that we can have 
a chance to visualize the simulation processing.

In case only the resulting image is important, we 
can simply replace 

  glutMainLoop(); 

in the main() function with

  iterate(iteration_num);
  keyboard('3',0,0);

By doing so, the simulation will perform at its maximum speed.


//////////////////////////////////////////////////////////////

The folders are bin, src, lib and data.

"bin" is the executable binary files.

"lib" is the library folder containing all the necessary library
to compile our demo project, including cgsdk, glew, glut, JpegLib
and zlib.

"data" is the image folder containing some sample images.

src/g_common - some utilities routines not directly related to
this demo.

src/cnngpu_demo - the visual studio project contains all the source
code.

  cnn_utility.cpp and cnn_utility.h
    - the actual routines to perform CNN SOM simulation.

  shader.cg, shader.cpp, shader.h
    - the Cg script of CNN SOM simulation and the necessary 
      routines to expose its functionalities.

  cnngpu_demo.cpp - the glut window program.


This package is a demostration program of GPU SOM training
for color quantization.  In the root folder, there are 
three files and fhree folders.

//////////////////////////////////////////////////////////////

The files are readme.txt, run.bat and screen.png.

readme.txt is the breif description of this package.

screen.png is a screen captured during the execution of our
demo.

To execute the demo, we can simply double click the batch file
run.bat.

At the beginning of the demo execution, an sample image
"05112500.JPG" will be loaded and will start the training
of a SOM codebook.

We can turn toggle the training by pressing 'space' key
any time during the training.

At the center of the screen, we will see a 3D model of
the SOM codebook which is going to be unfolded guadually
during the training.  At the bottom of screen are the souce 
image, the codebook and the codewords.

Menu commands for generating a different initial codebook,
saving the codebook and saving the codewords, and loading
another image can be found in the right click menu.

The codebook file and the codewords file are in 
protable-floating-map (.pfm) format.  We are free to
further quantize the codewords file at different bitrate
depends on specific application.


//////////////////////////////////////////////////////////////

The folders are src, lib and data.

"lib" is the library folder containing all the necessary library
to compile our demo project, including cgsdk, glew, glut, JpegLib
and zlib.

"data" is the image folder containing some sample images.

src/g_common - some utilities routines not directly related to
this demo.

src/somgpu_demo - the visual studio project contains all the source
code.

  gpu_som.cpp and gpu_som.h 
    - the actual class object to perform GPU SOM training.

  shader_som.cg, shader_som.cpp, shader_som.h
    - the Cg script of GPU SOM training and the necessary 
      routines to expose its functionalities.

  shader_codeword.cg, shader_codeword.cpp, shader_codeword.h
    - the Cg script to convert a source image into
      floating-point precision indices map and the necessary 
      routines to expose its functionalities.

  shader_lighting.cg, shader_lighting.cpp, shader_lighting.h
    - the Cg script of simple fragment lighting and the necessary 
      routines to expose its functionalities.

  somgpu_merge.cpp - the glut window program.


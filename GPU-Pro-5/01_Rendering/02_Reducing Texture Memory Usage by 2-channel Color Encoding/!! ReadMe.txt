
This directory contains source code and projects related to
2-channel texture compression technique presented by
Krzysztof Klcuzek in GPUPro 5 book.



The package contains two MSVC projects and following utilities:




1. Tex2 - the command-line compressor/decompressor utility.


	Usage:
	        tex2 -encode <input>.bmp [options] <output>.bmp <out_params>.txt
	        tex2 -decode <input>.bmp <in_params>.txt <output>.bmp
	        tex2 -colorprobe <input>.bmp <output>.bmp
	        tex2 -fulldemo <input>.bmp <original_and_transcoded_comparison>.bmp

	Options for encoder:
	        -weights <wR> <wG> <wB>         Specify channel importance
	                                          (relative integer values 1..1000)
	                                          defaults to 1/1/1

        	-colorprobe                     Stamp color probe in image corner



	Please note that the Tex2 tool currently supports only standard BMP files.

	The Tex2 project requires Allegro library (versions 4.x) to be set up.
	You can find precompiled binaries at:

		https://www.allegro.cc/files/?v=4.2



2. DxDemo - a small demo presenting decompression scheme in HLSL.


	The demo shows rotating sphere and a list of available textures to choose from.
	The directory containing compressed textures should be passed via command line.

	The demo searches for *.txt files in given directory, containing compression
	parameters as outputted by Tex2 tool. Each *.txt file should have accompanying
	encoded image file (in any format supported by D3DX).


	The demo requires DirectX SDK to be compiled.



3. "test" directory - a frmework for batch encoding and experiments.


	Upload your own images to this directory (after converting them to BMP first).
	Running "!encode_all_textures.bat" will encode all *.BMP files present
	producing for each one:
		
		*_encoded.bmp		- encoded 2-channel R/G image
		*_encoded.txt		- encoding parameters


	Then, you can run "!run_dxdemo_presentation.bat" to view decompressed textures
	mapped onto a sphere.


	To get you started, some Public Domain textures by Nobiax (nobiax.deviantart.com)
	are provided as an example. Note that the RGB weight set 1/1/1 is used here,
	as original weight set proposed in the book did not work for some of the textures
	from this set (the lava texture in particular).



4. "framework" directory


	This directory contains framework libraries written by the author
	to hide most common tasks related to DirectX and general tools.

	For easy integration with sample projects, these libraries were
	compacted into single source/header pairs and just added to the project.

	As most private frameworks, there is no in-depth documentation for these
	libraries, but this knowledge is not required for understanding of
	2-channel texture encoding.



Thanks,
Krzysztof Kluczek

DevKK.net

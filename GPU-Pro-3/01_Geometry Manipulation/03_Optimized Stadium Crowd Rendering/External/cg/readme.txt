NVIDIA Cg 2.2 April 2009 README  Copyright (C) 2002-2009 NVIDIA Corp.
===============================

This distribution contains
--------------------------

- NVIDIA Cg toolkit documentation
  in the docs directory
     
- NVIDIA Cg compiler (cgc)
  in the bin directory

- NVIDIA Cg runtime libraries
  in the lib directory

- Example Cg applications
  in the examples directory

- Under Microsoft Windows, a Cg language syntax highlighter
  for Microsoft Visual Studio is provided in the
  msdev_syntax_highlighting directory

- Under Microsoft Windows, if selected at install time, 64-bit
  binaries and libraries are in the bin.x64 and lib.x64 directories.

See the release notes (docs/CgReleaseNotes.pdf) for detailed
information about this release.

The Cg toolkit is available for a number of different hardware and
OS platforms.  As of this writing, supported platforms include:

  - Microsoft NT 4, 2000, and Windows XP & Vista on IA32/x86/x86-64 (Intel, AMD)
  - Linux on IA32/x86 (Intel, AMD)
  - Linux for x64 (AMD64 and EMT64)
  - MacOS X 10.4 and 10.5 (Tiger and Leopard)
  - Solaris (x86/x86_64)

Visit the NVIDIA Cg website at http://developer.nvidia.com/page/cg_main.html
for updates and complete compatibility information.

Changes since Cg 2.2 beta February 2009
---------------------------------------
- New features
  - Support for pack_matrix() pragma
  - Arrays of shaders can now be used in CgFX files
  - Support for 64-bit Solaris
  - Bug fixes (see release notes for details)

Changes since Cg 2.1 November 2008
----------------------------------
- New features
  - DirectX10 and GLSL geometry profiles (gs_4_0 AND glslg)
  - Support for "latest" profile keyword in CgFX compile statements
  - Additional API routines (see release notes for a complete list)
  - Migrated the OpenGL examples onto GLEW
- New examples
  - Direct3D10/advanced/combine_programs
  - Direct3D10/advanced/gs_shrinky
  - Direct3D10/advanced/gs_simple
  - OpenGL/advanced/cgfx_latest
  - Tools/cgfxcat
  - Tools/cginfo
- New documentation
  - Updated reference manual for new profiles and entry points

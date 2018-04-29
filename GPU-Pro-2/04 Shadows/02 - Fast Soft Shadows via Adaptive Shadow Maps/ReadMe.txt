Requires shader model 3.0.

The demo uses depth stencil textures, which are not supported on Radeons 1x00.
It also uses D3DFMT_D16 shadow maps and D3DFMT_R16F textures, which are not 
supported on GeForces 6x00/7x00.

There are minor filtering artifacts of unclear origin on GeForce cards.
It works flawlessly on Radeons :)

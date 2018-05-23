*************************
*  Introduction:
*************************

Sample code for a 'Quaternions revisited' article from GPU Pro 5

Article 'Quaternions Revisited' for GPU Pro 5
Authors: Peter Sikachev, Sergey Makeev, Vladimir Egorov

Sample code accompanying the article
Author: Sergey Makeev


*************************
*  License:
*************************

  Copyright (c) 2013, Sergey Makeev

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software.

  2. If you use this software in a non-commercial product, an acknowledgment
     in the product documentation would be appreciated but is not required.

  3. If you use this software in a commercial product you requried to credit
     the author.

  4. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.

  5. This notice may not be removed or altered from any source distribution.
    

   Please let me know if you use this code in your products or have any questions or suggestions.

   e-mail: sergeymakeev@inbox.ru
   http://linkedin.com/in/sergeymakeev/


Graphical assets Copyright (C) 2013, Mail.Ru Games, LLC.
All graphical assets provided for educational purposes only and can not be used in commercial products.

Parts of the code are based on examples from the DirectX SDK June 2010
Copyright (c) Microsoft Corporation. All rights reserved.


*************************
*  Information:
*************************
This sample illustrates the article Quaternions revisited.
Is a simple but at the same time, the complete pipeline for graphic resources based on quaternions.

The following pipeline parts are shown in sample
- Geometry importer from Autodesk FBX, calculation of aligned quaternions from TBN.
- Animation importer from Autodesk FBX, calculation of aligned quaternions for animation data

Also in the example is implemented runtime part for rendering imported data.
- Animation player that stores bones data in the texture for easy use instancing for animated geometry.
- Geometry render that use geometry instancing. Rendering support animations and normal mapping using quaternions.


*************************
*  Requires:
*************************
- Microsoft Visual Studio 2010
- Microsoft DirectX SDK (executable compiled with DirectX SDK June 2010)
- Autodesk FBX SDK (executable compiled with FBX SDK 2014.1)


*************************
*  Building:
*************************
You should set up next environment variables:

DXSDK_DIR (Usually set automatically by DX SDK installer)
Example: C:\Program Files (x86)\Microsoft DirectX SDK (June 2010)\

FBX_SDK (You need to set this environment variable manually)
Example: C:\Program Files\Autodesk\FBX\FBX SDK\2014.1\

When the required environment variables are set, you can open QuaternionsRevisited.sln and build solution.


*************************
*  Acknowledgements:
*************************
Special thanks for Konstantin Antipov for helping with assets for the sample.
All graphical assets courtesy of Mail.Ru Group.


Enjoy!
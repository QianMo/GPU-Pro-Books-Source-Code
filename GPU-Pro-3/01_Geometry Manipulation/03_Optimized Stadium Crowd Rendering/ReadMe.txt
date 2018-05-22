GPU Pro 3
Optimized Stadium Crowd Rendering
Alan Chambers

INFO
This sample shows the basic setup for crowd rendering in a stadium environment. It sets up the instancing and mirrored rendering using OpenGL and Cg on PC and OSX. At the time of writing, ATI / AMD were not supporting the gp4vp and gp4fp Cg shader profiles needed to run this demo. I am currently working on a GLSL version of the demo which will allow it to run on both nVidia and ATI / AMD graphics cards. You can download the latest version from my website http://www.kickblade.com/downloads.

SETUP
Working directory needs to be set to the Data directory. XCode has a weird pathing setup, which I've tried to account for. If the data does not load you won't see any crowd dudes so make sure this is set properly.

REQUIRES
OpenGL 3.2
GL_ARB_draw_instanced extension
gp4 shader capable GFX card

INCLUDES
CG 2.2
GLEW 1.7
GLUT 3.7

NOTES
There is no animation or offscreen rendering included in the sample.
A full animation system and model renderer is required for this so I've left it so the code can easily plug into your existing system.
For this reason, the atlas does not update. This means some camera view angles will not look correct.
The crowd billboard rendering is not deferred.
There is a known problem with trying to instance more than around 15,000 quads in one draw call.
The camera math is a bit rubbish, you can get weird results quite easily, but it works for the context of a demo.
The atlas does not include mipmaps, in a real world scenario we also need to use these.
Don't forget to check out my website at http://www.kickblade.com for updates and other cool downloads.

KEYS
w - move forward
s - move back
a - move left
d - move right
j - turn left
l - turn right
i - turn up
k - turn down
y - increase density
h - decrease density

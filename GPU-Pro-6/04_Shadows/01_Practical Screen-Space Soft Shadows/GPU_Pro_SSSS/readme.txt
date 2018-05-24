Practical Screen Space Soft Shadows
by
Marton Tamas (twitter: @0martint)
and
Viktor Heisenberger (twitter: @TheVico87)

Things I figured out after completing the article:
-rotating the Gauss blurring direction by about 26.6 degrees solves banding artifacts on
 straight up walls (vertical samples are no longer wasted)
-add the usual bias to the blocker search to overcome "vertical lines" artifacts at PCSS
-fixed the Gauss blur weights, so now it should look better

Building:
Download cmake (win32 installer): http://www.cmake.org/download/
Install cmake (default settings are ok)
Open cmake GUI
set "Where is the source code:" with "browse source" to the folder where the source code is
set "Where to build the binaries:" with "browse build" to the "build" folder (relative to where the source code is)
hit "configure" at the bottom
hit "generate"
open the visual studio .sln file in the "build" folder
select "Release" as the target solution configuration
press "F7" on the keyboard to build the solution
run the exe in the "Release" folder (relative to where the source code is)

Controls:
Move by right mouse click + WASD
You can adjust the light size and radius
You can adjust light color
You can go to 9 preselected locations
You can reload all the shaders
You can switch between the techniques
You can select a light by left clicking on it
You can move a light by pressing T and moving the cursor
You can add a light by pressing the + button
You can remove a light by pressing the - button

WIP (in order of importance):
[done] select light
[done] move light
[done] add/remove lights
handle transparent layers properly (currently only works hacked/hardcoded)
handle any number of layers
add spot lights (culling doesn't work at the moment)
rotate light if spotlight
change spot exponent, spot cutoff
change light type
add lighting for translucent objects (need to save per workgroup light culling data)
change attenuation type, attenuation coefficient
change fullscreen

Broken ATM:
make sure it works on nvidia --> driver issue?
exponential shadows don't work with SSSS --> driver issue?
changing display resolution breaks lighting for some reason --> driver issue?
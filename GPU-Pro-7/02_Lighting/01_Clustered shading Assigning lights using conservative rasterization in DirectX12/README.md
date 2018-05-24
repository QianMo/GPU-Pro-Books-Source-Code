# Clustered Shading: Assigning Lights Using Conservative Rasterization in DirectX 12
DirectX 12 light culling technique featured in GPU Pro 7


Platform: Windows 10 x64 on Graphics cards with D3D12 Conservative Rasterization feature (Currently Nvidia Maxwell 2.0 or newer and Intel Skylake GPUs or newer)


## Build

1. Run "generate_VS_solution.bat" and a /build folder will contain the Visual Studio 2015 solution.

2. Open solution and go to project properties to change "Target Platform Version" from 8.1 to 10.0.10240.0 (or 10.0.10586.0 or relevant Windows 10 SDK version)


To build Debug you must install DirectX Graphics Tools from Windows 10 Feature-On-Demand (in Settings).


Constants.h contains some tweakable variables.


## Controls and Shortcuts
### Keyboard & Mouse

Hold LMB + move mouse to look around

W, A, S, D  - Move forward, backward & strafe

LSHIFT      - Move down

SPACE       - Move up

C           - Print number of light indices in the Light Linked List

V           - Take a snap shot of one light and show all the clusters that light is assigned to

F           - Take a snap shot of current view and display all the clusters where lights were fetched from that frame

ESC         - Exit program


### Gamepad
![Mapping](https://raw.githubusercontent.com/kevinortegren/kevinortegren.github.io/master/images/ConservativeClusteredShading/controller_mapping.png)


All shortcuts are accessible through the UI.

![Settings](https://raw.githubusercontent.com/kevinortegren/kevinortegren.github.io/master/images/ConservativeClusteredShading/Settings.png)

## UI

The UI has the following functionality:

**Lights**

- Toggle point/spot lights on or off.
- Toggle light movement.
- Set number of lights to display.

**Light shapes**

- Toggle display of light shapes.
- Select which LOD to use for the light shapes.
- Select the number of light shapes to be rendered.

**Clustering**

- Select the depth distribution mode.
- Select precise or approximated clustering(Use approx. pipe). 
- Show density map with radar colors that shows the number of lights processed per pixel. See legend for color explanation.

![Legend](https://raw.githubusercontent.com/kevinortegren/kevinortegren.github.io/master/images/ConservativeClusteredShading/legend.png)

- Print number of indices in the Linked Light List.
- Take a snapshot of all the clusters that were accessed for lighting the frame. Display the clusters in world space to look at.

**Snap shot**

- Select the light type and the light index to take a snap shot of. 
- Hit "Shape cluster snap" to show all the clusters that the specified light was assigned to during the snap shot frame.

**Benchmark**

- Displays timings in milliseconds (apart from FPS) for different passes on the GPU.
- Total LA == The sum of alla shell pass times and fill pass times
- Total GPU time == The sum of Total LA and Shading time

**Note:**

The UI uses a modified version of AntTweakBar to work with DX12 and does exhibit some minor issues. Numpad keys don't work as input. Input fields are not cleared immediately on new input. Mouse events are forwarded to the camera movement through the UI...

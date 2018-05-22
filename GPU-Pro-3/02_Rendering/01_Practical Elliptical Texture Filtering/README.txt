Practical Elliptical Texture Filtering on GPU - Demo
----------------------------------------------------

(Press 'h' inside the demo for help)

Increasing the filter width, filter sharpness, texels/pixel and the maximum anisotropy will give you better filtering quality at the expense of speed.

The demo has two stress test scenes for texture filtering: infinite tunnel and infinite planes.
The infinite planes scene is more demanding in terms of filtering than the tunnel one, so the default filter width and sharpness are adjusted accordingly when switching scenes. As a consequence the performance in the infinite planes scene is lower than the performance in the tunnel scene. 

The temporal filtering mode requires v-sync enabled. 
Use the 'v' key to toggle  v-sync inside the app (the driver should not override the application settings)

For any inquiries, please contact me at pmavridis@gmail.com

Pavlos Mavridis

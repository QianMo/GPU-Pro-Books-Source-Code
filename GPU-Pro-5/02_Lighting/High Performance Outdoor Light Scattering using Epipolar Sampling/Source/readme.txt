Intel Corporation - Outdoor Light Scattering Sample

This sample shows how high-quality light scattering effects in large outdoor environments can be rendered on Intel HD graphics in real time. The technique extends the approach shown in the previously published sample and exploits the same main ideas. Epipolar sampling helps minimize the number of samples for which expensive ray marching algorithm is executed while 1D min/max binary trees are used to accelerate computations.

In this sample, a more complex physical model is exploited to simulate light propagation in the atmosphere. It assumes that the planet is spherical and that the particle density decreases exponentially with the altitude computed with respect to the planet surface. To facilitate shadows in large outdoor environments, cascaded shadow maps are used. Scattering contribution from each cascade is computed and accumulated to obtain the final result.

Note: “Media Elements” are the images, clip art, animations, sounds, music, shapes, video clips, 2D Images, 2D and 3D Mesh’s and mesh data, animation and animation data, and Textures included in the software. This license does not grant you any rights in the Media Elements and you may not reproduce, prepare derivative works, distribute, publicly display, or publicly perform the Media Elements.

Note: The source code sample is provided under the BSD license.  See the license folder within the sample source directory for additional details.
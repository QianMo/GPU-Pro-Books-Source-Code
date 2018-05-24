/*

Copyright 2013,2014 Sergio Ruiz, Benjamin Hernandez

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>

In case you, or any of your employees or students, publish any article or
other material resulting from the use of this  software, that publication
must cite the following references:

Sergio Ruiz, Benjamin Hernandez, Adriana Alvarado, and Isaac Rudomin. 2013.
Reducing Memory Requirements for Diverse Animated Crowds. In Proceedings of
Motion on Games (MIG '13). ACM, New York, NY, USA, , Article 55 , 10 pages.
DOI: http://dx.doi.org/10.1145/2522628.2522901

Sergio Ruiz and Benjamin Hernandez. 2015. A Parallel Solver for Markov Decision Process
in Crowd Simulations. Fourteenth Mexican International Conference on Artificial
Intelligence (MICAI), Cuernavaca, 2015, pp. 107-116.
DOI: 10.1109/MICAI.2015.23

*/

enum { SX, SY, SZ, SW };
enum { SA, SB, SC, SD };

float floorVertices[4][3] = {
	//{-PLANE_SCALE, 0, -PLANE_SCALE},
	//{-PLANE_SCALE, 0,  PLANE_SCALE},
	//{ PLANE_SCALE, 0,  PLANE_SCALE},
	//{ PLANE_SCALE, 0, -PLANE_SCALE},
	{-256.0f, 0, -256.0f},
	{-256.0f, 0,  256.0f},
	{ 256.0f, 0,  256.0f},
	{ 256.0f, 0, -256.0f},
};

float plane[4];
float shadowMat[4][4];
float shadow_mat[16];

// Find the plane equation given 3 points.
void findPlane( float v0[3], float v1[3], float v2[3] )
{
      GLfloat vec0[3], vec1[3];

      // Need 2 vectors to find cross product.
      vec0[SX] = v1[SX] - v0[SX];
      vec0[SY] = v1[SY] - v0[SY];
      vec0[SZ] = v1[SZ] - v0[SZ];

      vec1[SX] = v2[SX] - v0[SX];
      vec1[SY] = v2[SY] - v0[SY];
      vec1[SZ] = v2[SZ] - v0[SZ];

      // find cross product to get A, B, and C of plane equation
      plane[SA] = vec0[SY] * vec1[SZ] - vec0[SZ] * vec1[SY];
      plane[SB] = -(vec0[SX] * vec1[SZ] - vec0[SZ] * vec1[SX]);
      plane[SC] = vec0[SX] * vec1[SY] - vec0[SY] * vec1[SX];
      plane[SD] = -(plane[SA] * v0[SX] + plane[SB] * v0[SY] + plane[SC] * v0[SZ]);
}


// Create a matrix that will project the desired shadow.
void shadowMatrix( float lightpos[4] )
{
    GLfloat dot;
    findPlane(floorVertices[1], floorVertices[2], floorVertices[3]);

      // Find dot product between light position vector and ground plane normal.
      dot = plane[SX] * lightpos[SX] +
        plane[SY] * lightpos[SY] +
        plane[SZ] * lightpos[SZ] +
        plane[SW] * lightpos[SW];

      shadowMat[0][0] = dot - lightpos[SX] * plane[SX];
      shadowMat[1][0] = 0.f - lightpos[SX] * plane[SY];
      shadowMat[2][0] = 0.f - lightpos[SX] * plane[SZ];
      shadowMat[3][0] = 0.f - lightpos[SX] * plane[SW];

      shadowMat[SX][1] = 0.f - lightpos[SY] * plane[SX];
      shadowMat[1][1] = dot - lightpos[SY] * plane[SY];
      shadowMat[2][1] = 0.f - lightpos[SY] * plane[SZ];
      shadowMat[3][1] = 0.f - lightpos[SY] * plane[SW];

      shadowMat[SX][2] = 0.f - lightpos[SZ] * plane[SX];
      shadowMat[1][2] = 0.f - lightpos[SZ] * plane[SY];
      shadowMat[2][2] = dot - lightpos[SZ] * plane[SZ];
      shadowMat[3][2] = 0.f - lightpos[SZ] * plane[SW];

      shadowMat[SX][3] = 0.f - lightpos[SW] * plane[SX];
      shadowMat[1][3] = 0.f - lightpos[SW] * plane[SY];
      shadowMat[2][3] = 0.f - lightpos[SW] * plane[SZ];
      shadowMat[3][3] = dot - lightpos[SW] * plane[SW];

      shadow_mat[0] = dot - lightpos[SX] * plane[SX];
      shadow_mat[4] = 0.f - lightpos[SX] * plane[SY];
      shadow_mat[8] = 0.f - lightpos[SX] * plane[SZ];
      shadow_mat[12] = 0.f - lightpos[SX] * plane[SW];

      shadow_mat[1] = 0.f - lightpos[SY] * plane[SX];
      shadow_mat[5] = dot - lightpos[SY] * plane[SY];
      shadow_mat[9] = 0.f - lightpos[SY] * plane[SZ];
      shadow_mat[13] = 0.f - lightpos[SY] * plane[SW];

      shadow_mat[2] = 0.f - lightpos[SZ] * plane[SX];
      shadow_mat[6] = 0.f - lightpos[SZ] * plane[SY];
      shadow_mat[10] = dot - lightpos[SZ] * plane[SZ];
      shadow_mat[14] = 0.f - lightpos[SZ] * plane[SW];

      shadow_mat[3] = 0.f - lightpos[SW] * plane[SX];
      shadow_mat[7] = 0.f - lightpos[SW] * plane[SY];
      shadow_mat[11] = 0.f - lightpos[SW] * plane[SZ];
      shadow_mat[15] = dot - lightpos[SW] * plane[SW];
}

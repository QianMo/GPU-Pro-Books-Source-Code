
//#include <optix_world.h>
#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_matrix_namespace.h>
#include <optixu/optixu_aabb_namespace.h>

using namespace optix;

// Start index and vertex offset in an index and vertex buffer, storing multiple objects
rtDeclareVariable(int, start_index, , ) = 0;
rtDeclareVariable(int, vertex_offset, , ) = 0;

// Buffer data
rtBuffer<float3> vertex_buffer;     
rtBuffer<float3> normal_buffer;
rtBuffer<float2> texcoord_buffer;
rtBuffer<int3>   index_buffer;

// Output attributes
rtDeclareVariable(float2, texcoord, attribute texcoord, ); 
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, ); 
rtDeclareVariable(float3, shading_normal, attribute shading_normal, ); 

// The current ray (input)
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );

// Test the current ray against a triangle
RT_PROGRAM void intersect( int primIdx )
{	
	// initialize
	geometric_normal = make_float3(0,1,0);
	shading_normal = make_float3(0,1,0);
	texcoord = make_float2(0,0);

	// read vertex indices
	int3 v_idx = index_buffer[start_index + primIdx] + make_int3(vertex_offset, vertex_offset, vertex_offset);

	// read the vertex positions
	float3 p0 = vertex_buffer[ v_idx.x ];
	float3 p1 = vertex_buffer[ v_idx.y ];
	float3 p2 = vertex_buffer[ v_idx.z ];

	// Intersect ray with triangle
	float3 n;
	float  t, beta, gamma;
	if( intersect_triangle( ray, p0, p1, p2, n, t, beta, gamma ) ) 
	{
		if(  rtPotentialIntersection( t ) ) 
		{
			// Calculate normals and tex coords 
			geometric_normal = -normalize( n );	// different winding order! -> flip the normal (then shading and geometric normal will approximately coincide)
			float3 n0 = normal_buffer[ v_idx.x ];
			float3 n1 = normal_buffer[ v_idx.y ];
			float3 n2 = normal_buffer[ v_idx.z ];
			shading_normal = normalize( n1*beta + n2*gamma + n0*(1.0f-beta-gamma) );
		
			float2 t0 = texcoord_buffer[ v_idx.x ];
			float2 t1 = texcoord_buffer[ v_idx.y ];
			float2 t2 = texcoord_buffer[ v_idx.z ];
			texcoord = t1*beta + t2*gamma + t0*(1.0f-beta-gamma);

			rtReportIntersection(0);
		}
	}
}

// computes the bounding box for a triangle
RT_PROGRAM void bounds (int primIdx, float result[6])
{
	// read index
	const int3 v_idx = index_buffer[start_index + primIdx];
  
	// read vertices
	const float3 v0 = vertex_buffer[ v_idx.x + vertex_offset ];
	const float3 v1 = vertex_buffer[ v_idx.y + vertex_offset];
	const float3 v2 = vertex_buffer[ v_idx.z + vertex_offset];
  
	// compute area of triangle
	const float  area = length(cross(v1-v0, v2-v0));

	// create an aabb
	optix::Aabb* aabb = (optix::Aabb*)result;

	// check if the area is invalid and then set the bounds
	if(area > 0.0f && !isinf(area)) {
		aabb->m_min = fminf( fminf( v0, v1), v2 );
		aabb->m_max = fmaxf( fmaxf( v0, v1), v2 );
	} else {
		aabb->invalidate();
	} 
}


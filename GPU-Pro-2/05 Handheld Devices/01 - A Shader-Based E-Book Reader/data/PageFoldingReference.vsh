
attribute highp   vec3  inVertex;
attribute mediump vec2  inTexCoord;

// Algorithm input
uniform highp float Fold;
uniform highp float Radius;

// Precalculated values
uniform highp vec2 Direction;
uniform highp vec2 Corner;
uniform highp vec2 Tangent;
uniform highp vec2 Point;


varying mediump vec2   TexCoord;

const highp float PI = 3.141592;
const highp float INV_PI_2 = 2.0 / PI;


void main()
{
	// Transform vertex, accounting for scale and aspect ratio
	highp vec2 vertex = inVertex.xy;
	highp vec2 v = vertex - Point;
	
	// If distance is positive, then the point is on the left of the vector.
	// If distance is negative, then the point is on the right of the vector.
	//highp float distance = cross(vec3(Tangent, 0.0), vec3(v, 0.0)).z;
	highp float distance = -dot(v, Direction);
	
	if (distance > 0.0)
	{
		highp vec2 proj = Point + dot(v, Tangent) * Tangent;

		highp float angle = distance / Radius;
		// Position calculation
		if (angle < PI)
		{
			gl_Position.xy = proj - Radius * sin(angle) * Direction;
			gl_Position.z = -INV_PI_2 * distance;			
		}
		else
		{
			gl_Position.xy = proj + (distance - PI * Radius) * Direction;
			gl_Position.z = -2.0 * Radius;			
		}
	}
	else
	{
		gl_Position.xyz = vec3(vertex, 0.0);
	}
	gl_Position.w = 1.0;

	TexCoord = inTexCoord;
}
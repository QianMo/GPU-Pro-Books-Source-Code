// All Defines:
//#define ENABLE_FOLDING
//#define HAS_TEXTURE
//#define EDGE_GRADIENT
//#define EDGE_GRADIENT_INT
//#define EDGE_GRADIENT_EXT
//#define FOLD_GRADIENT


attribute highp   vec3  inVertex;
#ifdef HAS_TEXTURE
attribute mediump vec2  inTexCoord;
#endif

// Transformations required bt left and right page positioning
uniform highp vec2 InputScale;
uniform highp vec2 InputTranslation;

#ifdef ENABLE_FOLDING
// Algorithm input
uniform highp float Radius;

// Precalculated values
uniform highp vec2 Direction;
uniform highp vec2 Tangent;
uniform highp vec2 Point;

uniform mediump float AspectRatio;
#endif

uniform bool		Rotate;

uniform mediump float ZOffset;
#ifdef HAS_TEXTURE
uniform mediump int   FlipTexture;
#endif

varying mediump vec2   TexCoord;
#ifdef EDGE_GRADIENT
varying lowp float EdgeGradient;
#endif

#ifdef FOLD_GRADIENT
varying lowp float FoldGradient;
#endif

const highp float PI = 3.141592;
const highp float INV_PI_2 = 2.0 / PI;
#ifdef EDGE_GRADIENT
#ifdef EDGE_GRADIENT_INT
const highp float EDGE = 0.01;
#elif defined EDGE_GRADIENT_EXT
const highp float EDGE = 0.03;
#endif
const highp float ONE_OVER_EDGE = 1.0 / EDGE;
#endif
#ifdef FOLD_GRADIENT
const lowp float PI_2 = 1.570796;
const highp float CURL_THRESHOLD = 0.25 * PI;
const highp float ONE_OVER_CURL_THRESHOLD = 1.0 / CURL_THRESHOLD;
#endif

#ifdef EDGE_GRADIENT
highp float MixColor(vec2 vertex)
{
	float absy = abs(vertex.y);
	float absx = abs(vertex.x);
#ifdef EDGE_GRADIENT_INT
	if (absy > 1.0 - EDGE || absx > 1.0 - EDGE)
	{
		return 0.5 * (0.5 + (max(absy, absx) - (1.0 - EDGE)) * ONE_OVER_EDGE * 0.5);
	}
	return 0.0;
#endif
#ifdef EDGE_GRADIENT_EXT
	if (absy > 1.0 || absx > 1.0)
	{
		return 1.0 - (max(absy, absx) - 1.0) * ONE_OVER_EDGE;
	}
	return 1.0;
#endif
}
#endif

#ifdef FOLD_GRADIENT
highp float CalculateFoldGradient(highp float angle)
{
	if (angle > 0.5 * (PI + CURL_THRESHOLD) && angle < PI_2 + CURL_THRESHOLD)
	{
		return ((PI_2 + CURL_THRESHOLD) - angle) * ONE_OVER_CURL_THRESHOLD;
	}
	if (angle > PI_2 && angle < 0.5 * (PI + CURL_THRESHOLD))
	{
		return (angle - PI_2) * ONE_OVER_CURL_THRESHOLD;
	}
	return 0.0;
}
#endif

void RotatePos()
{
	if (Rotate)
	{
		highp float x = gl_Position.x;
		gl_Position.x = -gl_Position.y;
		gl_Position.y = x;
	}
}

void main()
{
#ifdef EDGE_GRADIENT
	EdgeGradient = 0.5 * MixColor(inVertex.xy);
#endif

	// Transform vertex, accounting for scale and aspect ratio
	highp vec2 vertex = inVertex.xy;

	vertex *= InputScale;
#ifdef ENABLE_FOLDING
	vertex.x *= AspectRatio;
	
	highp vec2 v = vertex - Point;
	
	//highp float distance = cross(vec3(Tangent, 0.0), vec3(v, 0.0)).z;
	highp float distance = -dot(v, Direction);	

	if (distance > 0.0)
	{
		highp vec2 proj = Point.xy + dot(v, Tangent) * Tangent;

		highp float angle = distance / Radius;
		// Position calculation
		if (angle < PI)
		{
			gl_Position.xy = proj - Radius * sin(angle) * Direction;
			gl_Position.z = -INV_PI_2 * distance;
#ifdef FOLD_GRADIENT			
			FoldGradient = CalculateFoldGradient(angle);
#endif
		}
		else
		{
			gl_Position.xy = proj + (distance - PI * Radius) * Direction;
			gl_Position.z = -2.0 * Radius;
#ifdef FOLD_GRADIENT			
			FoldGradient = 0.0;
#endif
		}
	}
	else
	{
		gl_Position.xyz = vec3(vertex, 0.0);
#ifdef FOLD_GRADIENT	
		FoldGradient = 0.0;
#endif
	}
	// Remap x coordinate to viewport
	gl_Position.x /= AspectRatio;
#else
	gl_Position.xyz = vec3(vertex, 0.0);
#endif // ENABLE_FOLDING
	// Add translation component
	gl_Position.xy += InputTranslation;
	// Add Z offset
	gl_Position.z += ZOffset;
	// Set scale value
	gl_Position.w = 1.0;
	
	RotatePos();


#ifdef HAS_TEXTURE
	TexCoord = inTexCoord;
	if (FlipTexture == 1)
		TexCoord.x = 1.0 - TexCoord.x;
#endif
}
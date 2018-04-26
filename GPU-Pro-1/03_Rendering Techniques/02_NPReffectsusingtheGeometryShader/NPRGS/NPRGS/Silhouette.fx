/**
 *	Pedro Hermosilla
 *
 *	Moving Group - UPC
 *	Silhouette.fx
 */
 
#define PI 3.1416f

//Constant buffer.
cbuffer constantBuffer
{       
	float4x4 WorldView;
	float4x4 projMatrix;
	float edgeSize;
	float4 aabbPos;
	float lengthPer;
	float scale;
};

Texture2D texureDiff;

SamplerState samLinear
{
    Filter = MIN_MAG_MIP_POINT;
    AddressU = Wrap;
    AddressV = Wrap;
};

BlendState NoBlend
{
    BlendEnable[0] = FALSE;
};

DepthStencilState EnableDepth
{
    DepthEnable = TRUE;
    DepthWriteMask = ALL;
    DepthFunc = LESS_EQUAL;
};

struct VERTEXin
{
	float4 Pos : POSITION;
	float3 Normal : NORMAL; 
};

struct VERTEXout
{
	float4 Pos : POSITION;
	float3 Normal : NORMAL;
};

struct GEOMETRYin
{
	float4 Pos : POSITION;
	float3 Normal : NORMAL;
};

struct GEOMETRYout
{
	float4 Pos : SV_Position;
	float3 Uv : TEXCOORD0;
};

struct PIXELin
{
	float4 Pos : SV_POSITION;
	float3 UV : TEXCOORD0;
};

//Method for get the normal of a triangle.
float3 getNormal( float3 A, float3 B, float3 C )
{
	float3 AB = B - A;
	float3 AC = C - A;
	return normalize( cross(AB,AC) );
};

////////////////////VERTEX SHADER/////////////////////////


VERTEXout silVertex(VERTEXin inPut)
{
	VERTEXout outPut;
    outPut.Pos = mul(inPut.Pos,WorldView);
    outPut.Normal = mul(float4(inPut.Normal,0.0f),WorldView).xyz;
	return outPut;
}

////////////////////GEOMETRY SHADER/////////////////////////

[maxvertexcount(21)]
void silGeom( triangleadj GEOMETRYin input[6], inout TriangleStream<GEOMETRYout> TriStream )
{

	//Calculate the triangle normal and view direction.
	float3 normalTrian = getNormal( input[0].Pos.xyz, input[2].Pos.xyz, input[4].Pos.xyz );
	float3 viewDirect = normalize(-input[0].Pos.xyz - input[2].Pos.xyz - input[4].Pos.xyz);
	
	//If the triangle is frontfacing
	[branch]if(dot(normalTrian,viewDirect) > 0.0f)
	{		
		[loop]for(uint i = 0; i < 6; i+=2)
		{
			//Calculate the normal for this triangle.
			float auxIndex = (i+2)%6;
			float3 auxNormal = getNormal( input[i].Pos.xyz, input[i+1].Pos.xyz, input[auxIndex].Pos.xyz );
			float3 auxDirect = normalize(- input[i].Pos.xyz - input[i+1].Pos.xyz - input[auxIndex].Pos.xyz);
			
			//If the triangle is backfacing
			[branch]if(dot(auxNormal,auxDirect) <= 0.0f)
			{							
				//Transform the positions to screen space.
				float4 transPos1 = mul(input[i].Pos,projMatrix);
				transPos1 = transPos1/transPos1.w;
				float4 transPos2 = mul(input[auxIndex].Pos,projMatrix);
				transPos2 = transPos2/transPos2.w;
				
				//Calculate the edge direction in screen space.
				float2 edgeDirection = normalize(transPos2.xy - transPos1.xy);
				
				//Calculate the extrude vector in screen space.
				float4 extrudeDirection = float4(normalize(float2(-edgeDirection.y,edgeDirection.x)),0.0f,0.0f);
				
				//Calculate the extrude vector alogn the vertex normal in screen space.
				float4 normExtrude1 = mul(input[i].Pos + input[i].Normal,projMatrix);
				normExtrude1 = normExtrude1 / normExtrude1.w;
				normExtrude1 = normExtrude1 - transPos1;				
				normExtrude1 = float4(normalize(normExtrude1.xy),0.0f,0.0f);
				float4 normExtrude2 = mul(input[auxIndex].Pos + input[auxIndex].Normal,projMatrix);
				normExtrude2 = normExtrude2 / normExtrude2.w;
				normExtrude2 = normExtrude2 - transPos2;
				normExtrude2 = float4(normalize(normExtrude2.xy),0.0f,0.0f);
								
				//Detect if the extrude direction and the normal have opposite directions.
				float dot1 = dot(extrudeDirection.xy,normExtrude1.xy);	
				float dot2 = dot(extrudeDirection.xy,normExtrude2.xy);		
				
				//Scale the extrude directions with the edge size.
				normExtrude1 = normExtrude1 * edgeSize;
				normExtrude2 = normExtrude2 * edgeSize;
				extrudeDirection = extrudeDirection * edgeSize;
				
				//Calculate the extruded vertexs.
				float4 normVertex1 = (dot1 < 0.0f)?transPos1 + (normExtrude1*-1.0f):transPos1 + normExtrude1;
				float4 extruVertex1 = (transPos1 + extrudeDirection);
				float4 normVertex2 = (dot2 < 0.0f)?transPos2 + (normExtrude2*-1.0f):transPos2 + normExtrude2;				
				float4 extruVertex2 = (transPos2 + extrudeDirection);
				
				//Calculate the poligons distances for interpolate the position for correct texturing.
				float a = length(extruVertex1 - normVertex1);
				float b = length(transPos1 - transPos2);
				float c = length(extruVertex2 - normVertex2);
				float d = a + b + c;
				
				//Create the output polygons
				GEOMETRYout outVert;
				
				outVert.Pos = float4(normVertex1.xyz,1.0f);
				outVert.Uv = float3(transPos1.xy,0.0f);
				TriStream.Append(outVert);
				
				outVert.Pos = float4(extruVertex1.xyz,1.0f);
				outVert.Uv = float3(transPos1.xy+(edgeDirection*((b*a)/d)),0.0f);
				TriStream.Append(outVert);
				
				outVert.Pos = float4(transPos1.xyz,1.0f);
				outVert.Uv = float3(transPos1.xy,1.0f);
				TriStream.Append(outVert);
				
				outVert.Pos = float4(extruVertex2.xyz,1.0f);
				outVert.Uv = float3(transPos2.xy-(edgeDirection*((b*c)/d)),0.0f);
				TriStream.Append(outVert);
				
				outVert.Pos = float4(transPos2.xyz,1.0f);
				outVert.Uv = float3(transPos2.xy,1.0f);
				TriStream.Append(outVert);
				
				outVert.Pos = float4(normVertex2.xyz,1.0f);
				outVert.Uv = float3(transPos2.xy,0.0f);
				TriStream.Append(outVert);
				
				TriStream.RestartStrip();					
			}
		}
	}
	
	//Create the mesh triangle
	GEOMETRYout outVert;
	
	outVert.Pos = mul(input[0].Pos,projMatrix);
	outVert.Uv = float3(0.0f,0.0f,2.0f);
	TriStream.Append(outVert);
	
	outVert.Pos = mul(input[2].Pos,projMatrix);
	outVert.Uv = float3(0.0f,0.0f,2.0f);
	TriStream.Append(outVert);
	
	outVert.Pos = mul(input[4].Pos,projMatrix);
	outVert.Uv = float3(0.0f,0.0f,2.0f);
	TriStream.Append(outVert);
	
	TriStream.RestartStrip();	
}

////////////////////PIXEL SHADER/////////////////////////


float4 silPixel(PIXELin inPut):SV_Target
{
	float4 col;
	if(inPut.UV.z < 1.5f)
	{
	
		//Initial texture coordinate.
		float2 coord = float2(0.0f,inPut.UV.z);
		
		//Vector from the projected center bounding box to 
		//the location.
		float2 vect = inPut.UV.xy - aabbPos.xy;
		
		//Calculate the polar coordinate.
		float angle = atan(vect.y/vect.x);
		angle = (vect.x < 0.0f)?angle+PI:(vect.y < 0.0f)?angle+(2*PI):angle;
		
		//Assign the angle plus distance to the u texture coordinate.
		coord.x = ((angle/(2*PI)) + (length(vect)*lengthPer))*scale;
		
		//Get the texture colour.
		col = texureDiff.Sample(samLinear,coord); 
	    
		//Alpha test.
		if(col.a < 0.4f)
			discard;
		//col = float4(1.0f,0.0f,0.0f,1.0f);
			
	}else{
		col = float4(1.0f,1.0f,1.0f,1.0f);
	}
		
	//Return colour.
	return col;
}

////////////////////TECHNIQUE/////////////////////////

technique10 Silhouette
{
    pass p0
    {
        SetVertexShader( CompileShader( vs_4_0, silVertex() ) );
        SetGeometryShader( CompileShader( gs_4_0, silGeom() ) );
        SetPixelShader( CompileShader( ps_4_0, silPixel() ) );
        
        SetDepthStencilState( EnableDepth, 0 );
        SetBlendState( NoBlend, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xFFFFFFFF );
    }
};
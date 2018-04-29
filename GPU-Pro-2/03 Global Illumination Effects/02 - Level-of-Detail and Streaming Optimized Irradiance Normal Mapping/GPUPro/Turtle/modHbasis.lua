-- Globals
shbands = 3
shbasis = shbands*shbands

--This defines the range the linear and quadratic coefficients lie in: -compressRange..compressRange
--This can be set by hand or defined by the absolute maximum value in the textures [texturename]_1..5
--The textures need to be range decompressed by the application using this value
compressRange = 0.75

function setup()

	bset("gather.sampletype", "indirect illumination")
	bset("gather.space", "tangent space") --we need radiance in tangent space
	bset("gather.distribution", "equiareal")
	bset("gather.minsamples", 100)
	bset("gather.maxsamples", 500)
	bset("gather.useRadianceCache", true)
	bset("gather.coneangle", 360) --results look weird with 180
	
	--bset("gather.clamp", false) --set to false for HDR output

	bset("basis.type", "sh") --Turtle has SH build in
	bset("basis.rgb", true)
	bset("basis.sh.bands", shbands)

	bset("output.size", 24)

end

function bake() 


	ctR = getBasisCoefficients(1) --this function performs a Monte-Carlo-Integration, returning the SH coefficient vector
	ctG = getBasisCoefficients(2)
	ctB = getBasisCoefficients(3)

	nlights = getLights()  --the standard point lights are expanded into SH and added 
	for i = 1, nlights do
		col = getLightCol(i)
		if (not getLightAmb(i)) then
			localLightDir = normalize(worldToGather(getLightDir(i)))
			c = evalSHBasis(localLightDir, shbands)
			for j = 1, shbasis do
				ctR[j] = ctR[j] + c[j] * col[1]
				ctG[j] = ctG[j] + c[j] * col[2]
				ctB[j] = ctB[j] + c[j] * col[3]
			end
		end
	end
	
-- diffuse convolution using Funk-Hecke theorem
  
  diff_kernel = {1.0,2.0/3.0,1.0/4.0} --built in div by pi, so no pi in evaluation
  
  for l = 0, shbands-1 do --loop with index start 0
    for m = -l, l do
      
      ser = (l*(l+1)+m)+1 --correct for index start 1, LUA starts with 1
      ctR[ser] = ctR[ser] * diff_kernel[l+1] --correction for index start 1
      ctG[ser] = ctG[ser] * diff_kernel[l+1]
	  ctB[ser] = ctB[ser] * diff_kernel[l+1]
	  end
	end
	
--projection into the modified H-basis
	
	HbasisR = {}
	HbasisG = {}
	HbasisB = {}


	HbasisR[1] = 0.70711 * ctR[1] + 1.2247  * ctR[3] + 1.1859 * ctR[7]
	HbasisR[2] = 0.70711 * ctR[2] + 0.59293 * ctR[6]
	HbasisR[3] = -0.70711* ctR[3] - 1.3693  * ctR[7]
	HbasisR[4] = 0.70711 * ctR[4] + 0.59293 * ctR[8]
	HbasisR[5] = 0.70711 * ctR[5] 
	HbasisR[6] = 0.70711 * ctR[9]
	
	HbasisG[1] = 0.70711 * ctG[1] + 1.2247  * ctG[3] + 1.1859 * ctG[7]
	HbasisG[2] = 0.70711 * ctG[2] + 0.59293 * ctG[6]
	HbasisG[3] = -0.70711* ctG[3] - 1.3693  * ctG[7]
	HbasisG[4] = 0.70711 * ctG[4] + 0.59293 * ctG[8]
	HbasisG[5] = 0.70711 * ctG[5] 
	HbasisG[6] = 0.70711 * ctG[9]
	
	HbasisB[1] = 0.70711 * ctB[1] + 1.2247  * ctB[3] + 1.1859 * ctB[7]
	HbasisB[2] = 0.70711 * ctB[2] + 0.59293 * ctB[6]
	HbasisB[3] = -0.70711* ctB[3] - 1.3693  * ctB[7]
	HbasisB[4] = 0.70711 * ctB[4] + 0.59293 * ctB[8]
	HbasisB[5] = 0.70711 * ctB[5] 
	HbasisB[6] = 0.70711 * ctB[9]

--debug range, red means lightmap is >1, green means abs(h_2..5) > 0.75

--	if HbasisR[1] > 1.0 then
--		HbasisR[1] = 1.0  
--		HbasisG[1] = 0.0
--		HbasisB[1] = 0.0
--	end
--	if HbasisG[1] > 1.0 then
--		HbasisR[1] = 1.0  
--		HbasisG[1] = 0.0
--		HbasisB[1] = 0.0
--	end

--	if HbasisB[1] > 1.0 then
--		HbasisR[1] = 1.0  
--		HbasisG[1] = 0.0
--		HbasisB[1] = 0.0
--	end


--	for i = 2, 6 do
--		if math.abs(HbasisR[i]) > 0.75 then 
--			HbasisR[i] = 0.0  
--			HbasisG[i] = 1.0
--			HbasisB[i] = 0.0
--		end
	
--		if math.abs(HbasisG[i]) > 0.75 then 
--			HbasisR[i] = 0.0  
--			HbasisG[i] = 1.0
--			HbasisB[i] = 0.0
--		end
	
--		if math.abs(HbasisB[i]) > 0.75 then 
--			HbasisR[i] = 0.0  
--			HbasisG[i] = 1.0
--			HbasisB[i] = 0.0
--		end

--	end		
	
		
--Convert first coefficient to sRGB
    HbasisR[1] = math.pow(HbasisR[1],1.0/2.2)
    HbasisG[1] = math.pow(HbasisG[1],1.0/2.2)
    HbasisB[1] = math.pow(HbasisB[1],1.0/2.2)
	

--range compressing with a global factor all but the constant term 
--These coefficients are transported in linear space
  for i = 2, 6 do
    HbasisR[i] = (HbasisR[i]+compressRange)/(2.0*compressRange)
    HbasisG[i] = (HbasisG[i]+compressRange)/(2.0*compressRange)
    HbasisB[i] = (HbasisB[i]+compressRange)/(2.0*compressRange)
  end

--neagtive clamp light map, sometimes there are values very slightly in the negative
  if HbasisR[1] < 0.0 then HbasisR[1] = 0.0 end  
  if HbasisG[1] < 0.0 then HbasisG[1] = 0.0 end 
  if HbasisB[1] < 0.0 then HbasisB[1] = 0.0 end 
  
--output
    out = {}
    
    out[1] = HbasisR[1] -- contains light map
	out[2] = HbasisG[1]
	out[3] = HbasisB[1]
    out[4] = 1.0
    
    out[5] = HbasisR[2]
	out[6] = HbasisG[2]
	out[7] = HbasisB[2]
    out[8] = 1.0
    
    out[9] = HbasisR[3]
	out[10] = HbasisG[3]
	out[11] = HbasisB[3]
    out[12] = 1.0
    
    out[13] = HbasisR[4]
	out[14] = HbasisG[4]
	out[15] = HbasisB[4]
    out[16] = 1.0
    
    out[17] = HbasisR[5]
	out[18] = HbasisG[5]
	out[19] = HbasisB[5]
    out[20] = 1.0
	
	out[21] = HbasisR[6]
	out[22] = HbasisG[6]
	out[23] = HbasisB[6]
    out[24] = 1.0

	
	return out
end

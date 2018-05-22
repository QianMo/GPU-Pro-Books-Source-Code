    var gl;

	function checkGLError() {
    // Add enum strings to context (for debugging purposes)
    if (gl.enum_strings === undefined) {
        gl.enum_strings = {};
        for (var propertyName in gl) {
          if (typeof gl[propertyName] == 'number') {
            gl.enum_strings[gl[propertyName]] = propertyName;
          }
        }
    }

    var error = gl.getError();
    if (error != gl.NO_ERROR) {
        var str = "GL Error: " + error + " " + gl.enum_strings[error];
        console.log(str);
 //       throw str;
    }
}

    function initGL(canvas) {
        try {
            gl = canvas.getContext("experimental-webgl", {antialias : false});
            gl.viewportWidth = canvas.width;
            gl.viewportHeight = canvas.height;
        } catch (e) {
        }
        if (!gl) {
            alert("Could not initialise WebGL, sorry :-(");
        }
    }


    function getShader(gl, id) {
        var shaderScript = document.getElementById(id);
        if (!shaderScript) {
            return null;
        }

        var str = "";
        var k = shaderScript.firstChild;
        while (k) {
            if (k.nodeType == 3) {
                str += k.textContent;
            }
            k = k.nextSibling;
        }

        var shader;
        if (shaderScript.type == "x-shader/x-fragment") {
            shader = gl.createShader(gl.FRAGMENT_SHADER);
        } else if (shaderScript.type == "x-shader/x-vertex") {
            shader = gl.createShader(gl.VERTEX_SHADER);
        } else {
            return null;
        }

        gl.shaderSource(shader, str);
        gl.compileShader(shader);

        if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
            alert(gl.getShaderInfoLog(shader));
            return null;
        }

        return shader;
    }


    var shaderProgram;
	var fsQuadProgram;
	
	function makeProgram(vShader, fShader){
		var fragmentShader = getShader(gl, fShader);
        var vertexShader = getShader(gl, vShader);

        var shProgram = gl.createProgram();
        gl.attachShader(shProgram, vertexShader);
        gl.attachShader(shProgram, fragmentShader);
        gl.linkProgram(shProgram);

		if (!gl.getProgramParameter(shProgram, gl.LINK_STATUS)) {
            alert("Could not initialise shaders");
        }
		
		return shProgram;
	}	
	

    function initShaders() {
		
		shaderProgram = makeProgram("forward.vert", "forward.frag");

        gl.useProgram(shaderProgram);

        shaderProgram.vertexPositionAttribute = gl.getAttribLocation(shaderProgram, "aVertexPosition");
        gl.enableVertexAttribArray(shaderProgram.vertexPositionAttribute);

        shaderProgram.vertexNormalAttribute = gl.getAttribLocation(shaderProgram, "aVertexNormal");
        gl.enableVertexAttribArray(shaderProgram.vertexNormalAttribute);

        shaderProgram.textureCoordAttribute = gl.getAttribLocation(shaderProgram, "aTextureCoord");
        gl.enableVertexAttribArray(shaderProgram.textureCoordAttribute);

        shaderProgram.pMatrixUniform = gl.getUniformLocation(shaderProgram, "uPMatrix");
        shaderProgram.mvMatrixUniform = gl.getUniformLocation(shaderProgram, "uMVMatrix");
        shaderProgram.nMatrixUniform = gl.getUniformLocation(shaderProgram, "uNMatrix");
        shaderProgram.samplerUniform = gl.getUniformLocation(shaderProgram, "uSampler");
		shaderProgram.render_mode = gl.getUniformLocation(shaderProgram, "render_mode");
		
		
		fsQuadProgram = makeProgram("simple.vert", "simple.frag");
		
		gl.useProgram(fsQuadProgram);
		fsQuadProgram.vertexPositionAttribute = gl.getAttribLocation(fsQuadProgram, "aVertexPosition");
        gl.enableVertexAttribArray(fsQuadProgram.vertexPositionAttribute);
		
		fsQuadProgram.samplerUniform = gl.getUniformLocation(fsQuadProgram, "uSampler");
		fsQuadProgram.filterUniform = gl.getUniformLocation(fsQuadProgram, "filter_type");
		fsQuadProgram.render_mode = gl.getUniformLocation(fsQuadProgram, "render_mode");
		
    }
	
    var rttFramebuffer;
    var rttTexture;
	
	function initTextureFramebuffer() {
        rttFramebuffer = gl.createFramebuffer();
        gl.bindFramebuffer(gl.FRAMEBUFFER, rttFramebuffer);
		//Google chrome requires poer of two render targets!
        rttFramebuffer.width = 1024;
        rttFramebuffer.height = 512;

        rttTexture = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, rttTexture);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);

        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, rttFramebuffer.width, rttFramebuffer.height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);

        var renderbuffer = gl.createRenderbuffer();
        gl.bindRenderbuffer(gl.RENDERBUFFER, renderbuffer);
        gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT16, rttFramebuffer.width, rttFramebuffer.height);

        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, rttTexture, 0);
        gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, renderbuffer);

        gl.bindTexture(gl.TEXTURE_2D, null);
        gl.bindRenderbuffer(gl.RENDERBUFFER, null);
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    }

	var TextureIsReady=0;

    function handleLoadedTexture(texture) {
        gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
        gl.bindTexture(gl.TEXTURE_2D, texture);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, texture.image);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_NEAREST);
        gl.generateMipmap(gl.TEXTURE_2D);

        gl.bindTexture(gl.TEXTURE_2D, null);
		
		TextureIsReady=1;
    }


    var galvanizedTexture;

    function initTextures() {

        galvanizedTexture = gl.createTexture();
        galvanizedTexture.image = new Image();
        galvanizedTexture.image.onload = function () {
            handleLoadedTexture(galvanizedTexture)
        }
        galvanizedTexture.image.src = "testtexture.jpg";
    }


    var mvMatrix = mat4.create();
    var mvMatrixStack = [];
    var pMatrix = mat4.create();

    function mvPushMatrix() {
        var copy = mat4.create();
        mat4.set(mvMatrix, copy);
        mvMatrixStack.push(copy);
    }

    function mvPopMatrix() {
        if (mvMatrixStack.length == 0) {
            throw "Invalid popMatrix!";
        }
        mvMatrix = mvMatrixStack.pop();
    }

    function setMatrixUniforms() {
        gl.uniformMatrix4fv(shaderProgram.pMatrixUniform, false, pMatrix);
        gl.uniformMatrix4fv(shaderProgram.mvMatrixUniform, false, mvMatrix);

        var normalMatrix = mat3.create();
        mat4.toInverseMat3(mvMatrix, normalMatrix);
        mat3.transpose(normalMatrix);
        gl.uniformMatrix3fv(shaderProgram.nMatrixUniform, false, normalMatrix);
    }

    function degToRad(degrees) {
        return degrees * Math.PI / 180;
    }

	var teapot_model = null;

    function loadTeapot() {	
		var teapot = create_teapot(25, 16);
		teapot_model = new $.glModel2(gl, teapot);
	
    }
	
    var quadPositionBuffer;
	var quadIndexBuffer;
	
	function initBuffers() {
		quadPositionBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, quadPositionBuffer);
        vertices = [
            // Front face
            -1.0, -1.0,  0.0,
             1.0, -1.0,  0.0,
             1.0,  1.0,  0.0,
            -1.0,  1.0,  0.0,
			];
		gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices), gl.STATIC_DRAW);
        quadPositionBuffer.itemSize = 3;
        quadPositionBuffer.numItems = 4;
	
		quadIndexBuffer = gl.createBuffer();
		gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, quadIndexBuffer);
		var indices = [ 0, 1, 2, 0, 2, 3 ];	
		gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(indices), gl.STATIC_DRAW);
        quadIndexBuffer.itemSize = 1;
        quadIndexBuffer.numItems = 6;
		
	}
	
	
    var teapotAngle = 0;
		
	 function drawRTTScene() {
		gl.viewport(0, 0, gl.viewportWidth, gl.viewportHeight);
		gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
		
		


		var rnd_mode =  parseInt(document.getElementById("render_mode").value,10);
		
        mat4.identity(mvMatrix);

        mat4.translate(mvMatrix, [0, -5, -150]);
        mat4.rotate(mvMatrix, degToRad(30), [1, 0, 0]);
        mat4.rotate(mvMatrix, degToRad(teapotAngle), [0, 1, 0]);
	    //mat4.rotate(mvMatrix, degToRad(40), [0, 1, 0]);

		gl.useProgram(shaderProgram);
		
        gl.activeTexture(gl.TEXTURE0);
		gl.bindTexture(gl.TEXTURE_2D, galvanizedTexture);

        gl.uniform1i(shaderProgram.samplerUniform, 0);
		gl.uniform1i(shaderProgram.render_mode,rnd_mode);

		gl.bindBuffer(gl.ARRAY_BUFFER, teapot_model.VertexBuffer);
		
		
		function set_vattribf(s, stream, attrib) {
			var loc = gl.getAttribLocation(shaderProgram, attrib);
			gl.bindAttribLocation(shaderProgram, loc, attrib);
			gl.vertexAttribPointer(loc, s, gl.FLOAT, false, 0, teapot_model.VertexStreamBufferOffsets[stream]);
			gl.enableVertexAttribArray(loc);
		}

		set_vattribf(3, "Positions", "aVertexPosition");
		set_vattribf(2, "Texcoords", "aTextureCoord");
		set_vattribf(3, "Normals",   "aVertexNormal");

		gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, teapot_model.IndexBuffer);
		setMatrixUniforms();
		if(TextureIsReady)
		gl.drawElements(gl.TRIANGLES, teapot_model.IndexCount, teapot_model.IndexStreamGLType, 0);
	   
		
	 }
	 
	 

    function drawScene() {
		var rnd_mode = parseInt(document.getElementById("render_mode").value,10);
		//enable / disable the filter control depending on rendering mode
		if(rnd_mode!=1 && rnd_mode!=4 ){
			document.getElementById('filter').disabled = true;
			gl.clearColor( 1.0,  1.0,  1.0, 1.0);
			if(document.getElementById('contrast').checked)
				gl.clearColor( 1.0,  0.5,  1.0, 1.0); //high contrast magenta color
		}
		else{
			document.getElementById('filter').disabled = false;
			gl.clearColor( 1.0,  0.5,  0.5, 1.0);  //white in scaled YCoCg
			
			if(document.getElementById('contrast').checked)
				gl.clearColor( 1.0,  0.0,  0.0, 1.0);  //high contrast magenta color in scaled YCoCg
		}
		
		gl.bindFramebuffer(gl.FRAMEBUFFER, rttFramebuffer);
        drawRTTScene();
		gl.bindFramebuffer(gl.FRAMEBUFFER, null);

		gl.viewport(0, 0, gl.viewportWidth, gl.viewportHeight);
		gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
		
		gl.useProgram(fsQuadProgram);
		gl.activeTexture(gl.TEXTURE0);
		gl.bindTexture(gl.TEXTURE_2D, rttTexture);
		
		gl.uniform1i(fsQuadProgram.samplerUniform, 0);
		var filter = document.getElementById("filter").value;
		gl.uniform1i(fsQuadProgram.filterUniform, parseInt(filter,10));
		gl.uniform1i(fsQuadProgram.render_mode, rnd_mode);
		
		
			
		gl.bindBuffer(gl.ARRAY_BUFFER, quadPositionBuffer);
        gl.vertexAttribPointer(fsQuadProgram.vertexPositionAttribute, quadPositionBuffer.itemSize, gl.FLOAT, false, 0, 0);
		gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, quadIndexBuffer);
        gl.drawElements(gl.TRIANGLES, quadIndexBuffer.numItems, gl.UNSIGNED_SHORT, 0);
	
    }

    var lastTime = 0;
	var frame_num=0;
	var fps_lastTime = 0;
	
	function trunc(n) {
	  return n | 0; // bitwise operators convert operands to 32-bit integers
	}

    function animate() {
        var timeNow = new Date().getTime();
        if (lastTime != 0) {
            var elapsed = timeNow - lastTime;
			var animation = document.getElementById("animate").checked;
				if(animation)
            teapotAngle += 0.02 * elapsed;
        }
       
		if (frame_num>60){
			var fps = timeNow - fps_lastTime;
			$("#frameRate").text((60*1000/(fps)).toFixed(1));
			frame_num=0;
			fps_lastTime= timeNow;
		}
		
		lastTime = timeNow;
		frame_num++;
    }

    function tick() {
 //       requestAnimationFrame(tick);
        drawScene();
        animate();
		setTimeout(tick, 0);
    }
	
	function setupGL() {
		//gl.clearColor(105./255, 172./255, 215./255, 1.0);
		//gl.clearColor(0.658, 0.296, 0.527, 1.0);
		

		gl.enable(gl.DEPTH_TEST);
		
		mat4.perspective(45, gl.viewportWidth / gl.viewportHeight, 1.0, 1000.0, pMatrix);
		
	}

    function webGLStart() {
        var canvas = document.getElementById("WebGL-Demo");
        initGL(canvas);
		initTextureFramebuffer();
		setupGL();
        initShaders();
		initBuffers();
        initTextures();
        loadTeapot();     

        tick();
    }

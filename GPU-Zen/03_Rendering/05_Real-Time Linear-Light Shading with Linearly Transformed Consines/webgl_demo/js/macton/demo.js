var TeapotDemo = function() 
{
  var c      = document.getElementById("macton-teapot-canvas");
  var width  = c.width;
  var height = c.height;
  var gl     = WebGLUtils.setupWebGL(c);

  if (!gl) return;
  
  // Add enum strings to context
  if (gl.enum_strings === undefined) {
    gl.enum_strings = { };
    for (var propertyName in gl) {
      if (typeof gl[propertyName] == 'number') {
        gl.enum_strings[gl[propertyName]] = propertyName;
      }
    }
  }

  var TeapotBumpReflectAttributeBindings = null;
  
  // Could manually set these bindings. But using best guess function gives same result.
  // var TeapotBumpReflectAttributeBindings =
  // { 
  //   Positions: 'g_Position', 
  //   Texcoords: 'g_TexCoord0', 
  //   Tangents:  'g_Tangent', 
  //   Binormals: 'g_Binormal',
  //   Normals:   'g_Normal'
  // };
  
  var bump_reflect_program = null;
  var teapot_model         = null;
  var bump_texture         = null;
  var spec_texture         = null;
  var env_texture          = null;
  
  // The "model" matrix is the "world" matrix in Standard Annotations
  // and Semantics
  var model      = new Matrix4x4();
  var view       = new Matrix4x4();
  var projection = new Matrix4x4();
  var controller = null;

  var bump_reflect_program_config = 
  {
  	VertexProgramURL:   'shaders/bump_reflect.vs',
    FragmentProgramURL: 'shaders/bump_blinn.fs',
  };
  
  var bump_texture_config =
  {
    Type:      'TEXTURE_2D',
//    ImageURL:  'images/bump.jpg',
    ImageURL:  'images/png/bump_0.png',
    MipURL:
    [
		'images/png/bump_1.png',
		'images/png/bump_2.png',
		'images/png/bump_3.png',
		'images/png/bump_4.png',
		'images/png/bump_5.png',
		'images/png/bump_6.png',
		'images/png/bump_7.png',
		'images/png/bump_8.png',
		'images/png/bump_9.png'
    ],
    TexParameters: 
    {
      TEXTURE_MIN_FILTER: 'LINEAR_MIPMAP_LINEAR',
      TEXTURE_MAG_FILTER: 'LINEAR',
      TEXTURE_WRAP_S:     'REPEAT',
      TEXTURE_WRAP_T:     'REPEAT'
    },
    PixelStoreParameters:
    {
      UNPACK_FLIP_Y_WEBGL: false
    },
  };
  
  var spec_texture_config =
  {
    Type:      'TEXTURE_2D',
    ImageURL:  'images/png/spec_0.png',
    MipURL:
    [
		'images/png/spec_1.png',
		'images/png/spec_2.png',
		'images/png/spec_3.png',
		'images/png/spec_4.png',
		'images/png/spec_5.png',
		'images/png/spec_6.png',
		'images/png/spec_7.png',
		'images/png/spec_8.png',
		'images/png/spec_9.png'
    ],
    TexParameters: 
    {
      TEXTURE_MIN_FILTER: 'LINEAR_MIPMAP_LINEAR',
      TEXTURE_MAG_FILTER: 'LINEAR',
      TEXTURE_WRAP_S:     'REPEAT',
      TEXTURE_WRAP_T:     'REPEAT'
    },
    PixelStoreParameters:
    {
      UNPACK_FLIP_Y_WEBGL: false
    },
  };
  
  var env_texture_config =
  {
    Type: 'TEXTURE_CUBE_MAP',
    ImageURL: 
    [ 
      'images/skybox-posx.jpg',
      'images/skybox-negx.jpg',
      'images/skybox-posy.jpg',
      'images/skybox-negy.jpg',
      'images/skybox-posz.jpg',
      'images/skybox-negz.jpg' 
    ],
    ImageType: 
    [ 
      'TEXTURE_CUBE_MAP_POSITIVE_X',
      'TEXTURE_CUBE_MAP_NEGATIVE_X',
      'TEXTURE_CUBE_MAP_POSITIVE_Y',
      'TEXTURE_CUBE_MAP_NEGATIVE_Y',
      'TEXTURE_CUBE_MAP_POSITIVE_Z',
      'TEXTURE_CUBE_MAP_NEGATIVE_Z'
    ],
    TexParameters: 
    {
      TEXTURE_WRAP_S:     'CLAMP_TO_EDGE',
      TEXTURE_WRAP_T:     'CLAMP_TO_EDGE',
      TEXTURE_MIN_FILTER: 'LINEAR_MIPMAP_LINEAR',
      TEXTURE_MAG_FILTER: 'LINEAR'
    },
    PixelStoreParameters:
    {
      UNPACK_FLIP_Y_WEBGL: false
    }
  };

  var shaders_loaded      = false;
  var model_loaded        = false;
  var bump_texture_loaded = false;
  var spec_texture_loaded = false;
  var env_texture_loaded  = false;
  
  var test = 0;

  var TryMain = function()
  {  
    if ( shaders_loaded && model_loaded && bump_texture_loaded && spec_texture_loaded && env_texture_loaded ) 
    {
      TeapotBumpReflectAttributeBindings = bump_reflect_program.CreateBestVertexBindings( teapot_model ); 
      main();
    }
  }

  var ProgramLoaded = function()
  {
    shaders_loaded = true;
    TryMain();
  }

  var ModelLoaded = function()
  {
    model_loaded = true;
    TryMain();
  }
  
  var bump_textureLoaded = function()
  {
    bump_texture_loaded = true;
    TryMain();
  }

  var spec_textureLoaded = function()
  {
    spec_texture_loaded = true;
    TryMain();
  }

  var env_textureLoaded = function()
  {
    env_texture_loaded = true;
    TryMain();
  }

  bump_reflect_program = new $.glProgram( gl, bump_reflect_program_config, ProgramLoaded     );
  teapot_model         = new $.glModel(   gl, 'models/teapot2.json',      ModelLoaded       );
  bump_texture         = new $.glTexture( gl, bump_texture_config,         bump_textureLoaded );
  spec_texture         = new $.glTexture( gl, spec_texture_config,         spec_textureLoaded );
  env_texture          = new $.glTexture( gl, env_texture_config,          env_textureLoaded  );
  
  function main() 
  {
    controller = new CameraController(c);
    // Try the following (and uncomment the "pointer-events: none;" in
    // the index.html) to try the more precise hit detection
    //  controller = new CameraController(document.getElementById("body"), c, gl);
    controller.onchange = function(xRot, yRot) {
        draw();
    };

    gl.enable(gl.DEPTH_TEST);
    gl.clearColor(0.18, 0.18, 0.18, 1.0);

    draw();
  }
  
  function draw() 
  {
    // Note: the viewport is automatically set up to cover the entire Canvas.
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    // Set up the model, view and projection matrices
    projection.loadIdentity();
    projection.perspective(45, width / height, 10, 500);
    view.loadIdentity();
    view.translate(0, -10, -200.0);

    // Add in camera controller's rotation
    model.loadIdentity();
    model.rotate(controller.xRot, 1, 0, 0);
    model.rotate(controller.yRot, 0, 1, 0);

    // Correct for initial placement and orientation of model
    //model.translate(0, -80, 0);
    //model.rotate(90, 1, 0, 0);

    bump_reflect_program.Use();

    // Compute necessary matrices
    var mvp = new Matrix4x4();
    mvp.multiply(model);
    mvp.multiply(view);
    mvp.multiply(projection);
    var worldInverseTranspose = model.inverse();
    worldInverseTranspose.transpose();
    var viewInverse = view.inverse();

    bump_reflect_program.BindUniform( 'world',                 model.elements );
    bump_reflect_program.BindUniform( 'worldInverseTranspose', worldInverseTranspose.elements );
    bump_reflect_program.BindUniform( 'worldViewProj',         mvp.elements );
    bump_reflect_program.BindUniform( 'viewInverse',           viewInverse.elements );
    bump_reflect_program.BindUniform( 'normalSampler',         bump_texture );
    bump_reflect_program.BindUniform( 'specSampler',           spec_texture );
    bump_reflect_program.BindUniform( 'envSampler',            env_texture.Texture );

    bump_reflect_program.BindModel( teapot_model, TeapotBumpReflectAttributeBindings );
    bump_reflect_program.DrawModel();

    $.glCheckError( gl, output );
  }
  
  function output(str) 
  {
    document.body.appendChild(document.createTextNode(str));
    document.body.appendChild(document.createElement("br"));
  }
}

$(document).ready( TeapotDemo );

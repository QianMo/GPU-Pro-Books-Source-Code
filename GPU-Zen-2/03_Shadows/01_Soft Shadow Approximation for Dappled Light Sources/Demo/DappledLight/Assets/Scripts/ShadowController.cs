using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEditor;
using UnityEngine.Rendering;

// Note:
// The Unity setup code is very much placeholder, and is not optimized for performance but for readability.
public class ShadowController : MonoBehaviour
{
    public bool showGUI = false;
    public Texture2D[] bokehShapeTextures;
    public Transform treeTransform;
    public Renderer bokehDebugRenderer;

    public GameObject receiverContainer;

    public float bokehRotationVelocity = 1f;
    public float neighborVarianceThreshold = 1f;

    public float weightThreshold = .95f;

    public float varianceThreshold = 3f;
    
    public Material shadowBokehPass;
    public Renderer debugger;

    public Shader shadowPassShader;
    public Light sourceLight;
    public int projectionSize = 10;
    public int shadowMapSize = 2048;
    public int pinholeGridSize = 10;

    public int pinholesPerCell = 20;

    public RenderTexture shadowMap;
    public RenderTexture bokehMap;

    public ComputeShader computePinholes;

    public ComputeShader computeTemporal;

    private Camera mainCamera;
    private Camera lightCamera;
    private Renderer[] receivers;

    private int computeKernelIndex;
    private int computeTemporalKernelIndex;

    private ComputeBuffer pinholeLinkBuffer;
    private ComputeBuffer pinholeOffsetBuffer;

    private ComputeBuffer temporalPinholeBuffer;
    private ComputeBuffer temporalPinholeCountBuffer;

    private int currentBokehShape = 0;

    private byte[] indexBufferReset;

    private bool enableBokehRotation = false;
    private float currentBokehRotation = 25f;

    private float bokehMaxSize = 0.01f;
    private float bokehSize = .00025f;
    private float bokehIntensity = 1f;

    public void Start()
    {
        this.receivers = receiverContainer.GetComponentsInChildren<Renderer>();

        shadowMap = new RenderTexture(shadowMapSize, shadowMapSize, 24, RenderTextureFormat.ARGBFloat);
        shadowMap.autoGenerateMips = true;
        shadowMap.name = "Shadow map";
        shadowMap.useMipMap = true;
        shadowMap.filterMode = FilterMode.Point;
        shadowMap.Create();

        bokehMap = new RenderTexture(shadowMapSize, shadowMapSize, 32, RenderTextureFormat.ARGBFloat);
        bokehMap.name = "BokehDebugMap";
        bokehMap.autoGenerateMips = true;
        bokehMap.useMipMap = true;
        bokehMap.enableRandomWrite = true;
        bokehMap.filterMode = FilterMode.Point;
        bokehMap.Create();

        // Ugly...
        lightCamera = new GameObject("LightCamera").AddComponent<Camera>();
        lightCamera.enabled = false;
        // lightCamera.gameObject.hideFlags = HideFlags.HideAndDontSave;
        lightCamera.clearFlags = CameraClearFlags.Color;
        lightCamera.backgroundColor = Color.black;
        lightCamera.farClipPlane = 1000f;
        lightCamera.useOcclusionCulling = false;
        lightCamera.nearClipPlane = 1f;
        lightCamera.renderingPath = RenderingPath.Forward;
        lightCamera.allowMSAA = false;
        lightCamera.orthographic = true;
        lightCamera.targetTexture = shadowMap;

        Shader.SetGlobalTexture("_ShadowDepth", shadowMap);
        Shader.SetGlobalTexture("_ShadowBokeh", bokehMap);
        Shader.SetGlobalTexture("_RandomSamplesTex", BuildSampleTexture());

        this.mainCamera = GetComponent<Camera>();
        //this.mainCamera.AddCommandBuffer(CameraEvent.BeforeForwardOpaque, shadowBuffer);

        debugger.material.mainTexture = bokehMap;

        computeKernelIndex = computePinholes.FindKernel("ComputePinholes");
        computeTemporalKernelIndex = computeTemporal.FindKernel("ComputeTemporalFilter");

        int maxPinholeCount = pinholesPerCell * pinholeGridSize * pinholeGridSize;
        int pinholeSizeBytes = 4 * 6; // 5 floats
        pinholeLinkBuffer = new ComputeBuffer(maxPinholeCount, pinholeSizeBytes, ComputeBufferType.Counter);
        pinholeLinkBuffer.SetData(Enumerable.Repeat((byte)0, maxPinholeCount * pinholeSizeBytes).ToArray());
        pinholeLinkBuffer.SetCounterValue(0);
        
        pinholeOffsetBuffer = new ComputeBuffer(pinholeGridSize * pinholeGridSize, 4, ComputeBufferType.Raw);
        indexBufferReset = Enumerable.Repeat((byte)0xFF, pinholeGridSize * pinholeGridSize * 4).ToArray();

        temporalPinholeBuffer = new ComputeBuffer(pinholeGridSize*pinholeGridSize*pinholesPerCell, pinholeSizeBytes, ComputeBufferType.Default);
        temporalPinholeBuffer.SetData(Enumerable.Repeat((byte)0, pinholeGridSize*pinholeGridSize*pinholesPerCell * pinholeSizeBytes).ToArray());

        int sizeOfCount = 4; // sizeof(int)
        temporalPinholeCountBuffer = new ComputeBuffer(pinholeGridSize * pinholeGridSize, sizeOfCount, ComputeBufferType.Default);
        temporalPinholeCountBuffer.SetData(Enumerable.Repeat((byte)0, pinholeGridSize * pinholeGridSize * sizeOfCount).ToArray());
    }

    public void OnDestroy()
    {
        pinholeLinkBuffer.Release();
        pinholeOffsetBuffer.Release();
        temporalPinholeBuffer.Release();
        temporalPinholeCountBuffer.Release();
    }

    private void DispatchCompute()
    {
        int threadsPerGroup = 32;

        // We need to reset the global buffer counter
        pinholeLinkBuffer.SetCounterValue(0);

        // Also, reset indices so we dont accumulate previous pinholes
        pinholeOffsetBuffer.SetData(indexBufferReset);

        int maxPinholeCount = pinholesPerCell * pinholeGridSize * pinholeGridSize;

        computePinholes.SetTexture(computeKernelIndex, Shader.PropertyToID("BokehMap"), bokehMap);
        computePinholes.SetTexture(computeKernelIndex, Shader.PropertyToID("ShadowMap"), shadowMap);
        
        computePinholes.SetInt(Shader.PropertyToID("ShadowMapWidth"), shadowMap.width);
        computePinholes.SetInt(Shader.PropertyToID("ShadowMapHeight"), shadowMap.height);
        computePinholes.SetInt(Shader.PropertyToID("PinholeGridSize"), pinholeGridSize);
        computePinholes.SetInt(Shader.PropertyToID("PinholesPerCell"), pinholesPerCell);
        computePinholes.SetInt(Shader.PropertyToID("TotalMaxPinholeCount"), maxPinholeCount);
        
        computePinholes.SetFloat(Shader.PropertyToID("NeighborVarianceThreshold"), neighborVarianceThreshold);
        computePinholes.SetFloat(Shader.PropertyToID("WeightThreshold"), weightThreshold);
        computePinholes.SetFloat(Shader.PropertyToID("VarianceThreshold"), varianceThreshold);
        
        computePinholes.SetBuffer(computeKernelIndex, Shader.PropertyToID("g_PinholeLinkBuffer"), pinholeLinkBuffer);
        computePinholes.SetBuffer(computeKernelIndex, Shader.PropertyToID("g_PinholeOffsetBuffer"), pinholeOffsetBuffer);

        // Every shadow map pixel
        int groupsX = shadowMap.width / threadsPerGroup;
        int groupsY = shadowMap.height / threadsPerGroup;
        computePinholes.Dispatch(computeKernelIndex, groupsX, groupsY, 1);

        // Temporal filter
        computeTemporal.SetInt(Shader.PropertyToID("PinholeGridSize"), pinholeGridSize);
        computeTemporal.SetInt(Shader.PropertyToID("PinholesPerCell"), pinholesPerCell);
        computeTemporal.SetBuffer(computeTemporalKernelIndex, Shader.PropertyToID("g_PinholeLinkBuffer"), pinholeLinkBuffer);
        computeTemporal.SetBuffer(computeTemporalKernelIndex, Shader.PropertyToID("g_PinholeOffsetBuffer"), pinholeOffsetBuffer);
        computeTemporal.SetBuffer(computeTemporalKernelIndex, Shader.PropertyToID("g_TemporalPinholesBuffer"), temporalPinholeBuffer);
        computeTemporal.SetBuffer(computeTemporalKernelIndex, Shader.PropertyToID("g_TemporalPinholesCountBuffer"), temporalPinholeCountBuffer);

        // Every grid cell
        groupsX = Mathf.CeilToInt(pinholeGridSize / (float)threadsPerGroup);
        groupsY = Mathf.CeilToInt(pinholeGridSize / (float)threadsPerGroup);
        computeTemporal.Dispatch(computeTemporalKernelIndex, groupsX, groupsY, 1);

        foreach(Renderer r in receivers)
        {
            // Shadow receiver material
            r.sharedMaterial.SetBuffer(Shader.PropertyToID("g_PinholeLinkBuffer"), pinholeLinkBuffer);
            r.sharedMaterial.SetBuffer(Shader.PropertyToID("g_PinholeOffsetBuffer"), pinholeOffsetBuffer);
            r.sharedMaterial.SetBuffer(Shader.PropertyToID("g_TemporalPinholesBuffer"), temporalPinholeBuffer);
            r.sharedMaterial.SetBuffer(Shader.PropertyToID("g_TemporalPinholesCountBuffer"), temporalPinholeCountBuffer);
            r.sharedMaterial.SetInt(Shader.PropertyToID("PinholeGridSize"), pinholeGridSize);
            r.sharedMaterial.SetInt(Shader.PropertyToID("PinholesPerCell"), pinholesPerCell);
            r.sharedMaterial.SetMatrix("_BokehRotation", Matrix4x4.TRS(Vector3.zero, Quaternion.Euler(Vector3.forward * currentBokehRotation), Vector3.one));
        }
    }

    private Texture2D BuildSampleTexture()
    {
        int samples = 2048;
        Texture2D tex = new Texture2D(samples, 1, TextureFormat.RGBAFloat, false);
        tex.filterMode = FilterMode.Point;
        tex.wrapMode = TextureWrapMode.Repeat;

        Color[] pixels = new Color[samples];

        for(int i = 0; i < samples; i++)
            pixels[i] = new Color(Random.value, Random.value, Random.value, Random.value) * 2f - new Color(1f, 1f, 1f, 1f);

        tex.SetPixels(pixels);
        tex.Apply();
        return tex;
    }

    private bool animateTree = false;

    private float lightRotation = .5f;

    public float containerRotation = 1f;

    private float animationTime = 0f;

    public void LateUpdate()
    {
        if(animateTree)
        {
            foreach(Transform t in treeTransform)
                t.localRotation = Quaternion.Euler(Vector3.up * animationTime * containerRotation);

            animationTime += Time.deltaTime;
        }

        if(enableBokehRotation)
            currentBokehRotation += bokehRotationVelocity * Time.deltaTime;

        if(Input.GetKeyDown(KeyCode.Escape))
            Application.Quit();

        lightCamera.orthographicSize = projectionSize;
        lightCamera.transform.parent = sourceLight.transform;
        lightCamera.transform.localPosition = Vector3.zero;
        lightCamera.transform.localRotation = Quaternion.identity;
        lightCamera.transform.localScale = Vector3.one;
        // lightCamera.allowMSAA = true; // Enable this for MSAA on the shadow pass

        Shader.SetGlobalMatrix("_LightTransform", lightCamera.projectionMatrix * lightCamera.worldToCameraMatrix);
        Shader.SetGlobalVector("_LightPosition", sourceLight.transform.position);
        Shader.SetGlobalVector("_LightDirection", sourceLight.transform.forward);
        Shader.SetGlobalInt("PinholeGridSize", pinholeGridSize);
        Shader.SetGlobalTexture("_BokehShape", bokehShapeTextures[currentBokehShape]);

        Shader.SetGlobalFloat("_BokehSize", bokehSize);
        Shader.SetGlobalFloat("_BokehMaxDistance", bokehMaxSize);
        Shader.SetGlobalFloat("_BokehIntensity", bokehIntensity);

        bokehDebugRenderer.material.mainTexture = bokehShapeTextures[currentBokehShape];

        lightCamera.ResetProjectionMatrix();
        lightCamera.RenderWithShader(shadowPassShader, "RenderType");
        
        DispatchCompute();

        if(Input.GetKeyDown(KeyCode.R))
            WriteFrameToDisk();
    }

    private void WriteFrameToDisk()
    {
        string path = System.IO.Path.Combine(System.IO.Path.Combine(Application.dataPath, ".."), "screenshot.png");
        Debug.Log("Saving screenshot to " + path);
        ScreenCapture.CaptureScreenshot(path, 1);
    }

    public void OnGUI()
    {
        if(showGUI)
        {
            // GUI.skin = EditorGUIUtility.GetBuiltinSkin(EditorSkin.Scene);
            GUILayout.BeginVertical("Shadow Bokeh", "Window");

            GUILayout.BeginHorizontal();
                GUILayout.Space(30f);
                GUILayout.Label(bokehShapeTextures[currentBokehShape], GUILayout.Width(128f), GUILayout.Height(128f));
            GUILayout.EndHorizontal();

            if(GUILayout.Button(animateTree ? "Stop animation" : "Play animation"))
                animateTree = !animateTree;
            
            GUILayout.BeginHorizontal();
            if(GUILayout.Button("Previous bokeh"))
                currentBokehShape = (currentBokehShape == 0) ? bokehShapeTextures.Length - 1 : currentBokehShape - 1;
                
            if(GUILayout.Button("Next bokeh"))
                currentBokehShape = (currentBokehShape + 1 == bokehShapeTextures.Length) ? 0 : currentBokehShape + 1;
            GUILayout.EndHorizontal();
            
            // GUILayout.Label("Light rotation");
            // lightRotation = GUILayout.HorizontalSlider(lightRotation, 0f, 1f);

            enableBokehRotation = GUILayout.Toggle(enableBokehRotation, "Bokeh rotation");
            
            GUILayout.BeginHorizontal();
            GUILayout.Label("Bokeh Intensity");
            bokehIntensity = GUILayout.HorizontalSlider(bokehIntensity, 0f, 1f);
            GUILayout.EndHorizontal();
            
            GUILayout.Label("Bokeh Size: " + bokehSize);
            bokehSize = GUILayout.HorizontalSlider(bokehSize, 0f, .01f);
            
            GUILayout.Label("Bokeh Max Size: " + bokehMaxSize);
            bokehMaxSize = GUILayout.HorizontalSlider(bokehMaxSize, 0f, .25f);

            GUILayout.Label("Shadow map size: " + shadowMapSize + "x" + shadowMapSize);
            GUILayout.Label("Uniform grid cells: " + pinholeGridSize + "x"+ pinholeGridSize);
            GUILayout.Label("Pinholes per cell: " + pinholesPerCell);

            GUILayout.EndVertical();        
        }
    }
}

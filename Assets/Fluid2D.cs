using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Fluid2D : MonoBehaviour
{
    public Vector2Int gridCount = new Vector2Int(256, 256);
    const int dim = 2;
    public float cellSize = .01f;
    public float timeStep=.01f;
    public float viscosity = .01f;

    public int pressure_iteration = 40;//40-80
    public int viscosity_iteration = 20;//20-50

    RenderTexture velocity;
    RenderTexture velocity_new;
    RenderTexture velocity_divergence;
    RenderTexture pressure;
    RenderTexture pressure_new;


    public ComputeShader shader;

    private void Start()
    {
        Init();
        SetBoundaryCondition();
    }
    private void FixedUpdate()
    {
        Step();
        GetComponent<MeshRenderer>().material.SetTexture("_MainTex", velocity);
    }

    public void Init()
    {
        var desc = new RenderTextureDescriptor(gridCount.x, gridCount.y);
        desc.enableRandomWrite = true;

        desc.colorFormat = RenderTextureFormat.RGFloat;//float2
        velocity = new RenderTexture(desc);
        velocity_new = new RenderTexture(desc);

        desc.colorFormat = RenderTextureFormat.RFloat;//float
        velocity_divergence = new RenderTexture(desc);
        pressure = new RenderTexture(desc);
        pressure_new = new RenderTexture(desc);
    }
    public void SetBoundaryCondition()
    {
        int kid; Vector3Int groupCount;
        // Advect Diffuse AddForce RemovePressure
        shader.SetInts("gridCount", new int[] { gridCount.x, gridCount.y });
        shader.SetFloat("cellSize", cellSize);
        shader.SetFloat("timeStep", timeStep);

        {
            kid = shader.FindKernel("Init");
            shader.SetTexture(kid, "output_x", velocity);
            groupCount = CalcGroupCount(kid);
            shader.Dispatch(kid, groupCount.x, groupCount.y, groupCount.z);

        }

    }
    public void Step()
    {
        int kid; Vector3Int groupCount;
        // Advect Diffuse AddForce RemovePressure
        shader.SetInts("gridCount", new int[] { gridCount.x, gridCount.y });
        shader.SetFloat("cellSize", cellSize);
        shader.SetFloat("timeStep", timeStep);

        //Advect Velocity Field
        {
            kid = shader.FindKernel("AdvectionFloat2");
            shader.SetTexture(kid, "input_x", velocity);
            shader.SetTexture(kid, "output_x", velocity_new);
            shader.SetTexture(kid, "input_vector", velocity);
            groupCount = CalcGroupCount(kid);
            shader.Dispatch(kid, groupCount.x, groupCount.y, groupCount.z);
            Swap(ref velocity_new, ref velocity);
        }

        //Diffuse Velocity using Jacobi Method
        if (viscosity > 0)
            for (int i = 0; i < viscosity_iteration; ++i)
            {
                kid = shader.FindKernel("JacobiIterationFloat2");
                shader.SetTexture(kid, "input_x", velocity);
                shader.SetTexture(kid, "output_x", velocity_new);
                shader.SetTexture(kid, "input_b", velocity);
                float tmp = cellSize * cellSize + 2 * dim * viscosity * timeStep;
                float alpha_inv_beta = cellSize * cellSize / tmp;
                float inv_beta = viscosity * timeStep / tmp;
                shader.SetFloat("jac_alpha_inv_beta", alpha_inv_beta);
                shader.SetFloat("jac_inv_beta", inv_beta);
                groupCount = CalcGroupCount(kid);
                shader.Dispatch(kid, groupCount.x, groupCount.y, groupCount.z);
                Swap(ref velocity_new, ref velocity);
            }

        //Remove Pressure
        //b=¨Œ@w
        {
            kid = shader.FindKernel("Divergence");
            shader.SetTexture(kid, "input_vector", velocity);
            shader.SetTexture(kid, "output_x", velocity_divergence);
            groupCount = CalcGroupCount(kid);
            shader.Dispatch(kid, groupCount.x, groupCount.y, groupCount.z);
        }
        //¨Œ@¨Œp=b
        for(int i=0;i<pressure_iteration;++i)
        {
            {
                kid = shader.FindKernel("JacobiIterationFloat");
                shader.SetTexture(kid, "input_x", pressure);
                shader.SetTexture(kid, "output_x", pressure_new);
                shader.SetTexture(kid, "input_b", velocity_divergence);

                float alpha_inv_beta = -cellSize * cellSize / (2 * dim);
                float inv_beta = 1f / (2 * dim);
                shader.SetFloat("jac_alpha_inv_beta", alpha_inv_beta);
                shader.SetFloat("jac_inv_beta", inv_beta);
                groupCount = CalcGroupCount(kid);
                shader.Dispatch(kid, groupCount.x, groupCount.y, groupCount.z);
                Swap(ref pressure_new, ref pressure);
            }
            {
                kid = shader.FindKernel("UpdateNeumannBoundaryFloat");
                shader.SetTexture(kid, "output_x", pressure);
                groupCount = CalcGroupCount(kid, isBoundary: true);
                shader.Dispatch(kid, groupCount.x, groupCount.y, groupCount.z);
            }
        }
        //u=w-¨Œp
        {
            kid = shader.FindKernel("MinusGradient");
            shader.SetTexture(kid, "input_scalar", pressure);
            shader.SetTexture(kid, "input_x", velocity);
            shader.SetTexture(kid, "output_x", velocity_new);
            groupCount = CalcGroupCount(kid);
            shader.Dispatch(kid, groupCount.x, groupCount.y, groupCount.z);
            Swap(ref velocity_new, ref velocity);
        }
    }
    Vector3Int CalcGroupCount(int kid,bool isBoundary=false)
    {
        shader.GetKernelThreadGroupSizes(kid, out uint sx, out uint sy, out uint sz);
        if (isBoundary)
            return new Vector3Int(Mathf.CeilToInt((float)gridCount.x * gridCount.y / sx), 1, 1);
        else
            return new Vector3Int(
                    Mathf.CeilToInt((float)gridCount.x / sx),
                    Mathf.CeilToInt((float)gridCount.y / sy),
                    1
                );
    }
    void Swap(ref RenderTexture a, ref RenderTexture b)
    {
        var tmp = a; a = b; b = tmp;
    }

}

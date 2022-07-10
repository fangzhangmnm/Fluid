int2 gridCount;
// Upgrade NOTE: excluded shader from DX11 because it uses wrong array syntax (type[size] name)
#pragma exclude_renderers d3d11
float cellSize;
float timeStep;
float2 GetUV(uint3 id){
    return (id.xy+.5)/gridCount;
}
bool IsOnBoundary(uint3 id){
    return ((id.x==0 || id.x==gridCount.x-1) && 0<=id.y && id.y<gridCount.y)
         ||((id.y==0 || id.y==gridCount.y-1) && 0<=id.x && id.x<gridCount.x);
}
bool IsInBulk(uint3 id){
    return (0<id.x && id.x<gridCount.x-1 &&
            0<id.y && id.y<gridCount.y-1);
}
bool IsInBulkBoundary(uint3 id){
    return (0<=id.x && id.x<gridCount.x &&
            0<=id.y && id.y<gridCount.y);
}


Texture2D<TYPE> input_x,input_b;
RWTexture2D<TYPE> output_x;

Texture2D<float> input_scalar;
Texture2D<float2> input_vector;



#pragma kernel JacobiIterationFloat TYPE=float JacobiIteration=JacobiIterationFloat
#pragma kernel JacobiIterationFloat2 TYPE=float2 JacobiIteration=JacobiIterationFloat2
// input_x, output_x, input_b
float jac_alpha_inv_beta,jac_inv_beta;

[numthreads(8,8,1)]
void JacobiIteration(uint3 id : SV_DispatchThreadID){
    if(IsInBulk(id)){
        TYPE b_c=input_b[id.xy];
        TYPE x_xm=input_x[id.xy+int2(-1,0)];
        TYPE x_xp=input_x[id.xy+int2(1,0)];
        TYPE x_yp=input_x[id.xy+int2(0,1)];
        TYPE x_ym=input_x[id.xy+int2(0,-1)];
        output_x[id.xy]=(x_xp+x_xm+x_yp+x_ym)*jac_inv_beta+jac_alpha_inv_beta*b_c;
    }
    else if(IsOnBoundary(id))
        output_x[id.xy]=input_x[id.xy];
}

#pragma kernel UpdateNeumannBoundaryFloat TYPE=float UpdateNeumannBoundary=UpdateNeumannBoundaryFloat
// output_x

[numthreads(64,1,1)]
void UpdateNeumannBoundary(uint3 id_:SV_DispatchThreadID){
    uint id=id_.x;
    if(0<=id && id<gridCount.x){
        output_x[uint2(id,0)]=output_x[uint2(id,1)];
        output_x[uint2(id,gridCount.y-1)]=output_x[uint2(id,gridCount.y-2)];
    }
    if(0<=id && id<gridCount.y){
        output_x[uint2(0,id)]=output_x[uint2(1,id)];
        output_x[uint2(gridCount.x-1,id)]=output_x[uint2(gridCount.x-2,id)];
    }
}

#pragma kernel AdvectionFloat2 TYPE=float2 Advection=AdvectionFloat2
// input_x, output_x, input_vector

SamplerState linear_clamp_sampler;
[numthreads(8,8,1)]
void Advection(uint3 id : SV_DispatchThreadID){
    if(IsInBulk(id)){
        float2 uv=GetUV(id);
        float2 bias=-input_vector[id.xy]*timeStep/(cellSize*gridCount);
        output_x[id.xy]=input_x.SampleLevel(linear_clamp_sampler,uv+bias,0);
    }
    else if(IsOnBoundary(id))
        output_x[id.xy]=input_x[id.xy];
}

#pragma kernel AdvectionRefine TYPE=float2
// MacCormack method
// Seems not make a difference
// input_x=n+1hat output_x=n=>n+1
[numthreads(8,8,1)]
void AdvectionRefine(uint3 id : SV_DispatchThreadID){
    if(IsInBulkBoundary(id)){
        float2 velocity_ref;
        if(IsInBulk(id))
        {
            float2 uv=GetUV(id);
            float2 bias=input_x[id.xy]*timeStep/(cellSize*gridCount);//Here we reverse the advection
            velocity_ref=input_x.SampleLevel(linear_clamp_sampler,uv+bias,0);
        }else
            velocity_ref=input_x[id.xy];
        //here I used a slightly different method to save one copy of cache
        float2 minVal=min(input_x[id.xy],output_x[id.xy]);
        float2 maxVal=max(input_x[id.xy],output_x[id.xy]);
        output_x[id.xy]=input_x[id.xy]+.5*(output_x[id.xy]-velocity_ref);
        output_x[id.xy]=clamp(output_x[id.xy],minVal,maxVal);
        //TODO Add Limiter!
    }
}


#pragma kernel Divergence TYPE=float
// input_vector, output_x
#pragma kernel Curl TYPE=float
// input_vector, output_x
[numthreads(8,8,1)]
void Divergence(uint3 id: SV_DISPATCHTHREADID){
    if(IsInBulk(id)){
        float2 w_xm=input_vector[id.xy+int2(-1,0)];
        float2 w_xp=input_vector[id.xy+int2(1,0)];
        float2 w_yp=input_vector[id.xy+int2(0,1)];
        float2 w_ym=input_vector[id.xy+int2(0,-1)];
        output_x[id.xy]=(w_xp.x-w_xm.x+w_yp.y-w_ym.y)/(2*cellSize);
    }
}
[numthreads(8,8,1)]
void Curl(uint3 id:SV_DISPATCHTHREADID){
    if(IsInBulk(id)){
        float2 u_xm=input_vector[id.xy+int2(-1,0)];
        float2 u_xp=input_vector[id.xy+int2(1,0)];
        float2 u_yp=input_vector[id.xy+int2(0,1)];
        float2 u_ym=input_vector[id.xy+int2(0,-1)];
        output_x[id.xy]=(u_xp.y-u_xm.y-u_yp.x+u_ym.x)/(2*cellSize);
    }
}

//¨Œ@¨Œp=¨Œ@w
//u=w-¨Œp
#pragma kernel MinusGradient TYPE=float2
// input_scalar, output_x

[numthreads(8,8,1)]
void MinusGradient(uint3 id: SV_DISPATCHTHREADID){
    if(IsInBulk(id)){
        float p_xm=input_scalar[id.xy+int2(-1,0)];
        float p_xp=input_scalar[id.xy+int2(1,0)];
        float p_yp=input_scalar[id.xy+int2(0,1)];
        float p_ym=input_scalar[id.xy+int2(0,-1)];
        float2 grad=float2(p_xp-p_xm,p_yp-p_ym)/(2*cellSize);
        output_x[id.xy]=output_x[id.xy]-grad;
    }
    else if(IsOnBoundary(id))
        output_x[id.xy]=output_x[id.xy];
}

//¦Ø=¨Œ¡Áu
//u=u+eps dt dx normalize(¨Œ|¦Ø|)¡Á¦Ø

#pragma kernel AddVorticity TYPE=float2
// input_scalar, output_x
float vorticity_eps;
[numthreads(8,8,1)]
void AddVorticity(uint3 id: SV_DISPATCHTHREADID){
    if(IsInBulk(id)){
        float w_c=input_scalar[id.xy];
        float w_xm=input_scalar[id.xy+int2(-1,0)];
        float w_xp=input_scalar[id.xy+int2(1,0)];
        float w_yp=input_scalar[id.xy+int2(0,1)];
        float w_ym=input_scalar[id.xy+int2(0,-1)];
        float2 grad=float2(w_xp-w_xm,w_yp-w_ym)/(2*cellSize);
        if(length(grad)>0)
            output_x[id.xy]=output_x[id.xy]
                    +vorticity_eps*cellSize*timeStep
                    *normalize(float2(-grad.y,grad.x))*w_c;
    }
    else if(IsOnBoundary(id))
        output_x[id.xy]=output_x[id.xy];
}


#pragma kernel InitCondition TYPE=float2
// output_x
float initial_velocity;
[numthreads(8,8,1)]
void InitCondition(uint3 id: SV_DISPATCHTHREADID){
    if(IsInBulkBoundary(id)){
        float2 uv=GetUV(id);
        float2 uv1=(uv-.5)*2;
        {
            output_x[id.xy]=float2(uv1.y,-uv1.x);
            //output_x[id.xy]=float2(cos(uv1.y*1.6)+.01*cos(uv1.x*12+uv1.y*23),0);
            //output_x[id.xy]=float2(0,1);

        }


        output_x[id.xy]*=initial_velocity;
    }
}


//Fluid is incompressible, since we do not capture sound waves or shockwaves.
// Create a RenderTexture with enableRandomWrite flag and set it with cs.SetTexture
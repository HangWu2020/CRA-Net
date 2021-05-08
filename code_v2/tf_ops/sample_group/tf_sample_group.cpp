#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include <cuda_runtime.h>

using namespace tensorflow;

REGISTER_OP("FarthestPointSample")
  .Attr("npoint: int")
  .Attr("radius: float")
  .Attr("minnum: int")
  .Input("inp: float32")
  .Output("out: int32")
  .Output("cores: float32")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    int npoint;
    TF_RETURN_IF_ERROR(c->GetAttr("npoint", &npoint));
    ::tensorflow::shape_inference::ShapeHandle output = c->MakeShape({npoint});
    c->set_output(0, output);
    return Status::OK();
  });


void farthestpointsamplingLauncher(int b,int n,int m,float r,int minnum,const float * inp,float * temp,int * out,float * cores);
class FarthestPointSampleGpuOp: public OpKernel{
  public:
    explicit FarthestPointSampleGpuOp(OpKernelConstruction* context):OpKernel(context) {
                    OP_REQUIRES_OK(context, context->GetAttr("npoint", &npoint_));
                    OP_REQUIRES_OK(context, context->GetAttr("radius", &radius_));
                    OP_REQUIRES_OK(context, context->GetAttr("minnum", &minnum_));
                    OP_REQUIRES(context, npoint_ > 0, errors::InvalidArgument("FarthestPointSample expects positive npoint"));
                }
    void Compute(OpKernelContext * context)override{
      int m = npoint_;
      float r = radius_;
      int mn = minnum_;

      const Tensor& inp_tensor=context->input(0);
      OP_REQUIRES(context,inp_tensor.dims()==3 && inp_tensor.shape().dim_size(2)==3 && inp_tensor.shape().dim_size(0)==1,
                  errors::InvalidArgument("FarthestPointSample expects (1,num_points,3) inp shape"));
      
      int b=inp_tensor.shape().dim_size(0);
      int n=inp_tensor.shape().dim_size(1);
      auto inp_flat=inp_tensor.flat<float>();
      const float * inp=&(inp_flat(0));
      
      Tensor * out_tensor;
      OP_REQUIRES_OK(context,context->allocate_output(0,TensorShape{b,m},&out_tensor));      
      auto out_flat=out_tensor->flat<int>();
      int * out=&(out_flat(0));

      Tensor * cores_tensor;
      OP_REQUIRES_OK(context,context->allocate_output(1,TensorShape{b,m,3},&cores_tensor));      
      auto cores_flat=cores_tensor->flat<float>();
      float * cores=&(cores_flat(0));

      Tensor temp_tensor(DT_FLOAT, TensorShape{b,n});
      OP_REQUIRES_OK(context,context->allocate_temp(DT_FLOAT,TensorShape{b,n},&temp_tensor));
      auto temp_flat=temp_tensor.flat<float>();
      float * temp = temp_flat.data();

      farthestpointsamplingLauncher(b,n,m,r,mn,inp,temp,out,cores);
    }
  private:
      int npoint_;
      float radius_;
      int minnum_;

};
REGISTER_KERNEL_BUILDER(Name("FarthestPointSample").Device(DEVICE_GPU),FarthestPointSampleGpuOp);


REGISTER_OP("SampleGroup")
  .Attr("npoint: int")
  .Attr("radius: float")
  .Attr("minnum: int")
  .Input("inp: float32")
  .Output("cores: float32")
  .Output("radiuses: float32")
  .Output("local_region: float32")
  .Output("cnt: int32");

void samplegroupLauncher(int b,int n,int m,float r,int minnum,const float * dataset, float * temp, int * idxs,float * cores, float * dist, int * flag, 
                         float * radiuses, float * local_region, int * cnt);

class SampleGroupGpuOp: public OpKernel{
  public:
    explicit SampleGroupGpuOp(OpKernelConstruction* context):OpKernel(context) {
                    OP_REQUIRES_OK(context, context->GetAttr("npoint", &npoint_));
                    OP_REQUIRES_OK(context, context->GetAttr("radius", &radius_));
                    OP_REQUIRES_OK(context, context->GetAttr("minnum", &minnum_));
                    OP_REQUIRES(context, npoint_ > 0, errors::InvalidArgument("FarthestPointSample expects positive npoint"));
                }
    void Compute(OpKernelContext * context)override{
      int m = npoint_;
      float r = radius_;
      int mn = minnum_;

      const Tensor& inp_tensor=context->input(0);
      OP_REQUIRES(context,inp_tensor.dims()==3 && inp_tensor.shape().dim_size(2)==3 && inp_tensor.shape().dim_size(0)==1,
                  errors::InvalidArgument("FarthestPointSample expects (1,num_points,3) inp shape"));
      
      int b=inp_tensor.shape().dim_size(0);
      int n=inp_tensor.shape().dim_size(1);
      auto inp_flat=inp_tensor.flat<float>();
      const float * inp=&(inp_flat(0));
      
      //*** Temp vars ***//
      Tensor temp_tensor(DT_FLOAT, TensorShape({b,n}));
      OP_REQUIRES_OK(context,context->allocate_temp(DT_FLOAT,TensorShape{b,n},&temp_tensor));			
      auto temp_flat=temp_tensor.flat<float>();
      float * temp=temp_flat.data();
      
      Tensor out_tensor(DT_INT32, TensorShape({b,m}));
      OP_REQUIRES_OK(context,context->allocate_temp(DT_INT32,TensorShape{b,m},&out_tensor));
      auto out_flat=out_tensor.flat<int>();
      int * out=out_flat.data();
      
      Tensor dist_tensor(DT_FLOAT, TensorShape({b,n}));
      OP_REQUIRES_OK(context,context->allocate_temp(DT_FLOAT,TensorShape{b,n},&dist_tensor));
      auto dist_flat=dist_tensor.flat<float>();
      float * dist=dist_flat.data();
      
      Tensor flag_tensor(DT_INT32, TensorShape({b,n}));
      OP_REQUIRES_OK(context,context->allocate_temp(DT_INT32,TensorShape{b,n},&flag_tensor));
      auto flag_flat=flag_tensor.flat<int>();
      int * flag=flag_flat.data();

      //*** Output vars ***//
      Tensor * cores_tensor;
      OP_REQUIRES_OK(context,context->allocate_output(0,TensorShape{b,m,3},&cores_tensor));      
      auto cores_flat=cores_tensor->flat<float>();
      float * cores=&(cores_flat(0));

      Tensor * radiuses_tensor;
      OP_REQUIRES_OK(context,context->allocate_output(1,TensorShape{b,m},&radiuses_tensor));      
      auto radiuses_flat=radiuses_tensor->flat<float>();
      float * radiuses=&(radiuses_flat(0));

      Tensor * local_region_tensor;
      OP_REQUIRES_OK(context,context->allocate_output(2,TensorShape{b*m,1024,3},&local_region_tensor));      
      auto local_region_flat=local_region_tensor->flat<float>();
      float * local_region=&(local_region_flat(0));

      Tensor * cnt_tensor;
      OP_REQUIRES_OK(context,context->allocate_output(3,TensorShape{b*m},&cnt_tensor));      
      auto cnt_flat=cnt_tensor->flat<int>();
      int * cnt=&(cnt_flat(0));

      samplegroupLauncher (b,n,m,r,mn,inp,temp,out,cores,dist,flag,radiuses,local_region,cnt);

    }
  private:
      int npoint_;
      float radius_;
      int minnum_;
};
REGISTER_KERNEL_BUILDER(Name("SampleGroup").Device(DEVICE_GPU),SampleGroupGpuOp);

REGISTER_OP("FarthestPointSampleAll")
  .Attr("npoint: int")
  .Input("inp: float32")
  .Output("out: int32");


void farthestpointsamplingallLauncher(int b,int n,int m,const float * inp,float * temp,int * out);
class FarthestPointSampleAllGpuOp: public OpKernel{
  public:
    explicit FarthestPointSampleAllGpuOp(OpKernelConstruction* context):OpKernel(context) {
                    OP_REQUIRES_OK(context, context->GetAttr("npoint", &npoint_));
                    OP_REQUIRES(context, npoint_ > 0, errors::InvalidArgument("FarthestPointSample expects positive npoint"));
                }
    void Compute(OpKernelContext * context)override{
      int m = npoint_;

      const Tensor& inp_tensor=context->input(0);
      OP_REQUIRES(context,inp_tensor.dims()==3 && inp_tensor.shape().dim_size(2)==3,errors::InvalidArgument("FarthestPointSample expects (batch_size,num_points,3) inp shape"));
      int b=inp_tensor.shape().dim_size(0);
      int n=inp_tensor.shape().dim_size(1);
      auto inp_flat=inp_tensor.flat<float>();
      const float * inp=&(inp_flat(0));
      Tensor * out_tensor;
      OP_REQUIRES_OK(context,context->allocate_output(0,TensorShape{b,m},&out_tensor));
      auto out_flat=out_tensor->flat<int>();
      int * out=&(out_flat(0));
      Tensor temp_tensor;
      OP_REQUIRES_OK(context,context->allocate_temp(DataTypeToEnum<float>::value,TensorShape{32,n},&temp_tensor));
      auto temp_flat=temp_tensor.flat<float>();
      float * temp=&(temp_flat(0));
      farthestpointsamplingallLauncher(b,n,m,inp,temp,out);
    }
    private:
        int npoint_;
};
REGISTER_KERNEL_BUILDER(Name("FarthestPointSampleAll").Device(DEVICE_GPU),FarthestPointSampleAllGpuOp);




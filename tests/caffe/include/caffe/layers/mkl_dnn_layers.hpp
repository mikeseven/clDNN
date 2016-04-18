#ifndef CAFFE_MKL_DNN_LAYERS_HPP_
#define CAFFE_MKL_DNN_LAYERS_HPP_

#include <string>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layers/base_conv_layer.hpp"
#include "caffe/layers/neuron_layer.hpp"
#include "caffe/layers/conv_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "boost/enable_shared_from_this.hpp"

#include "neural.h"


namespace caffe {
using namespace neural;

template <typename Dtype, bool is_diff>
struct MKL_DNNMemoryDescriptor : PrvMemDescr, boost::enable_shared_from_this<MKL_DNNMemoryDescriptor<Dtype, is_diff> > {
  MKL_DNNMemoryDescriptor() : layout_usr(memory::format::bfyx_f32), name("UKNOWN") {};
  ~MKL_DNNMemoryDescriptor()
  {
    if(internal_ptr) CaffeFreeHost(internal_ptr, use_cuda);
  }

  shared_ptr<MKL_DNNMemoryDescriptor<Dtype, is_diff> > get_shared_ptr() {
    return this->shared_from_this();
  }

  int layout_usr = memory::format::bfyx_f32;
  int layout_int = memory::format::yxfb_f32;
  Dtype* internal_ptr;
  primitive memory = nullptr;
  primitive memory_usr = nullptr;
  primitive to_internal = nullptr;
  primitive from_internal = nullptr;
  std::string name;  // for debugging purposes
  bool use_cuda;
  void create_conversions() {
    if (layout_usr != layout_int)
    {
        CaffeMallocHost((void**)&internal_ptr, sizeof(Dtype)*prv_count(), &use_cuda);
        to_internal   = reorder::create(reorder::arguments({engine::reference, memory_usr, memory}));
        from_internal = reorder::create(reorder::arguments({engine::reference, memory, memory_usr}));
    }
  }

  virtual size_t prv_count() {return (memory.as<const neural::memory&>().count());};
  virtual void convert_from_prv(void* prv_ptr, void* cpu_ptr);
  virtual PrvDescrType get_descr_type() {return PRV_DESCR_MKL_DNN;};
  Dtype* get_converted_prv(Blob<Dtype>* blob, bool set_prv_ptr, MKL_DNNMemoryDescriptor<Dtype, is_diff>* converted_in_fwd=nullptr);
};

template <typename Dtype>
struct MKL_DNNData : MKL_DNNMemoryDescriptor<Dtype, false>
{};

template <typename Dtype>
struct MKL_DNNDiff : MKL_DNNMemoryDescriptor<Dtype, true>
{};

template <typename Dtype>
class MKL_DNNConvolutionLayer : public ConvolutionLayer<Dtype> {
public:
  explicit MKL_DNNConvolutionLayer(const LayerParameter& param, neural::engine::type engine = neural::engine::reference);

  virtual inline const char* type() const { return "DnnConvolution"; }
  virtual ~MKL_DNNConvolutionLayer();

protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  // Customized methods
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);
  virtual void compute_output_shape();

private:
  neural::engine::type engine_;
  primitive convolution_fwd = nullptr;
  primitive convolution_bwd = nullptr;
  /* Fwd step */
  shared_ptr<MKL_DNNData<Dtype> > fwd_bottom_data, fwd_top_data, fwd_filter_data, fwd_bias_data;

  /* Bwd data step */
  shared_ptr<MKL_DNNDiff<Dtype> > bwd_top_diff, bwd_bottom_diff;

  /* Bwd filter step */
  shared_ptr<MKL_DNNDiff<Dtype> > bwd_filter_diff;

  /* Bwd bias step */
  shared_ptr<MKL_DNNDiff<Dtype> > bwd_bias_diff;

 // TODO: temp. compatibility vs. older cafe
 size_t width_,
        height_,
        width_out_,
        height_out_,
        kernel_w_,
        kernel_h_,
        stride_w_,
        stride_h_;
 int    pad_w_,
        pad_h_;
};

#if 0 // not yet implemented except ReLU
/**
 * @brief Normalize the input in a local region across feature maps.
 */

template <typename Dtype>
class MKL_DNNLRNLayer : public Layer<Dtype> {
 public:
  explicit MKL_DNNLRNLayer(const LayerParameter& param)
      : Layer<Dtype>(param), layout_usr_(NULL) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual ~MKL_DNNLRNLayer();

  virtual inline const char* type() const { return "DnnLRN"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  virtual void CrossChannelForward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void CrossChannelBackward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  virtual void CrossChannelForward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void CrossChannelBackward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int size_;
  int pre_pad_;
  Dtype alpha_;
  Dtype beta_;
  Dtype k_;
  int num_;
  int channels_;
  int height_;
  int width_;
  // Fields used for normalization ACROSS_CHANNELS
  // scale_ stores the intermediate summing results
private:
  primitive lrnFwd, lrnBwd;
  shared_ptr<MKL_DNNData<Dtype> > fwd_top_data;
  shared_ptr<MKL_DNNDiff<Dtype> > bwd_bottom_diff;
  Dtype *lrn_buffer_;
  memory::format layout_usr_;
};



template <typename Dtype>
class MKL_DNNPoolingLayer : public Layer<Dtype> {
public:
  explicit MKL_DNNPoolingLayer(const LayerParameter& param)
    : Layer<Dtype>(param),
      fwd_top_data    (new MKL_DNNData<Dtype>()),
      fwd_bottom_data (new MKL_DNNData<Dtype>()),
      bwd_top_diff    (new MKL_DNNDiff<Dtype>()),
      bwd_bottom_diff (new MKL_DNNDiff<Dtype>()),
      poolingFwd(NULL), poolingBwd(NULL)
  {}
  ~MKL_DNNPoolingLayer();
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "DnnPooling"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int MinTopBlobs() const { return 1; }
  // MAX POOL layers can output an extra top blob for the mask;
  // others can only output the pooled inputs.
  virtual inline int MaxTopBlobs() const {
    return (this->layer_param_.pooling_param().pool() ==
            PoolingParameter_PoolMethod_MAX) ? 2 : 1;
  }

protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int kernel_h_, kernel_w_;
  int stride_h_, stride_w_;
  int pad_h_, pad_w_;
  int channels_;
  int height_, width_;
  int pooled_height_, pooled_width_;
  bool global_pooling_;
  Blob<Dtype> rand_idx_;
  Blob<size_t> max_idx_;
private:
  size_t kernel_size[2],
         kernel_stride[4];
  int src_offset[2];
  shared_ptr<MKL_DNNData<Dtype> > fwd_top_data, fwd_bottom_data;
  shared_ptr<MKL_DNNDiff<Dtype> > bwd_top_diff, bwd_bottom_diff;

  primitive poolingFwd, poolingBwd;
};
#endif

template <typename Dtype>
class MKL_DNNReLULayer : public NeuronLayer<Dtype> {
public:
  /**
   * @param param provides ReLUParameter relu_param,
   *     with ReLULayer options:
   *   - negative_slope (\b optional, default 0).
   *     the value @f$ \nu @f$ by which negative values are multiplied.
   */
  explicit MKL_DNNReLULayer(const LayerParameter& param)
    : NeuronLayer<Dtype>(param) {}
  ~MKL_DNNReLULayer();

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "MKL_DNNReLU"; }

protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

private:
  primitive reluFwd_ = nullptr, reluBwd_ = nullptr;
  primitive bottom_data_ = nullptr, top_data_ = nullptr, 
          bottom_diff_ = nullptr, top_diff_ = nullptr;
};

} // namespace caffe
#endif // #ifndef CAFFE_MKL_DNN_LAYERS_HPP_

#ifndef CAFFE_MKL_DNN_LAYERS_HPP_
#define CAFFE_MKL_DNN_LAYERS_HPP_

#include <string>
#include <vector>

#include "boost/enable_shared_from_this.hpp"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layers/base_conv_layer.hpp"
#include "caffe/layers/conv_layer.hpp"
#include "caffe/layers/neuron_layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "neural.h"

using neural::memory;
using neural::primitive;
using neural::reorder;
using neural::execute;
using neural::engine;

namespace caffe {

template <typename Dtype, bool is_diff>
struct MKL_DNNMemory : PrvMemDescr,
    boost::enable_shared_from_this<MKL_DNNMemory<Dtype, is_diff> > {
      MKL_DNNMemory(
        const memory::format::type layout_usr,
        const memory::format::type layout_prv,
        primitive memory_usr,
        primitive memory_prv) :
          layout_usr(layout_usr),
          layout_prv(layout_prv),
          memory_usr(memory_usr),
          memory_prv(memory_prv) {
      if (layout_usr != layout_prv) {
        if (layout_usr == memory::format::bfyx_f32
                && layout_prv == memory::format::byxf_f32)
            engine_to_prv_ = engine::cpu;

        to_prv   = reorder::create(reorder::arguments({engine_to_prv_,
          memory_prv, memory_usr}));
        from_prv = reorder::create(reorder::arguments({engine_from_prv_,
          memory_usr, memory_prv}));
      }
    }

  ~MKL_DNNMemory() {
    if (prv_ptr_) CaffeFreeHost(prv_ptr_, use_cuda);
  }

  shared_ptr<MKL_DNNMemory<Dtype, is_diff> > get_shared_ptr() {
    return this->shared_from_this();
  }

  memory::format::type layout_usr;
  memory::format::type layout_prv;
  primitive memory_usr = nullptr;
  primitive memory_prv = nullptr;
  primitive to_prv     = nullptr;
  primitive from_prv   = nullptr;

  // TODO: use the same engine as for the layers
  engine::type engine_to_prv_ = engine::reference;
  engine::type engine_from_prv_ = engine::reference;
  std::string name = "UNKNOWN";  // for debugging purposes
  bool use_cuda;

  virtual size_t prv_count() {
      return memory_prv.as<const neural::memory&>().count();
  }

  void allocate() {
    CHECK(prv_ptr_ == nullptr);
    CaffeMallocHost(reinterpret_cast<void**>(&prv_ptr_),
      sizeof(Dtype)*prv_count(), &use_cuda);
  }
  Dtype* prv_ptr() {
    if (prv_ptr_ == nullptr)
      allocate();

    return prv_ptr_;
  }
  virtual void convert_from_prv(void* prv_ptr, void* cpu_ptr);
  virtual PrvDescrType get_descr_type() {return PRV_DESCR_MKL_DNN;}
  Dtype* get_converted_prv(Blob<Dtype>* blob, bool set_prv_ptr,
          MKL_DNNMemory<Dtype, is_diff>* converted_in_fwd = nullptr);
 private:
  Dtype* prv_ptr_ = nullptr;
};

template <typename Dtype>
struct MKL_DNNData : MKL_DNNMemory<Dtype, false> {
    MKL_DNNData(
      const memory::format::type layout_usr,
      const memory::format::type layout_prv,
      primitive memory_usr,
      primitive memory_prv) :
        MKL_DNNMemory<Dtype, false>(
          layout_usr, layout_prv, memory_usr, memory_prv) {}
};

template <typename Dtype>
struct MKL_DNNDiff : MKL_DNNMemory<Dtype, true> {
    MKL_DNNDiff(
      const memory::format::type layout_usr,
      const memory::format::type layout_prv,
      primitive memory_usr,
      primitive memory_prv) :
        MKL_DNNMemory<Dtype, true>(
          layout_usr, layout_prv, memory_usr, memory_prv){}
};

template <typename Dtype>
class MKL_DNNConvolutionLayer : public ConvolutionLayer<Dtype> {
 public:
    explicit MKL_DNNConvolutionLayer(
      const LayerParameter& param, engine::type engine)
      : ConvolutionLayer<Dtype>(param), engine_(engine) {}

  virtual ~MKL_DNNConvolutionLayer();

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom);
  // Customized methods
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);
  virtual void compute_output_shape();

 private:
  engine::type engine_;

  memory::format::type prv_layout_in_out_;
  memory::format::type prv_layout_filter_;
  const memory::format::type usr_layout_in_out_ = memory::format::bfyx_f32;
  const memory::format::type usr_layout_filter_ = memory::format::oiyx_f32;
  const memory::format::type layout_bias_       = memory::format::x_f32;
  vector<primitive> convolution_fwd_;
  vector<primitive> filters_;
  vector<primitive> biases_;
  // TODO: use vector for convolution_bwd
  primitive convolution_bwd = nullptr;

  /* Fwd step */
  shared_ptr<MKL_DNNData<Dtype> > fwd_bottom_data, fwd_top_data,
                                  fwd_filter_data, fwd_bias_data;
  /* Bwd data step */
  shared_ptr<MKL_DNNDiff<Dtype> > bwd_top_diff, bwd_bottom_diff;

  /* Bwd filter step */
  shared_ptr<MKL_DNNDiff<Dtype> > bwd_filter_diff;

  /* Bwd bias step */
  shared_ptr<MKL_DNNDiff<Dtype> > bwd_bias_diff;

  // TODO: temp. compatibility vs. older cafe
  uint32_t width_,
           height_,
           width_out_,
           height_out_,
           kernel_w_,
           kernel_h_,
           stride_w_,
           stride_h_;
  int      pad_w_,
           pad_h_;
};

/**
 * @brief Normalize the input in a local region across feature maps.
 */

template <typename Dtype>
class MKL_DNNLRNLayer : public Layer<Dtype> {
 public:
  explicit MKL_DNNLRNLayer(const LayerParameter& param,
        engine::type engine)
      : Layer<Dtype>(param), engine_(engine) {}

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual ~MKL_DNNLRNLayer();

  virtual inline const char* type() const { return "LRN"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down,
      const vector<Blob<Dtype>*>& bottom);

  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down,
      const vector<Blob<Dtype>*>& bottom);

  virtual void CrossChannelForward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void CrossChannelBackward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  virtual void CrossChannelForward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void CrossChannelBackward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  uint32_t size_;
  int pre_pad_;
  Dtype alpha_;
  Dtype beta_;
  Dtype k_;
  uint32_t num_;
  uint32_t channels_;
  uint32_t height_;
  uint32_t width_;
  // Fields used for normalization ACROSS_CHANNELS
  // scale_ stores the intermediate summing results
 private:
  engine::type engine_;
  memory::format::type prv_layout_in_out_;
  const memory::format::type usr_layout_in_out_ = memory::format::bfyx_f32;
  primitive lrnFwd_ = nullptr, lrnBwd_ = nullptr;
  shared_ptr<MKL_DNNData<Dtype> > fwd_top_data_, fwd_bottom_data_;
  shared_ptr<MKL_DNNDiff<Dtype> > bwd_top_diff_, bwd_bottom_diff_;
};


template <typename Dtype>
class MKL_DNNPoolingLayer : public Layer<Dtype> {
 public:
  explicit MKL_DNNPoolingLayer(const LayerParameter& param,
          engine::type engine)
    : Layer<Dtype>(param), engine_(engine) {}

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Pooling"; }
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
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom);

  uint32_t kernel_h_, kernel_w_;
  uint32_t stride_h_, stride_w_;
  int pad_h_, pad_w_;
  int channels_;
  int height_, width_;
  int pooled_height_, pooled_width_;
  bool global_pooling_;
  Blob<Dtype> rand_idx_;
  Blob<size_t> max_idx_;

 private:
  engine::type engine_;
  memory::format::type prv_layout_in_out_;
  const memory::format::type usr_layout_in_out_ = memory::format::bfyx_f32;
  shared_ptr<MKL_DNNData<Dtype> > fwd_top_data_, fwd_bottom_data_;
  shared_ptr<MKL_DNNDiff<Dtype> > bwd_top_diff_, bwd_bottom_diff_;

  primitive poolingFwd_ = nullptr, poolingBwd_ = nullptr;
};


template <typename Dtype>
class MKL_DNNReLULayer : public NeuronLayer<Dtype> {
 public:
  /**
   * @param param provides ReLUParameter relu_param,
   *     with ReLULayer options:
   *   - negative_slope (\b optional, default 0).
   *     the value @f$ \nu @f$ by which negative values are multiplied.
   */
  explicit MKL_DNNReLULayer(const LayerParameter& param,
          engine::type engine)
    : NeuronLayer<Dtype>(param), engine_(engine) {}

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "ReLU"; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom);

 private:
  engine::type engine_;
  primitive reluFwd_ = nullptr, reluBwd_ = nullptr;
  primitive bottom_data_ = nullptr, top_data_ = nullptr,
            bottom_diff_ = nullptr, top_diff_ = nullptr;
};

/**
 * @brief Computes the softmax function.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class MKL_DNNSoftmaxLayer : public Layer<Dtype> {
 public:
  explicit MKL_DNNSoftmaxLayer(const LayerParameter& param,
          engine::type engine)
      : Layer<Dtype>(param), engine_(engine) {}

  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Softmax"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
     const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int outer_num_;
  int inner_num_;
  int softmax_axis_;
  /// sum_multiplier is used to carry out sum using BLAS
  Blob<Dtype> sum_multiplier_;
  /// scale is an intermediate Blob to hold temporary results.
  Blob<Dtype> scale_;

 private:
  engine::type engine_;
  memory::format::type prv_layout_in_out_;
  const memory::format::type usr_layout_in_out_ = memory::format::bx_f32;
  primitive softmaxFwd_ = nullptr, softmaxBwd_ = nullptr;
  shared_ptr<MKL_DNNData<Dtype> > top_data_, bottom_data_;
  shared_ptr<MKL_DNNDiff<Dtype> > top_diff_, bottom_diff_;
};

/**
 * @brief Also known as a "fully-connected" layer, computes an inner product
 *        with a set of learned weights, and (optionally) adds biases.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class MKL_DNNInnerProductLayer : public Layer<Dtype> {
 public:
  explicit MKL_DNNInnerProductLayer(const LayerParameter& param,
          engine::type engine)
      : Layer<Dtype>(param), engine_(engine) {}

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "InnerProduct"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int M_;
  int K_;
  int N_;
  bool bias_term_;
  Blob<Dtype> bias_multiplier_;
  bool transpose_;  ///< if true, assume transposed weights

 private:
  engine::type engine_;
  memory::format::type prv_layout_in_out_;
  memory::format::type prv_layout_weights_;
  memory::format::type usr_layout_in_out_  = memory::format::bx_f32;
  const memory::format::type usr_layout_weights_ = memory::format::oi_f32;
  const memory::format::type layout_bias_        = memory::format::x_f32;
  primitive fcFwd_ = nullptr, fcBwd_ = nullptr;
  shared_ptr<MKL_DNNData<Dtype> > top_data_, bottom_data_,
                                  weights_data_, bias_data_;
  shared_ptr<MKL_DNNDiff<Dtype> > top_diff_, bottom_diff_,
                                  weights_diff_, bias_diff_;

  primitive  bottom_data_xb_ = nullptr;
};

}  // namespace caffe
#endif  // #ifndef CAFFE_MKL_DNN_LAYERS_HPP_

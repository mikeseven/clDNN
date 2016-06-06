#ifdef MKL_DNN_ENABLED
#include <vector>

#include "boost/make_shared.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/mkl_dnn_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <> void MKL_DNNLRNLayer<double>::LayerSetUp(const vector<Blob<double>*>& bottom,
      const vector<Blob<double>*>& top) {NOT_IMPLEMENTED;}
template <> void MKL_DNNLRNLayer<double>::Forward_cpu(const vector<Blob<double>*>& bottom,
    const vector<Blob<double>*>& top) {NOT_IMPLEMENTED;}
template <> void MKL_DNNLRNLayer<double>::Backward_cpu(const vector<Blob<double>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<double>*>& bottom) {NOT_IMPLEMENTED;}
template <> void MKL_DNNLRNLayer<double>::CrossChannelForward_cpu(
    const vector<Blob<double>*>& bottom, const vector<Blob<double>*>& top)
{NOT_IMPLEMENTED;}
template <> void MKL_DNNLRNLayer<double>::CrossChannelBackward_cpu(
    const vector<Blob<double>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<double>*>& bottom) {NOT_IMPLEMENTED;}

template <typename Dtype>
MKL_DNNLRNLayer<Dtype>::~MKL_DNNLRNLayer() {}

template <typename Dtype>
void MKL_DNNLRNLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  size_ = this->layer_param_.lrn_param().local_size();
  CHECK_EQ(size_ % 2, 1) << "LRN only supports odd values for local_size";

  alpha_ = this->layer_param_.lrn_param().alpha();
  beta_ = this->layer_param_.lrn_param().beta();
  k_ = this->layer_param_.lrn_param().k();

  channels_ = bottom[0]->channels();
  height_   = bottom[0]->height();
  width_    = bottom[0]->width();
  num_      = bottom[0]->num();

  //std::cout << "n "  <<  num_ << "  c " << channels_  << "  w "  << width_  <<  "  h " << height_  << "\n";
  //std::cout << "size "  <<  size_ << "  a " << alpha_ << "  b "  << beta_  <<  "  k " << k_  << "\n";

  // Choose layout according to the engine
  switch (engine_) {
    case  neural::engine::cpu:
      prv_layout_in_out_ = memory::format::byxf_b24_f32;
    break;
    case neural::engine::reference:
      prv_layout_in_out_ = memory::format::yxfb_f32;
    break;
    default:
      CHECK(0) << "Wrong mkl-dnn engine";
  }

  fwd_bottom_data_ = boost::make_shared<MKL_DNNData<Dtype> >(
          usr_layout_in_out_, prv_layout_in_out_,
          memory::describe({engine_, usr_layout_in_out_, {num_, {width_, height_}, channels_ }}),
          memory::describe({engine_, prv_layout_in_out_, {num_, {width_, height_}, channels_ }}));

  fwd_top_data_ = boost::make_shared<MKL_DNNData<Dtype> >(
          usr_layout_in_out_, prv_layout_in_out_,
          memory::describe({engine_, usr_layout_in_out_, {num_, {width_, height_}, channels_ }}),
          memory::describe({engine_, prv_layout_in_out_, {num_, {width_, height_}, channels_ }}));

  bwd_bottom_diff_ = boost::make_shared<MKL_DNNDiff<Dtype> >(
          usr_layout_in_out_, prv_layout_in_out_,
          memory::describe({engine_, usr_layout_in_out_, {num_, {width_, height_}, channels_ }}),
          memory::describe({engine_, prv_layout_in_out_, {num_, {width_, height_}, channels_ }}));

  bwd_top_diff_ = boost::make_shared<MKL_DNNDiff<Dtype> >(
          usr_layout_in_out_, prv_layout_in_out_,
          memory::describe({engine_, usr_layout_in_out_, {num_, {width_, height_}, channels_ }}),
          memory::describe({engine_, prv_layout_in_out_, {num_, {width_, height_}, channels_ }}));


  // Names are for debugging only
  fwd_bottom_data_->name = "fwd_bottom_data   @ " + this->layer_param_.name();
  fwd_top_data_->name =    "fwd_top_data      @ " + this->layer_param_.name();
  bwd_top_diff_->name =    "bwd_top_diff      @ " + this->layer_param_.name();
  bwd_bottom_diff_->name = "bwd_bottom_diff   @ " + this->layer_param_.name();

  lrnFwd_ = normalization::response::create({engine_,
          fwd_top_data_->memory_prv,
          fwd_bottom_data_->memory_prv,
          size_, padding::zero, k_, alpha_/size_, beta_});

  //  TODO:
  lrnBwd_ = NULL;

}

template <typename Dtype>
void MKL_DNNLRNLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
      << "corresponding to (num, channels, height, width)";
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  num_ = bottom[0]->num();
  switch (this->layer_param_.lrn_param().norm_region()) {
  case LRNParameter_NormRegion_ACROSS_CHANNELS:
    top[0]->Reshape(num_, channels_, height_, width_);
    break;
  case LRNParameter_NormRegion_WITHIN_CHANNEL:
    NOT_IMPLEMENTED;
    break;
  default:
    LOG(FATAL) << "Unknown normalization region.";
  }
}

template <typename Dtype>
void MKL_DNNLRNLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  switch (this->layer_param_.lrn_param().norm_region()) {
  case LRNParameter_NormRegion_ACROSS_CHANNELS:
    CrossChannelForward_cpu(bottom, top);
    break;
  case LRNParameter_NormRegion_WITHIN_CHANNEL:
    NOT_IMPLEMENTED;
    break;
  default:
    LOG(FATAL) << "Unknown normalization region.";
  }
}

template <typename Dtype>
void MKL_DNNLRNLayer<Dtype>::CrossChannelForward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  auto bottom_data = fwd_bottom_data_->get_converted_prv(bottom[0], true);
  Dtype* top_data = nullptr;

  if (fwd_top_data_->from_prv != nullptr) {
    top_data = fwd_top_data_->prv_ptr();
    top[0]->set_prv_data(top_data , fwd_top_data_, false);
  } else {
    top_data = top[0]->mutable_cpu_data();
    DLOG(INFO) << "Using cpu_data for top in DnnPooling.";
  }

  execute({fwd_bottom_data_->memory_prv(bottom_data),
           fwd_top_data_->memory_prv(top_data),
           lrnFwd_}).wait();
}

template <typename Dtype>
void MKL_DNNLRNLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  switch (this->layer_param_.lrn_param().norm_region()) {
  case LRNParameter_NormRegion_ACROSS_CHANNELS:
    CrossChannelBackward_cpu(top, propagate_down, bottom);
    break;
  case LRNParameter_NormRegion_WITHIN_CHANNEL:
    NOT_IMPLEMENTED;
    break;
  default:
    LOG(FATAL) << "Unknown normalization region.";
  }
}

template <typename Dtype>
void MKL_DNNLRNLayer<Dtype>::CrossChannelBackward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  // TODO
  NOT_IMPLEMENTED;
#if 0
  const void* top_diff = reinterpret_cast<const void*>(top[0]->prv_diff());
  const void* bottom_data =
      reinterpret_cast<const void*>(bottom[0]->prv_data());
  void* bottom_diff = NULL;

  if (top_diff && bottom_data) {
    bottom_diff = reinterpret_cast<void*>(bottom[0]->mutable_prv_diff());
    // Is it the first pass? Create a primitive.
    bottom[0]->set_prv_descriptor_diff(bwd_bottom_diff_);

  } else {
    DLOG(INFO) << "Using cpu_data in MKL_DNNLRNLayer.";
    top_diff = reinterpret_cast<const void*>(top[0]->cpu_diff());
    bottom_data = reinterpret_cast<const void*>(bottom[0]->cpu_data());
    bottom_diff = reinterpret_cast<void*>(bottom[0]->mutable_cpu_diff());
  }
#endif
}


#ifdef CPU_ONLY
STUB_GPU(MKL_DNNLRNLayer);
STUB_GPU_FORWARD(MKL_DNNLRNLayer, CrossChannelForward);
STUB_GPU_BACKWARD(MKL_DNNLRNLayer, CrossChannelBackward);
#else
template <typename Dtype>
void MKL_DNNLRNLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {NOT_IMPLEMENTED;}
template <typename Dtype>
void MKL_DNNLRNLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
  {NOT_IMPLEMENTED;}
template <typename Dtype>
void MKL_DNNLRNLayer<Dtype>::CrossChannelForward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
  {NOT_IMPLEMENTED;}
template <typename Dtype>
void MKL_DNNLRNLayer<Dtype>::CrossChannelBackward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {NOT_IMPLEMENTED;}

#endif

INSTANTIATE_CLASS(MKL_DNNLRNLayer);
}  // namespace caffe
#endif  // #ifdef MKL_DNN_ENABLED

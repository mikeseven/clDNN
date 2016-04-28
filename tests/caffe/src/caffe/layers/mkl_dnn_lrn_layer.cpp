#ifdef USE_MKL_DNN
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layers/mkl_dnn_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
MKL_DNNLRNLayer<Dtype>::~MKL_DNNLRNLayer() {

  // ???
  //dnnReleaseBuffer<Dtype>(lrn_buffer_);

}

template <typename Dtype>
void MKL_DNNLRNLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  size_ = this->layer_param_.lrn_param().local_size();
  CHECK_EQ(size_ % 2, 1) << "LRN only supports odd values for local_size";

  alpha_ = this->layer_param_.lrn_param().alpha();
  beta_ = this->layer_param_.lrn_param().beta();
  k_ = this->layer_param_.lrn_param().k();

  size_t dim = 4, sizes[4], strides[4];

  channels_ = bottom[0]->channels();
  height_   = bottom[0]->height();
  width_    = bottom[0]->width();
  num_      = bottom[0]->num();

  sizes[0] = width_;
  sizes[1] = height_;
  sizes[2] = channels_;
  sizes[3] = num_;

  strides[0] = 1;
  strides[1] = sizes[0];
  strides[2] = sizes[0]*sizes[1];
  strides[3] = sizes[0]*sizes[1]*sizes[2];

  lrn_buffer_ = NULL;
  //e = dnnLayoutCreate<Dtype>(&layout_usr_, dim, sizes, strides);


  // "Lazy" allocation because here we don't know
  // what layout is used by neighbours.
  // TODO: can it be done here for
  lrnFwd_ = NULL;  // Will be allocated in a "lazy" way in first forward pass
  lrnBwd_ = NULL;  // Will be allocated in a "lazy" way in first backward pass
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
  const void* bottom_data =
    reinterpret_cast<const void*>(bottom[0]->prv_data());
  void* top_data = NULL;

  if (NULL != bottom_data) {
    // Is it the first pass? Create a primitive.
    if (lrnFwd_ == nullptr) {
      CHECK_EQ((bottom[0]->get_prv_descriptor_data())->get_descr_type(),
              PrvMemDescr::PRV_DESCR_MKL_DNN);
      shared_ptr<MKL_DNNData<Dtype> > mem_descr
        =  boost::static_pointer_cast<MKL_DNNData<Dtype> >
              (bottom[0]->get_prv_descriptor_data());
      CHECK(mem_descr != NULL);


      //dnnLayout_t lrn_buffer_l = NULL;

      //dnnLRNCreateForward<Dtype>(
        //      &lrnFwd, NULL, mem_descr->layout_int, size_, alpha_, beta_, k_);

      //e = dnnLayoutCreateFromPrimitive<Dtype>(
        //      &lrn_buffer_l, lrnFwd, dnnResourceWorkspace);

      //dnnAllocateBuffer<Dtype>(
        //      reinterpret_cast<void **>(&lrn_buffer_), lrn_buffer_l);

      fwd_top_data_ = mem_descr;
    }
    top_data = top[0]->mutable_prv_data();
    top[0]->set_prv_descriptor_data(fwd_top_data_);

  } else {
    DLOG(INFO) << "Using cpu_data in MKL_DNNLRNLayer.";
    if (lrnFwd_ == NULL) {
      // First pass
//      dnnLayout_t lrn_buffer_l = NULL;
      //e = dnnLRNCreateForward<Dtype>(
      //        &lrnFwd, NULL, layout_usr_, size_, alpha_, beta_, k_);
      //
      //e = dnnLayoutCreateFromPrimitive<Dtype>(
      //        &lrn_buffer_l, lrnFwd, dnnResourceWorkspace);
      //
      //e = dnnAllocateBuffer<Dtype>(
      //        reinterpret_cast<void **>(&lrn_buffer_), lrn_buffer_l);
      //
      //dnnLayoutDelete<Dtype>(lrn_buffer_l);
    }
    bottom_data = reinterpret_cast<const void*>(bottom[0]->cpu_data());
    top_data = top[0]->mutable_cpu_data();
  }


  //lrn_res[dnnResourceSrc] = const_cast<void*>(bottom_data);
  //lrn_res[dnnResourceDst] = top_data;
  //lrn_res[dnnResourceWorkspace] = lrn_buffer_;

  //e = dnnExecute<Dtype>(lrnFwd, lrn_res);

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
  const void* top_diff = reinterpret_cast<const void*>(top[0]->prv_diff());
  const void* bottom_data =
      reinterpret_cast<const void*>(bottom[0]->prv_data());
  void* bottom_diff = NULL;

  if (top_diff && bottom_data) {
    bottom_diff = reinterpret_cast<void*>(bottom[0]->mutable_prv_diff());
    // Is it the first pass? Create a primitive.
    if (lrnBwd_ == NULL) {
      CHECK_EQ((top[0]->get_prv_descriptor_diff())->get_descr_type(),
              PrvMemDescr::PRV_DESCR_MKL_DNN);
      shared_ptr<MKL_DNNDiff<Dtype> > mem_descr
        =  boost::static_pointer_cast<MKL_DNNDiff<Dtype> >
              (top[0]->get_prv_descriptor_diff());
      CHECK(mem_descr != NULL);

      //
      //e = dnnLRNCreateBackward<Dtype>(&lrnBwd, NULL, mem_descr->layout_int,
      //        mem_descr->layout_int, size_, alpha_, beta_, k_);
      //

      bwd_bottom_diff_ = mem_descr;
    }
    bottom[0]->set_prv_descriptor_diff(bwd_bottom_diff_);

  } else {
    DLOG(INFO) << "Using cpu_data in MKL_DNNLRNLayer.";
    top_diff = reinterpret_cast<const void*>(top[0]->cpu_diff());
    bottom_data = reinterpret_cast<const void*>(bottom[0]->cpu_data());
    bottom_diff = reinterpret_cast<void*>(bottom[0]->mutable_cpu_diff());
    if (lrnBwd_ == nullptr) {

//      e = dnnLRNCreateBackward<Dtype>(&lrnBwd, NULL, layout_usr_,
  //            layout_usr_, size_, alpha_, beta_, k_);

    }
  }

  //lrn_res[dnnResourceSrc] = const_cast<void *>(bottom_data);
  //lrn_res[dnnResourceDiffDst] = const_cast<void *>(top_diff);
  //lrn_res[dnnResourceDiffSrc] = bottom_diff;
  //lrn_res[dnnResourceWorkspace] = lrn_buffer_;
  //e = dnnExecute<Dtype>(lrnBwd, lrn_res);

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
#endif  // #ifdef USE_MKL_DNN

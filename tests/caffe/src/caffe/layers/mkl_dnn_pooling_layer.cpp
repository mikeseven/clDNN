#ifdef MKL_DNN_ENABLED
#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/mkl_dnn_layers.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"

/* TODO:
 * Average pooling
 * backward
 * using of max_idx_ / topmask
 */

namespace caffe {
template <> void MKL_DNNPoolingLayer<double>::LayerSetUp(const vector<Blob<double>*>& bottom,
      const vector<Blob<double>*>& top) {NOT_IMPLEMENTED;}
template <> void MKL_DNNPoolingLayer<double>::Forward_cpu(const vector<Blob<double>*>& bottom,
    const vector<Blob<double>*>& top) {NOT_IMPLEMENTED;}
template <> void MKL_DNNPoolingLayer<double>::Backward_cpu(const vector<Blob<double>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<double>*>& bottom) {NOT_IMPLEMENTED;}

template <typename Dtype>
void MKL_DNNPoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  PoolingParameter pool_param = this->layer_param_.pooling_param();

  if (pool_param.global_pooling()) {
    CHECK(!(pool_param.has_kernel_size() ||
      pool_param.has_kernel_h() || pool_param.has_kernel_w()))
      << "With Global_pooling: true Filter size cannot specified";
  } else {
    CHECK(!pool_param.has_kernel_size() !=
      !(pool_param.has_kernel_h() && pool_param.has_kernel_w()))
      << "Filter size is kernel_size OR kernel_h and kernel_w; not both";
    CHECK(pool_param.has_kernel_size() ||
      (pool_param.has_kernel_h() && pool_param.has_kernel_w()))
      << "For non-square filters both kernel_h and kernel_w are required.";
  }
  CHECK((!pool_param.has_pad() && pool_param.has_pad_h()
      && pool_param.has_pad_w())
      || (!pool_param.has_pad_h() && !pool_param.has_pad_w()))
      << "pad is pad OR pad_h and pad_w are required.";
  CHECK((!pool_param.has_stride() && pool_param.has_stride_h()
      && pool_param.has_stride_w())
      || (!pool_param.has_stride_h() && !pool_param.has_stride_w()))
      << "Stride is stride OR stride_h and stride_w are required.";
  global_pooling_ = pool_param.global_pooling();
  if (global_pooling_) {
    kernel_h_ = bottom[0]->height();
    kernel_w_ = bottom[0]->width();
  } else {
    if (pool_param.has_kernel_size()) {
      kernel_h_ = kernel_w_ = pool_param.kernel_size();
    } else {
      kernel_h_ = pool_param.kernel_h();
      kernel_w_ = pool_param.kernel_w();
    }
  }
  CHECK_GT(kernel_h_, 0) << "Filter dimensions cannot be zero.";
  CHECK_GT(kernel_w_, 0) << "Filter dimensions cannot be zero.";
  if (!pool_param.has_pad_h()) {
    pad_h_ = pad_w_ = pool_param.pad();
  } else {
    pad_h_ = pool_param.pad_h();
    pad_w_ = pool_param.pad_w();
  }
  if (!pool_param.has_stride_h()) {
    stride_h_ = stride_w_ = pool_param.stride();
  } else {
    stride_h_ = pool_param.stride_h();
    stride_w_ = pool_param.stride_w();
  }
  if (global_pooling_) {
    CHECK(pad_h_ == 0 && pad_w_ == 0 && stride_h_ == 1 && stride_w_ == 1)
      << "With Global_pooling: true; only pad = 0 and stride = 1";
  }
  if (pad_h_ != 0 || pad_w_ != 0) {
    CHECK(this->layer_param_.pooling_param().pool()
        == PoolingParameter_PoolMethod_AVE
        || this->layer_param_.pooling_param().pool()
        == PoolingParameter_PoolMethod_MAX)
        << "Padding implemented only for average and max pooling.";
    CHECK_LT(pad_h_, kernel_h_);
    CHECK_LT(pad_w_, kernel_w_);
  }

  pooled_height_ = static_cast<int>(ceil(static_cast<float>(
      bottom[0]->height() + 2 * pad_h_ - kernel_h_) / stride_h_)) + 1;
  pooled_width_ = static_cast<int>(ceil(static_cast<float>(
      bottom[0]->width() + 2 * pad_w_ - kernel_w_) / stride_w_)) + 1;
  if (pad_h_ || pad_w_) {
    // If we have padding, ensure that the last pooling starts strictly
    // inside the image (instead of at the padding); otherwise clip the last.
    if ((pooled_height_ - 1) * stride_h_ >= bottom[0]->height() + pad_h_) {
      --pooled_height_;
    }
    if ((pooled_width_ - 1) * stride_w_ >= bottom[0]->height() + pad_w_) {
      --pooled_width_;
    }
    CHECK_LT((pooled_height_ - 1) * stride_h_, bottom[0]->height() + pad_h_);
    CHECK_LT((pooled_width_ - 1) * stride_w_, bottom[0]->height() + pad_w_);
  }

  auto iw = bottom[0]->width();
  auto ih = bottom[0]->height();
  auto c  = bottom[0]->channels();
  auto n  = bottom[0]->num();
  auto ow = pooled_width_;
  auto oh = pooled_height_;

  auto in_off_b = 0;
  auto in_off_x = -pad_w_;
  auto in_off_y = -pad_h_;
  auto in_off_z = 0;
  auto out_off_b = 0;
  auto out_off_y = 0;
  auto out_off_x = 0;
  auto out_off_z = 0;

  //std::cout << "iw "  <<  iw << "  ih " << ih << "  c "  << c  <<  "  n " << n  << "\n";
  //std::cout << "ow "  <<  ow << "  oh " << oh << "  in_off_x "  << in_off_x  <<  "  in_off_y " << in_off_y  << "\n";

  fwd_bottom_data_->memory_usr = memory::create({engine_, fwd_bottom_data_->layout_usr, {n, {ih, iw}, c }});
  fwd_top_data_->memory_usr    = memory::create({engine_, fwd_top_data_   ->layout_usr, {n, {oh, ow}, c }});
  fwd_bottom_data_->memory_prv = memory::create({engine_, fwd_bottom_data_->layout_prv, {n, {ih, iw}, c }});
  fwd_top_data_->memory_prv    = memory::create({engine_, fwd_top_data_   ->layout_prv, {n, {oh, ow}, c }});

  bwd_bottom_diff_->memory_usr = memory::create({engine_, bwd_bottom_diff_->layout_usr, {n, {ih, iw}, c }});
  bwd_top_diff_->memory_usr    = memory::create({engine_, bwd_top_diff_   ->layout_usr, {n, {oh, ow}, c }});
  bwd_bottom_diff_->memory_prv = memory::create({engine_, bwd_bottom_diff_->layout_prv, {n, {ih, iw}, c }});
  bwd_top_diff_->memory_prv    = memory::create({engine_, bwd_top_diff_   ->layout_prv, {n, {oh, ow}, c }});


  // Names are for debugging only
  fwd_bottom_data_->name = "fwd_bottom_data   @ " + this->layer_param_.name();
  fwd_top_data_->name =    "fwd_top_data      @ " + this->layer_param_.name();
  bwd_top_diff_->name =    "bwd_top_diff      @ " + this->layer_param_.name();
  bwd_bottom_diff_->name = "bwd_bottom_diff   @ " + this->layer_param_.name();

  fwd_bottom_data_->create_conversions();
  fwd_top_data_   ->create_conversions();
  bwd_top_diff_   ->create_conversions();
  bwd_bottom_diff_->create_conversions();

  pooling::mode::type mode;
  switch (this->layer_param_.pooling_param().pool()) {
    case PoolingParameter_PoolMethod_MAX:
      mode = pooling::mode::max;
      break;
    case PoolingParameter_PoolMethod_AVE:
      mode = pooling::mode::average;
      break;
    case PoolingParameter_PoolMethod_STOCHASTIC:
    default:
      NOT_IMPLEMENTED;
  }
  poolingFwd_ = pooling::create( {engine::reference,
                                    mode,
                                    fwd_top_data_->memory_prv,
                                    {out_off_b, {out_off_y, out_off_x}, out_off_z},
                                    {n,         {oh,        ow},        c},
                                    fwd_bottom_data_->memory_prv,
                                    {in_off_b,  {in_off_y, in_off_x},   in_off_z},
                                    {1,         {stride_h_, stride_w_}, 1},
                                    {1,         {kernel_h_, kernel_w_}, 1},
                                    padding::zero}
                                  );

  poolingBwd_ = NULL;
}

template <typename Dtype>
void MKL_DNNPoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
      << "corresponding to (num, channels, height, width)";
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  if (global_pooling_) {
    kernel_h_ = bottom[0]->height();
    kernel_w_ = bottom[0]->width();
  }
  pooled_height_ = static_cast<int>(ceil(static_cast<float>(
      height_ + 2 * pad_h_ - kernel_h_) / stride_h_)) + 1;
  pooled_width_ = static_cast<int>(ceil(static_cast<float>(
      width_ + 2 * pad_w_ - kernel_w_) / stride_w_)) + 1;
  if (pad_h_ || pad_w_) {
    // If we have padding, ensure that the last pooling starts strictly
    // inside the image (instead of at the padding); otherwise clip the last.
    if ((pooled_height_ - 1) * stride_h_ >= height_ + pad_h_) {
      --pooled_height_;
    }
    if ((pooled_width_ - 1) * stride_w_ >= width_ + pad_w_) {
      --pooled_width_;
    }
    CHECK_LT((pooled_height_ - 1) * stride_h_, height_ + pad_h_);
    CHECK_LT((pooled_width_ - 1) * stride_w_, width_ + pad_w_);
  }
  top[0]->Reshape(bottom[0]->num(), channels_, pooled_height_,
      pooled_width_);
  if (top.size() > 1) {
    (reinterpret_cast<Blob<size_t>* > (top[1]) )->Reshape(bottom[0]->num(),
            channels_, pooled_height_, pooled_width_);
  }
  // If max pooling, we will initialize the vector index part.
  if (this->layer_param_.pooling_param().pool() ==
      PoolingParameter_PoolMethod_MAX && top.size() == 1) {
    max_idx_.Reshape(bottom[0]->num(), channels_, pooled_height_,
            pooled_width_);
  }
  // If stochastic pooling, we will initialize the random index part.
  if (this->layer_param_.pooling_param().pool() ==
      PoolingParameter_PoolMethod_STOCHASTIC) {
    rand_idx_.Reshape(bottom[0]->num(), channels_, pooled_height_,
      pooled_width_);
  }
}


// TODO(Yangqing): Is there a faster way to do pooling in the channel-first
// case?
template <typename Dtype>
void MKL_DNNPoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // printf(" len(top_data) = %i\n", sizeof(top_data)/sizeof(Dtype));
  const int top_count = top[0]->count();
  // We'll output the mask to top[1] if it's of size >1.
  size_t* mask = NULL;  // suppress warnings about uninitalized variables

  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;

  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:
    {
    mask = (use_top_mask) ?
      reinterpret_cast<size_t*>(top[1]->mutable_cpu_data()) :
      (max_idx_.mutable_cpu_data());

    caffe_set<size_t>(top_count, -1, mask);
    //pooling_res[dnnResourceWorkspace] = reinterpret_cast<void*>(mask);

    auto bottom_data = fwd_bottom_data_->get_converted_prv(bottom[0], true);
    void *top_data = nullptr;
    if (fwd_top_data_->from_prv != nullptr) {
      top[0]->set_prv_data(fwd_top_data_->prv_ptr, fwd_top_data_, false);
      top_data = fwd_top_data_->prv_ptr;
    } else {
      top_data = top[0]->mutable_cpu_data();
      DLOG(INFO) << "Using cpu_data for top in DnnPooling.";
    }

    execute({fwd_bottom_data_->memory_prv(bottom_data),
             fwd_top_data_->memory_prv(top_data),
             poolingFwd_});

    }
    break;
  case PoolingParameter_PoolMethod_AVE:
  {
    auto bottom_data = fwd_bottom_data_->get_converted_prv(bottom[0], true);
    void *top_data = nullptr;
    if (fwd_top_data_->from_prv != nullptr) {
      top[0]->set_prv_data(fwd_top_data_->prv_ptr, fwd_top_data_, false);
      top_data = fwd_top_data_->prv_ptr;
    } else {
      top_data = top[0]->mutable_cpu_data();
      DLOG(INFO) << "Using cpu_data for top in DnnPooling.";
    }

    execute({fwd_bottom_data_->memory_prv(bottom_data),
             fwd_top_data_->memory_prv(top_data),
             poolingFwd_});
  }
    break;
  case PoolingParameter_PoolMethod_STOCHASTIC:
    NOT_IMPLEMENTED;
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
}

template <typename Dtype>
void MKL_DNNPoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  NOT_IMPLEMENTED;
#if 0
  // Different pooling methods. We explicitly do the switch outside the for
  // loop to save time, although this results in more codes.

  const size_t* mask = NULL;  // suppress warnings about uninitialized variables

  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:
    // The main loop
    mask = (top.size() > 1) ?
      reinterpret_cast<const size_t*>(top[1]->cpu_data()) :
      (max_idx_.cpu_data());

//    pooling_res[dnnResourceWorkspace] =
  //          reinterpret_cast<void *>(const_cast<size_t*>(mask));
   // pooling_res[dnnResourceDiffDst] = bwd_top_diff->get_converted_prv(top[0],
     ///       true);

    if (bwd_bottom_diff_->from_prv != nullptr) {
      bottom[0]->set_prv_diff(bwd_bottom_diff_->prv_ptr, bwd_bottom_diff_,
              false);
      //pooling_res[dnnResourceDiffSrc] =
        //      reinterpret_cast<void *>(bwd_bottom_diff->prv_ptr);
    } else {
      //pooling_res[dnnResourceDiffSrc] = bottom[0]->mutable_cpu_diff();
    }
    //caffe_set(bottom[0]->count(), Dtype(0),
      //      reinterpret_cast<Dtype *>(pooling_res[dnnResourceDiffSrc]));

    //e = dnnExecute<Dtype>(poolingBwd, pooling_res);


    break;
  case PoolingParameter_PoolMethod_AVE:
    NOT_IMPLEMENTED;
    break;
  case PoolingParameter_PoolMethod_STOCHASTIC:
    NOT_IMPLEMENTED;
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }

#endif  // #if 0
}


#ifdef CPU_ONLY
STUB_GPU(MKL_DNNPoolingLayer);
#else
template <typename Dtype>
void MKL_DNNPoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {NOT_IMPLEMENTED;}
template <typename Dtype>
void MKL_DNNPoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
  {NOT_IMPLEMENTED;}
#endif

INSTANTIATE_CLASS(MKL_DNNPoolingLayer);
}  // namespace caffe
#endif  // #ifdef MKL_DNN_ENABLED

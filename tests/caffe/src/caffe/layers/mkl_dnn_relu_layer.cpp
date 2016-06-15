#ifdef MKL_DNN_ENABLED
#include <algorithm>
#include <vector>

#include "caffe/layers/mkl_dnn_layers.hpp"
#include "neural.h"

using neural::relu;
using neural::relu_backward;

namespace caffe {
template <> void MKL_DNNReLULayer<double>::LayerSetUp(
      const vector<Blob<double>*>& bottom,
      const vector<Blob<double>*>& top) {NOT_IMPLEMENTED;}
template <> void MKL_DNNReLULayer<double>::Forward_cpu(
    const vector<Blob<double>*>& bottom,
    const vector<Blob<double>*>& top) {NOT_IMPLEMENTED;}
template <> void MKL_DNNReLULayer<double>::Backward_cpu(
    const vector<Blob<double>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<double>*>& bottom) {NOT_IMPLEMENTED;}

template <typename Dtype>
void MKL_DNNReLULayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  Dtype negative_slope = this->layer_param_.relu_param().negative_slope();

  const uint32_t count = bottom[0]->count();
  bottom_data_ = memory::describe({engine_, memory::format::yxfb_f32,
    {count, {1, 1}, 1}});
  top_data_    = memory::describe({engine_, memory::format::yxfb_f32,
    {count, {1, 1}, 1}});
  bottom_diff_ = memory::describe({engine_, memory::format::yxfb_f32,
    {count, {1, 1}, 1}});
  top_diff_    = memory::describe({engine_, memory::format::yxfb_f32,
    {count, {1, 1}, 1}});

  reluFwd_ = relu::create({engine_, top_data_, bottom_data_, negative_slope});
  reluBwd_ = relu_backward::create({engine_, {bottom_diff_},
    {top_diff_, bottom_data_}, negative_slope});
}

template <typename Dtype>
void MKL_DNNReLULayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Dtype* bottom_data = const_cast<Dtype*>(bottom[0]->prv_data());
  Dtype* top_data = NULL;

  if (bottom_data) {
    top_data = top[0]->mutable_prv_data();
  } else {
    DLOG(INFO) << "Using cpu_data in MKL_DNNReLULayer.";
    bottom_data = const_cast<Dtype*>(bottom[0]->cpu_data());
    top_data = top[0]->mutable_cpu_data();
  }

  execute({bottom_data_(bottom_data), top_data_(top_data), reluFwd_}).wait();
}

template <typename Dtype>
void MKL_DNNReLULayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

  if (propagate_down[0]) {
    Dtype* top_diff = const_cast<Dtype*>(top[0]->prv_diff());
    Dtype* bottom_data = const_cast<Dtype*>(bottom[0]->prv_data());
    Dtype* bottom_diff = NULL;

    if (top_diff && bottom_data) {
      bottom_diff = bottom[0]->mutable_prv_diff();
    } else {
      DLOG(INFO) << "Using cpu_data in MKL_DNNReLULayer.";
      top_diff = const_cast<Dtype*>(top[0]->cpu_diff());
      bottom_data = const_cast<Dtype*>(bottom[0]->cpu_data());
      bottom_diff = bottom[0]->mutable_cpu_diff();
    }

    execute({ top_diff_(top_diff), bottom_data_(bottom_data),
              bottom_diff_(bottom_diff), reluBwd_}).wait();
  }
}

#ifdef CPU_ONLY
STUB_GPU(MKL_DNNReLULayer);
#else
template <typename Dtype>
void MKL_DNNReLULayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
  {NOT_IMPLEMENTED;}
template <typename Dtype>
void MKL_DNNReLULayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom)
  {NOT_IMPLEMENTED;}
#endif

INSTANTIATE_CLASS(MKL_DNNReLULayer);
}  // namespace caffe
#endif  // #ifdef MKL_DNN_ENABLED

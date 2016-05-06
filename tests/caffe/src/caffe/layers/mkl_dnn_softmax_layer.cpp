#ifdef MKL_DNN_ENABLED
#include <algorithm>
#include <vector>
#include <iostream>

#include "caffe/layers/softmax_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/mkl_dnn_layers.hpp"

namespace caffe {
template <> void MKL_DNNSoftmaxLayer<double>::LayerSetUp(const vector<Blob<double>*>& bottom,
      const vector<Blob<double>*>& top) {NOT_IMPLEMENTED;}
template <> void MKL_DNNSoftmaxLayer<double>::Forward_cpu(const vector<Blob<double>*>& bottom,
    const vector<Blob<double>*>& top) {NOT_IMPLEMENTED;}
template <> void MKL_DNNSoftmaxLayer<double>::Backward_cpu(const vector<Blob<double>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<double>*>& bottom) {NOT_IMPLEMENTED;}

template <typename Dtype>
void MKL_DNNSoftmaxLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  Dtype negative_slope = this->layer_param_.relu_param().negative_slope();

  auto batch =  bottom[0]->shape(0);
  auto input_x = bottom[0]->shape(1);
  auto output_x = input_x;
  const auto z = 1;

  bottom_data_->memory_prv = memory::create({engine_, bottom_data_->layout_prv, {batch, {{input_x}}, z}});
  top_data_   ->memory_prv = memory::create({engine_, top_data_   ->layout_prv, {batch, {{input_x}}, z}});
  bottom_diff_->memory_prv = memory::create({engine_, bottom_diff_->layout_prv, {batch, {{input_x}}, z}});
  top_diff_   ->memory_prv = memory::create({engine_, top_diff_   ->layout_prv, {batch, {{input_x}}, z}});

  bottom_data_->memory_usr = memory::create({engine_, bottom_data_->layout_usr, {batch, {{input_x}}, z}});
  top_data_   ->memory_usr = memory::create({engine_, top_data_   ->layout_usr, {batch, {{input_x}}, z}});
  bottom_diff_->memory_usr = memory::create({engine_, bottom_diff_->layout_usr, {batch, {{input_x}}, z}});
  top_diff_   ->memory_usr = memory::create({engine_, top_diff_   ->layout_usr, {batch, {{input_x}}, z}});

  // Names are for debugging only
  bottom_data_->name = "fwd_bottom_data   @ " + this->layer_param_.name();
  top_data_->name =    "fwd_top_data      @ " + this->layer_param_.name();
  top_diff_->name =    "bwd_top_diff      @ " + this->layer_param_.name();
  bottom_diff_->name = "bwd_bottom_diff   @ " + this->layer_param_.name();

  bottom_data_->create_conversions();
  top_data_   ->create_conversions();
  bottom_diff_->create_conversions();
  top_diff_   ->create_conversions();
  softmaxFwd_ = normalization::softmax::create({engine_,
                                                top_data_->memory_prv,
                                                {0, {{0}}, 0},
                                                {batch, {{output_x}}, 1u},
                                                bottom_data_->memory_prv,
                                                {0, {{0}}, 0},
                                                });
  // TODO: softmax support in mkl-dnn
  //softmaxBwd_ = normalization::softmax_backward::create({engine::reference, {bottom_diff_}, {top_diff_, bottom_data_}, negative_slope});
}


template <typename Dtype>
void MKL_DNNSoftmaxLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  softmax_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.softmax_param().axis());
  top[0]->ReshapeLike(*bottom[0]);
  vector<int> mult_dims(1, bottom[0]->shape(softmax_axis_));
  sum_multiplier_.Reshape(mult_dims);
  Dtype* multiplier_data = sum_multiplier_.mutable_cpu_data();
  caffe_set(sum_multiplier_.count(), Dtype(1), multiplier_data);
  outer_num_ = bottom[0]->count(0, softmax_axis_);
  inner_num_ = bottom[0]->count(softmax_axis_ + 1);
  vector<int> scale_dims = bottom[0]->shape();
  scale_dims[softmax_axis_] = 1;
  scale_.Reshape(scale_dims);
}

template <typename Dtype>
void MKL_DNNSoftmaxLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  auto bottom_data = bottom_data_->get_converted_prv(bottom[0], true);
  void *top_data = nullptr;
  if (top_data_->from_prv != nullptr) {
    top[0]->set_prv_data(top_data_->prv_ptr, top_data_, false);
    top_data = top_data_->prv_ptr;
  } else {
    top_data = top[0]->mutable_cpu_data();
    DLOG(INFO) << "Using cpu_data for top in DnnPooling.";
  }
  execute({bottom_data_->memory_prv(bottom_data), top_data_->memory_prv(top_data), softmaxFwd_});
}

template <typename Dtype>
void MKL_DNNSoftmaxLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
  /*
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* top_data = top[0]->cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  Dtype* scale_data = scale_.mutable_cpu_data();
  int channels = top[0]->shape(softmax_axis_);
  int dim = top[0]->count() / outer_num_;
  caffe_copy(top[0]->count(), top_diff, bottom_diff);
  for (int i = 0; i < outer_num_; ++i) {
    // compute dot(top_diff, top_data) and subtract them from the bottom diff
    for (int k = 0; k < inner_num_; ++k) {
      scale_data[k] = caffe_cpu_strided_dot<Dtype>(channels,
          bottom_diff + i * dim + k, inner_num_,
          top_data + i * dim + k, inner_num_);
    }
    // subtraction
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels, inner_num_, 1,
        -1., sum_multiplier_.cpu_data(), scale_data, 1., bottom_diff + i * dim);
  }
  // elementwise multiplication
  caffe_mul(top[0]->count(), bottom_diff, top_data, bottom_diff);

   */
}


#ifdef CPU_ONLY
STUB_GPU(MKL_DNNSoftmaxLayer);
#else
template <typename Dtype>
void MKL_DNNSoftmaxLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
  {NOT_IMPLEMENTED;}
template <typename Dtype>
void MKL_DNNSoftmaxLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom)
  {NOT_IMPLEMENTED;}
#endif


INSTANTIATE_CLASS(MKL_DNNSoftmaxLayer);

}  // namespace caffe
#endif  // #ifdef MKL_DNN_ENABLED
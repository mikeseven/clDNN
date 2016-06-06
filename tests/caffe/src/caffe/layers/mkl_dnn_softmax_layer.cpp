#ifdef MKL_DNN_ENABLED
#include <algorithm>
#include <vector>
#include <iostream>

#include "boost/make_shared.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/mkl_dnn_layers.hpp"

// TODO: 4D input ???

namespace caffe {
template <> void MKL_DNNSoftmaxLayer<double>::LayerSetUp(
  const vector<Blob<double>*>& bottom,
  const vector<Blob<double>*>& top) {NOT_IMPLEMENTED;}
template <> void MKL_DNNSoftmaxLayer<double>::Forward_cpu(
  const vector<Blob<double>*>& bottom,
  const vector<Blob<double>*>& top) {NOT_IMPLEMENTED;}
template <> void MKL_DNNSoftmaxLayer<double>::Backward_cpu(
  const vector<Blob<double>*>& top,
  const vector<bool>& propagate_down,
  const vector<Blob<double>*>& bottom) {NOT_IMPLEMENTED;}

template <typename Dtype>
void MKL_DNNSoftmaxLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  uint32_t batch =  bottom[0]->shape(0);
  uint32_t input_x = bottom[0]->shape(1);
  uint32_t output_x = input_x;
  const uint32_t z = 1;

  // Choose layout according to the engine
  switch (engine_) {
    case  neural::engine::cpu:
      prv_layout_in_out_ = memory::format::xb_f32;
    break;
    case neural::engine::reference:
      prv_layout_in_out_ = memory::format::xb_f32;
    break;
    default:
      CHECK(0) << "Wrong mkl-dnn engine";
  }

  bottom_data_ = boost::make_shared<MKL_DNNData<Dtype> >(
          usr_layout_in_out_, prv_layout_in_out_,
          memory::describe({engine_, usr_layout_in_out_, {batch, {{input_x}}, z}}),
          memory::describe({engine_, prv_layout_in_out_, {batch, {{input_x}}, z}}));

  top_data_ = boost::make_shared<MKL_DNNData<Dtype> >(
          usr_layout_in_out_, prv_layout_in_out_,
          memory::describe({engine_, usr_layout_in_out_, {batch, {{input_x}}, z}}),
          memory::describe({engine_, prv_layout_in_out_, {batch, {{input_x}}, z}}));

  bottom_diff_ = boost::make_shared<MKL_DNNDiff<Dtype> >(
          usr_layout_in_out_, prv_layout_in_out_,
          memory::describe({engine_, usr_layout_in_out_, {batch, {{input_x}}, z}}),
          memory::describe({engine_, prv_layout_in_out_, {batch, {{input_x}}, z}}));

  top_diff_ = boost::make_shared<MKL_DNNDiff<Dtype> >(
          usr_layout_in_out_, prv_layout_in_out_,
          memory::describe({engine_, usr_layout_in_out_, {batch, {{input_x}}, z}}),
          memory::describe({engine_, prv_layout_in_out_, {batch, {{input_x}}, z}}));

  // Names are for debugging only
  bottom_data_->name = "fwd_bottom_data   @ " + this->layer_param_.name() + " ";
  top_data_->name =    "fwd_top_data      @ " + this->layer_param_.name() + " ";
  top_diff_->name =    "bwd_top_diff      @ " + this->layer_param_.name() + " ";
  bottom_diff_->name = "bwd_bottom_diff   @ " + this->layer_param_.name() + " ";

  softmaxFwd_ = normalization::softmax::create({engine_,
                                                top_data_->memory_prv,
                                                {0, {{0}}, 0},
                                                {batch, {{output_x}}, 1u},
                                                bottom_data_->memory_prv,
                                                {0, {{0}}, 0},
                                                });
  // TODO: softmax backward support in mkl-dnn
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
  Dtype *top_data = nullptr;
  if (top_data_->from_prv != nullptr) {
    top_data = top_data_->prv_ptr();
    top[0]->set_prv_data(top_data, top_data_, false);
  } else {
    top_data = top[0]->mutable_cpu_data();
    DLOG(INFO) << "Using cpu_data for top in DnnPooling.";
  }
  execute({bottom_data_->memory_prv(bottom_data), top_data_->memory_prv(top_data), softmaxFwd_}).wait();
}

template <typename Dtype>
void MKL_DNNSoftmaxLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
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
#ifdef USE_MKL_DNN
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/inner_product_layer.hpp"
#include "caffe/layers/mkl_dnn_layers.hpp"
#include "caffe/util/math_functions.hpp"

/* TODO
 *
 * support for fc with and without bias term ?
 *
 */

namespace caffe {

template <> void MKL_DNNInnerProductLayer<double>::LayerSetUp(const vector<Blob<double>*>& bottom,
      const vector<Blob<double>*>& top) {NOT_IMPLEMENTED;}
template <> void MKL_DNNInnerProductLayer<double>::Forward_cpu(const vector<Blob<double>*>& bottom,
    const vector<Blob<double>*>& top) {NOT_IMPLEMENTED;}
template <> void MKL_DNNInnerProductLayer<double>::Backward_cpu(const vector<Blob<double>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<double>*>& bottom) {NOT_IMPLEMENTED;}

template <typename Dtype>
void MKL_DNNInnerProductLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int num_output = this->layer_param_.inner_product_param().num_output();
  bias_term_ = this->layer_param_.inner_product_param().bias_term();
  // TODO: is it recently added parameter???
  // transpose_ = this->layer_param_.inner_product_param().transpose();
  N_ = num_output;
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.inner_product_param().axis());
  // Dimensions starting from "axis" are "flattened" into a single
  // length K_ vector. For example, if bottom[0]'s shape is (N, C, H, W),
  // and axis == 1, N inner products with dimension CHW are performed.
  K_ = bottom[0]->count(axis);
  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Initialize the weights
    vector<int> weight_shape(2);
    if (transpose_) {
      weight_shape[0] = K_;
      weight_shape[1] = N_;
    } else {
      weight_shape[0] = N_;
      weight_shape[1] = K_;
    }
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    // fill the weights
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.inner_product_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, intiialize and fill the bias term
    if (bias_term_) {
      vector<int> bias_shape(1, N_);
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.inner_product_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);


  auto batch = bottom[0]->count(0, axis); // TODO: is this ok?
  auto input_x = K_;
  auto bias_x = N_, output_x = N_;

  std::cout << "input_x: " << input_x << "  output_x: "  << output_x <<  " batch: " << batch <<" \n";
  /* MKL-DNN setup */
  bottom_data_->memory_prv = memory::create({engine_, bottom_data_->layout_int, {batch, {{input_x}},  1}});
  top_data_   ->memory_prv = memory::create({engine_, top_data_   ->layout_int, {batch, {{output_x}}, 1}});
  bottom_diff_->memory_prv = memory::create({engine_, bottom_diff_->layout_int, {batch, {{input_x}},  1}});
  top_diff_   ->memory_prv = memory::create({engine_, top_diff_   ->layout_int, {batch, {{output_x}}, 1}});

  bottom_data_->memory_usr = memory::create({engine_, bottom_data_->layout_usr, {batch, {{input_x}},  1}});
  top_data_   ->memory_usr = memory::create({engine_, top_data_   ->layout_usr, {batch, {{output_x}}, 1}});
  bottom_diff_->memory_usr = memory::create({engine_, bottom_diff_->layout_usr, {batch, {{input_x}},  1}});
  top_diff_   ->memory_usr = memory::create({engine_, top_diff_   ->layout_usr, {batch, {{output_x}}, 1}});

  weights_data_->memory_prv = memory::create({engine_, bottom_data_->layout_int, {input_x, {{output_x}}, 1}});
  bias_data_   ->memory_prv = memory::create({engine_, bias_data_  ->layout_int, {1,        {{bias_x}}, 1}});
  weights_diff_->memory_prv = memory::create({engine_, bottom_diff_->layout_int, {input_x, {{output_x}}, 1}});
  bias_diff_   ->memory_prv = memory::create({engine_, bias_diff_  ->layout_int, {1,        {{bias_x}}, 1}});

  weights_data_->memory_usr = memory::create({engine_, bottom_data_->layout_usr, {input_x, {{output_x}}, 1}});
  bias_data_   ->memory_usr = memory::create({engine_, bias_data_  ->layout_usr, {1,        {{bias_x}}, 1}});
  weights_diff_->memory_usr = memory::create({engine_, bottom_diff_->layout_usr, {input_x, {{output_x}}, 1}});
  bias_diff_   ->memory_usr = memory::create({engine_, bias_diff_  ->layout_usr, {1,        {{bias_x}}, 1}});

  // Names are for debugging only
  bottom_data_->name = "fwd_bottom_data   @ " + this->layer_param_.name();
  top_data_   ->name = "fwd_top_data      @ " + this->layer_param_.name();
  top_diff_   ->name = "bwd_top_diff      @ " + this->layer_param_.name();
  bottom_diff_->name = "bwd_bottom_diff   @ " + this->layer_param_.name();

  weights_data_->name = "weights_data      @ " + this->layer_param_.name();
  bias_data_   ->name = "bias_data         @ " + this->layer_param_.name();
  weights_diff_->name = "weights_diff      @ " + this->layer_param_.name();
  bias_diff_   ->name = "bias_diff         @ " + this->layer_param_.name();

  bottom_data_->create_conversions();
  top_data_   ->create_conversions();
  bottom_diff_->create_conversions();
  top_diff_   ->create_conversions();

  weights_data_->create_conversions();
  bias_data_   ->create_conversions();
  weights_diff_->create_conversions();
  bias_diff_   ->create_conversions();

  fcFwd_ = fully_connected::create({ engine_,
                                     top_data_->memory_prv,
                                     bottom_data_->memory_prv,
                                     weights_data_->memory_prv,
                                     bias_data_->memory_prv}
                                     );
  // TODO:
  // fcBwd =
}

template <typename Dtype>
void MKL_DNNInnerProductLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Figure out the dimensions
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.inner_product_param().axis());
  const int new_K = bottom[0]->count(axis);
  CHECK_EQ(K_, new_K)
      << "Input size incompatible with inner product parameters.";
  // The first "axis" dimensions are independent inner products; the total
  // number of these is M_, the product over these dimensions.
  M_ = bottom[0]->count(0, axis);
  // The top shape will be the bottom shape with the flattened axes dropped,
  // and replaced by a single axis with dimension num_output (N_).
  vector<int> top_shape = bottom[0]->shape();
  top_shape.resize(axis + 1);
  top_shape[axis] = N_;
  top[0]->Reshape(top_shape);
  // Set up the bias multiplier
  if (bias_term_) {
    vector<int> bias_shape(1, M_);
    bias_multiplier_.Reshape(bias_shape);
    caffe_set(M_, Dtype(1), bias_multiplier_.mutable_cpu_data());
  }
}

template <typename Dtype>
void MKL_DNNInnerProductLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  auto bottom_data = bottom_data_->get_converted_prv(bottom[0], true);
  auto weight = weights_data_->get_converted_prv(this->blobs_[0].get(), true);
  void *top_data = nullptr;
  if (top_data_->from_internal != nullptr) {
    top[0]->set_prv_data(top_data_->internal_ptr, top_data_, false);
    top_data = top_data_->internal_ptr;
  } else {
    top_data = top[0]->mutable_cpu_data();
    DLOG(INFO) << "Using cpu_data for top in MKL_DNNInnerProductLayer.";
  }

  if (bias_term_) {
      // biases are 1D, so actually conversion not necessary, but this is consistent..
      auto bias = bias_data_->get_converted_prv(this->blobs_[1].get(), true);
      execute({ bottom_data_ ->memory_prv(bottom_data),
                top_data_    ->memory_prv(top_data),
                weights_data_->memory_prv(weight),
                bias_data_   ->memory_prv(bias),
                fcFwd_ });
  } else
  {
    // TODO
    NOT_IMPLEMENTED;
#if 0
    execute({ bottom_data_ ->memory_prv(bottom_data),
            top_data_    ->memory_prv(top_data),
            weights_data_->memory_prv(weight)
            fcFwd_ });
#endif
  }
}

template <typename Dtype>
void MKL_DNNInnerProductLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  // TODO
    NOT_IMPLEMENTED;
#if 0
  if (this->param_propagate_down_[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* bottom_data = bottom[0]->cpu_data();
    // Gradient with respect to weight
    if (transpose_) {
      caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          K_, N_, M_,
          (Dtype)1., bottom_data, top_diff,
          (Dtype)1., this->blobs_[0]->mutable_cpu_diff());
    } else {
      caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          N_, K_, M_,
          (Dtype)1., top_diff, bottom_data,
          (Dtype)1., this->blobs_[0]->mutable_cpu_diff());
    }
  }
  if (bias_term_ && this->param_propagate_down_[1]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    // Gradient with respect to bias
    caffe_cpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
        bias_multiplier_.cpu_data(), (Dtype)1.,
        this->blobs_[1]->mutable_cpu_diff());
  }
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    // Gradient with respect to bottom data
    if (transpose_) {
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
          M_, K_, N_,
          (Dtype)1., top_diff, this->blobs_[0]->cpu_data(),
          (Dtype)0., bottom[0]->mutable_cpu_diff());
    } else {
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
          M_, K_, N_,
          (Dtype)1., top_diff, this->blobs_[0]->cpu_data(),
          (Dtype)0., bottom[0]->mutable_cpu_diff());
    }
  }

#endif
}

#ifdef CPU_ONLY
STUB_GPU(MKL_DNNInnerProductLayer);
#else
template <typename Dtype>
void MKL_DNNInnerProductLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
  {NOT_IMPLEMENTED;}
template <typename Dtype>
void MKL_DNNInnerProductLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom)
  {NOT_IMPLEMENTED;}
#endif


INSTANTIATE_CLASS(MKL_DNNInnerProductLayer);

}  // namespace caffe
#endif  // #ifdef USE_MKL_DNN
#ifdef MKL_DNN_ENABLED
#include <vector>

#include "boost/make_shared.hpp"
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

template <> void MKL_DNNInnerProductLayer<double>::LayerSetUp(
  const vector<Blob<double>*>& bottom,
  const vector<Blob<double>*>& top) {NOT_IMPLEMENTED;}
template <> void MKL_DNNInnerProductLayer<double>::Forward_cpu(
  const vector<Blob<double>*>& bottom,
  const vector<Blob<double>*>& top) {NOT_IMPLEMENTED;}
template <> void MKL_DNNInnerProductLayer<double>::Backward_cpu(
  const vector<Blob<double>*>& top,
  const vector<bool>& propagate_down,
  const vector<Blob<double>*>& bottom) {NOT_IMPLEMENTED;}

template <typename Dtype>
void MKL_DNNInnerProductLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom,
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

  uint32_t batch = bottom[0]->count(0, axis); // TODO: is this ok?
  uint32_t input_x = K_;
  uint32_t bias_x = N_, output_x = N_;

    // Choose layout according to the engine
  switch (engine_) {
    case  neural::engine::cpu:
      prv_layout_in_out_  = memory::format::xb_f32;
      prv_layout_weights_ = memory::format::io_f32;
    break;
    case neural::engine::reference:
      prv_layout_in_out_  = memory::format::xb_f32;
      prv_layout_weights_ = memory::format::oi_f32;
    break;
    default:
      CHECK(0) << "Wrong mkl-dnn engine";
  }

  // Memory setup
  bottom_data_ = boost::make_shared<MKL_DNNData<Dtype> >(
          usr_layout_in_out_, prv_layout_in_out_,
          memory::describe({engine_, usr_layout_in_out_, {batch, {{input_x}},  1}}),
          memory::describe({engine_, prv_layout_in_out_, {batch, {{input_x}},  1}}));

  top_data_ = boost::make_shared<MKL_DNNData<Dtype> >(
          usr_layout_in_out_, prv_layout_in_out_,
          memory::describe({engine_, usr_layout_in_out_, {batch, {{output_x}}, 1}}),
          memory::describe({engine_, prv_layout_in_out_, {batch, {{output_x}}, 1}}));

  weights_data_ = boost::make_shared<MKL_DNNData<Dtype> >(
          usr_layout_weights_, prv_layout_weights_,
          memory::describe({engine_, usr_layout_weights_, {input_x, {{output_x}}, 1}}),
          memory::describe({engine_, prv_layout_weights_, {input_x, {{output_x}}, 1}}));

  bias_data_ = boost::make_shared<MKL_DNNData<Dtype> >(
          layout_bias_, layout_bias_,
          memory::describe({engine_, layout_bias_, {1, {{bias_x}}, 1}}),
          memory::describe({engine_, layout_bias_, {1, {{bias_x}}, 1}}));

  bottom_diff_ = boost::make_shared<MKL_DNNDiff<Dtype> >(
          usr_layout_in_out_, prv_layout_in_out_,
          memory::describe({engine_, usr_layout_in_out_, {batch, {{input_x}},  1}}),
          memory::describe({engine_, prv_layout_in_out_, {batch, {{input_x}},  1}}));

  top_diff_ = boost::make_shared<MKL_DNNDiff<Dtype> >(
          usr_layout_in_out_, prv_layout_in_out_,
          memory::describe({engine_, usr_layout_in_out_, {batch, {{output_x}}, 1}}),
          memory::describe({engine_, prv_layout_in_out_, {batch, {{output_x}}, 1}}));

  weights_diff_ = boost::make_shared<MKL_DNNDiff<Dtype> >(
          usr_layout_weights_, prv_layout_weights_,
          memory::describe({engine_, usr_layout_weights_, {input_x, {{output_x}}, 1}}),
          memory::describe({engine_, prv_layout_weights_, {input_x, {{output_x}}, 1}}));

  bias_diff_ = boost::make_shared<MKL_DNNDiff<Dtype> >(
          layout_bias_, layout_bias_,
          memory::describe({engine_, layout_bias_, {1, {{bias_x}}, 1}}),
          memory::describe({engine_, layout_bias_, {1, {{bias_x}}, 1}}));


  // Names are for debugging only
  bottom_data_ ->name = "fwd_bottom_data   @ " + this->layer_param_.name() + "  ";
  top_data_    ->name = "fwd_top_data      @ " + this->layer_param_.name() + "  ";
  top_diff_    ->name = "bwd_top_diff      @ " + this->layer_param_.name() + "  ";
  bottom_diff_ ->name = "bwd_bottom_diff   @ " + this->layer_param_.name() + "  ";

  weights_data_->name = "weights_data      @ " + this->layer_param_.name() + "  ";
  bias_data_   ->name = "bias_data         @ " + this->layer_param_.name() + "  ";
  weights_diff_->name = "weights_diff      @ " + this->layer_param_.name() + "  ";
  bias_diff_   ->name = "bias_diff         @ " + this->layer_param_.name() + "  ";

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

  // TODO: is this temporary?
  // Conversion for first FC: yxfb_f32 ==> fyxb_f32 which is then used as xb_f32
  if(bottom[0]->get_prv_descriptor_data() != nullptr) {
    CHECK_EQ((bottom[0]->get_prv_descriptor_data())->get_descr_type(),
               PrvMemDescr::PRV_DESCR_MKL_DNN);
    auto bottom_descr =  boost::static_pointer_cast<MKL_DNNData<Dtype> >
               (bottom[0]->get_prv_descriptor_data());

    if(bottom_descr->layout_prv == neural::memory::format::yxfb_f32 &&
            bottom_data_xb_ == nullptr) {
      uint32_t n = bottom[0]->shape(0);
      uint32_t c = bottom[0]->shape(1);
      uint32_t h = bottom[0]->shape(2);
      uint32_t w = bottom[0]->shape(3);

      bottom_data_ = boost::make_shared<MKL_DNNData<Dtype> >(
          neural::memory::format::bfyx_f32, neural::memory::format::fyxb_f32,
          memory::describe({engine_, neural::memory::format::bfyx_f32, {n, {{w, h}}, c}}),
          memory::describe({engine_, neural::memory::format::fyxb_f32, {n, {{w, h}}, c}}));

      // Fake buffer for casting fyxb => xb
      bottom_data_xb_ =  memory::describe({engine_, neural::memory::format::xb_f32, {n, {{w*h*c}}, 1}});

      fcFwd_ = fully_connected::create({ engine_,
                                         top_data_->memory_prv,
                                         bottom_data_xb_,
                                         weights_data_->memory_prv,
                                         bias_data_->memory_prv}
                                         );
    }
  }

  auto bottom_data = bottom_data_->get_converted_prv(bottom[0], true);
  auto weight = weights_data_->get_converted_prv(this->blobs_[0].get(), true);
  Dtype *top_data = nullptr;
  if (top_data_->from_prv != nullptr) {
    top_data = top_data_->prv_ptr();
    top[0]->set_prv_data(top_data, top_data_, false);
  } else {
    top_data = top[0]->mutable_cpu_data();
    DLOG(INFO) << "Using cpu_data for top in MKL_DNNInnerProductLayer.";
  }

  if (bias_term_) {
      // biases are 1D, so actually conversion not necessary, but this is consistent..
      auto bias = bias_data_->get_converted_prv(this->blobs_[1].get(), true);

      execute({ (bottom_data_xb_ != nullptr) ? bottom_data_xb_(bottom_data) : bottom_data_->memory_prv(bottom_data),
                top_data_    ->memory_prv(top_data),
                weights_data_->memory_prv(weight),
                bias_data_   ->memory_prv(bias),
                fcFwd_ }).wait();
  } else {
    // TODO
    NOT_IMPLEMENTED;
#if 0
    execute({ bottom_data_ ->memory_prv(bottom_data),
            top_data_    ->memory_prv(top_data),
            weights_data_->memory_prv(weight)
            fcFwd_ }).wait();
#endif
  }
}

template <typename Dtype>
void MKL_DNNInnerProductLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  // TODO
    NOT_IMPLEMENTED;
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
#endif  // #ifdef MKL_DNN_ENABLED
#ifdef MKL_DNN_ENABLED
#include <vector>

#include "boost/make_shared.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/mkl_dnn_layers.hpp"

// #define CONVERSION_PRINT_DATA
// #define CONVERSION_PROFILING

#ifdef CONVERSION_PROFILING
#include "caffe/caffe.hpp"
using caffe::Timer;
#endif

// Uncomment to see where the layout conversions are done
// #undef DLOG
#ifndef DLOG
#define DLOG(x) std::cout
#endif

/*  TODO
 *  biast_term_ support
 *  backward support
 *  separate backwards for input, weight, bias
 *  1d, 3d conv
 */

using neural::convolution;
using neural::padding;

namespace caffe {

// *** Conversion methods for MKL_DNNMemory class
// TODO: move this code to separate file?
template <typename Dtype, bool is_diff>
void MKL_DNNMemory<Dtype, is_diff>::convert_from_prv(
  void* prv_ptr, void* cpu_ptr) {
  CHECK(prv_ptr);
  CHECK(cpu_ptr);
  CHECK(this->from_prv != nullptr);

  DLOG(INFO) << "convert priv =>           "  << this->name << " =>"
             << "                            || layouts: "
             << this->layout_prv << " =>  " << "\n";
#ifdef CONVERSION_PROFILING
  Timer timer;
  timer.Start();
#endif
  execute({memory_prv(prv_ptr), memory_usr(cpu_ptr), this->from_prv}).wait();
#ifdef CONVERSION_PROFILING
  DLOG(INFO) << " *** conversion time: " << timer.MilliSeconds() << " ms.\n";
#endif

#ifdef CONVERSION_PRINT_DATA
  DLOG(INFO) << "Before conversion: \n";
  for (auto i = 0; i < this->prv_count(); i++)
    DLOG(INFO) << reinterpret_cast<Dtype*>(prv_ptr)[i] << " ";
  DLOG(INFO) << " \n";
  DLOG(INFO) << "After  conversion: \n";
  for (auto i = 0; i < this->prv_count(); i++)
    DLOG(INFO) << reinterpret_cast<Dtype*>(cpu_ptr)[i] << " ";
  DLOG(INFO) << " \n\n";
#endif
}

template <typename Dtype, bool is_diff>
Dtype* MKL_DNNMemory<Dtype, is_diff>::get_converted_prv(
  Blob<Dtype>* blob, bool set_prv_ptr,
  MKL_DNNMemory<Dtype, is_diff>* converted_in_fwd) {
  if (this->to_prv != nullptr) {  // Checking is conversion is required
    const Dtype* prv_ptr = is_diff ?  blob->prv_diff() : blob->prv_data();
    if (prv_ptr == nullptr) {
      DLOG(INFO) << "convert      => priv                                => "
                 << this->name  << "  || layouts:   => " << this->layout_prv
                 << "\n";
      auto usr_ptr = is_diff ? const_cast<Dtype*>(blob->cpu_diff()) :
                               const_cast<Dtype*>(blob->cpu_data());
      if (this->prv_ptr_ == nullptr)
        this->allocate();

#ifdef CONVERSION_PROFILING
  Timer timer;
  timer.Start();
#endif
      execute({memory_usr(usr_ptr), memory_prv(this->prv_ptr_), this->to_prv})
        .wait();
#ifdef CONVERSION_PROFILING
  DLOG(INFO) << " *** conversion time: " << timer.MilliSeconds() << " ms.\n";
#endif
#ifdef CONVERSION_PRINT_DATA
      DLOG(INFO) << "Before conversion: \n";
      for (auto i = 0; i < blob->count(); i++)
        DLOG(INFO) << usr_ptr[i] << " ";
      DLOG(INFO) << " \n";
      DLOG(INFO) << "After  conversion: \n";
      for (auto i = 0; i < blob->count(); i++)
        DLOG(INFO) << this->prv_ptr_[i] << " ";
      DLOG(INFO) << " \n\n";
#endif
      if (set_prv_ptr) {
        if (is_diff)
          blob->set_prv_diff(this->prv_ptr_, get_shared_ptr(), true);
        else
          blob->set_prv_data(this->prv_ptr_, get_shared_ptr(), true);
      }

      return this->prv_ptr_;
    } else {
      // Make sure the layout is fine
      shared_ptr<PrvMemDescr> prv_mem_descriptor =
          is_diff ? (blob->get_prv_descriptor_diff()) :
                    (blob->get_prv_descriptor_data());

      CHECK(prv_mem_descriptor != nullptr);
      CHECK(prv_mem_descriptor->get_descr_type() ==
            PrvMemDescr::PRV_DESCR_MKL_DNN);

      shared_ptr<MKL_DNNMemory<Dtype, is_diff> > current_descr =
        boost::static_pointer_cast<MKL_DNNMemory<Dtype, is_diff> >
          (prv_mem_descriptor);

      if (current_descr->layout_prv != this->layout_prv) {
        if (converted_in_fwd != nullptr) {
          // hack for reusing previously done conversion
          if (converted_in_fwd->layout_prv == this->layout_prv) {
            DLOG(INFO) << "layout OK                 "
                      << converted_in_fwd->name << " == " << this->name << "\n";
            return converted_in_fwd->prv_ptr_;
          }
        }

        engine::type conversion_engine = engine::cpu;

        DLOG(INFO) << "convert priv => priv      " << current_descr->name
                   << " => " << this->name << "  || layouts: "
                   << current_descr->layout_prv  << " => " << this->layout_prv
                   << "\n";

        if (this->prv_ptr_ == nullptr)
          this->allocate();

#ifdef CONVERSION_PROFILING
  Timer timer;
  timer.Start();
#endif
        primitive convert = reorder::create(reorder::arguments
          ({conversion_engine, this->memory_prv, current_descr->memory_prv}));
        execute({current_descr->memory_prv(current_descr->prv_ptr_),
                 this->memory_prv(this->prv_ptr_), convert}).wait();

#ifdef CONVERSION_PROFILING
  DLOG(INFO) << " *** conversion time: " << timer.MilliSeconds() << " ms.\n";
#endif

        if (set_prv_ptr) {
          if (is_diff)
            blob->set_prv_diff(this->prv_ptr_, get_shared_ptr(), true);
          else
            blob->set_prv_data(this->prv_ptr_, get_shared_ptr(), true);
        }
        return this->prv_ptr_;
      } else if (current_descr.get() != this) {
        DLOG(INFO) << "layout OK                 " << current_descr->name
                << " == " << this->name << "\n";
      }
    }
    return const_cast<Dtype*> (prv_ptr);
  }

  return (is_diff ? const_cast<Dtype*> (blob->cpu_diff()) :
                    const_cast<Dtype*> (blob->cpu_data()));
}

template <typename Dtype>
void MKL_DNNConvolutionLayer<Dtype>::compute_output_shape() {
  ConvolutionLayer<Dtype>::compute_output_shape();
  this->height_out_ = (this->height_ + 2 * this->pad_h_ - this->kernel_h_)
      / this->stride_h_ + 1;
  this->width_out_ = (this->width_ + 2 * this->pad_w_ - this->kernel_w_)
      / this->stride_w_ + 1;
}

template <typename Dtype>
MKL_DNNConvolutionLayer<Dtype>::~MKL_DNNConvolutionLayer() {}

template <typename Dtype>
void MKL_DNNConvolutionLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {
  ConvolutionLayer<Dtype>::LayerSetUp(bottom, top);

  this->width_ = bottom[0]->width();
  this->height_ = bottom[0]->height();
  this->num_ = bottom[0]->num();

  // TODO: clean up this
  kernel_w_ = this->kernel_shape_.cpu_data()[0];
  kernel_h_ = this->kernel_shape_.cpu_data()[1];
  stride_w_ = this->stride_.cpu_data()[0];
  stride_h_ = this->stride_.cpu_data()[1];
  pad_w_ = this->pad_.cpu_data()[0];
  pad_h_ = this->pad_.cpu_data()[1];

  this->bottom_shape_ = &bottom[0]->shape();
  compute_output_shape();
  unsigned n, g;
  unsigned iw, ih, ic;
  unsigned ow, oh, oc;
  unsigned kw, kh; /* filter */

  g  = this->group_;
  n  = this->num_;
  iw = this->width_;
  ih = this->height_;
  ic = this->channels_;

  ow = this->width_out_;
  oh = this->height_out_;
  oc = this->num_output_;

  kw = this->kernel_w_;
  kh = this->kernel_h_;

  // Choose layout according to the engine
  switch (engine_) {
    case  engine::cpu:
    {
      if ((n % 24 == 0) && (ic % 8 == 0) && (oc % 4 == 0)) {
        prv_layout_in_out_ = memory::format::byxf_b24_f32;
        prv_layout_filter_ = memory::format::yxoi_o4_f32;
      } else {
        prv_layout_in_out_ = memory::format::byxf_f32;
        prv_layout_filter_ = memory::format::oyxi_o16_f32;
      }
    }
    break;
    case engine::reference:
      prv_layout_in_out_ = memory::format::yxfb_f32;
      prv_layout_filter_ = memory::format::oiyx_f32;

    break;
    default:
      CHECK(0) << "Wrong mkl-dnn engine";
  }
  // Forward setup
  fwd_bottom_data = boost::make_shared<MKL_DNNData<Dtype> >(
    usr_layout_in_out_, prv_layout_in_out_,
    memory::describe({engine_, usr_layout_in_out_, {n, {iw, ih}, ic}}),
    memory::describe({engine_, prv_layout_in_out_, {n, {iw, ih}, ic}}));

  fwd_top_data = boost::make_shared<MKL_DNNData<Dtype> >(
    usr_layout_in_out_, prv_layout_in_out_,
    memory::describe({engine_, usr_layout_in_out_, {n, {ow, oh}, oc }}),
    memory::describe({engine_, prv_layout_in_out_, {n, {ow, oh}, oc }}));

  fwd_filter_data = boost::make_shared<MKL_DNNData<Dtype> >(
    usr_layout_filter_, prv_layout_filter_,
    memory::describe({engine_, usr_layout_filter_, {1, {kw, kh}, {oc, ic/g}}}),
    memory::describe({engine_, prv_layout_filter_, {1, {kw, kh}, {oc, ic/g}}}));

  fwd_bias_data = boost::make_shared<MKL_DNNData<Dtype> >(
    layout_bias_, layout_bias_,
    memory::describe({engine_, layout_bias_, {1, {{oc}}, 1}}),
    memory::describe({engine_, layout_bias_, {1, {{oc}}, 1}}));

  if (engine_ == engine::cpu) {
    // TODO: For groups > 1 API for engine::cpu is different
    filters_.push_back(memory::describe({engine_, fwd_filter_data->layout_prv,
      {1, {kw, kh}, {oc, ic/g}}}));
    biases_. push_back(memory::describe({engine_, fwd_bias_data->layout_prv,
      {1, {{oc}}, 1}}));
    convolution_fwd_.push_back(
        convolution::create({engine_,
                             fwd_top_data->memory_prv,
                             {0, {0, 0}, 0},
                             {n, {ow, oh}, oc},
                             { fwd_bottom_data->memory_prv,
                               filters_[0],
                               biases_[0] },
                             {0, {-pad_w_, -pad_h_}, 0},
                             {1, {stride_w_, stride_h_}, 1},
                             padding::zero}));
  } else {
    for (unsigned i = 0; i < g; i++) {
      filters_.push_back(memory::describe({engine_, fwd_filter_data->layout_prv,
        {1, {kw, kh}, {oc/g, ic/g}}}));
      biases_. push_back(memory::describe({engine_, fwd_bias_data->layout_prv,
        {1, {{oc/g}}, 1}}));
      convolution_fwd_.push_back(
        convolution::create({engine_,
                             fwd_top_data->memory_prv,
                             {0, {0, 0}, i*(oc/g)},
                             {n, {ow, oh}, oc/g},
                             { fwd_bottom_data->memory_prv,
                               filters_[i],
                               biases_[i] },
                             {0, {-pad_w_, -pad_h_}, static_cast<int>(i*ic/g)},
                             {1, {stride_w_, stride_h_}, 1},
                             padding::zero}));
    }
  }

/*
 * Backward by setup
 */

  bwd_bottom_diff = boost::make_shared<MKL_DNNDiff<Dtype> >(
          usr_layout_in_out_, prv_layout_in_out_,
          memory::describe({engine_, usr_layout_in_out_, {n, {iw, ih}, ic}}),
          memory::describe({engine_, prv_layout_in_out_, {n, {iw, ih}, ic}}));

  bwd_top_diff = boost::make_shared<MKL_DNNDiff<Dtype> >(
          usr_layout_in_out_, prv_layout_in_out_,
          memory::describe({engine_, usr_layout_in_out_, {n, {ow, oh}, oc }}),
          memory::describe({engine_, prv_layout_in_out_, {n, {ow, oh}, oc }}));

  bwd_filter_diff = boost::make_shared<MKL_DNNDiff<Dtype> >(
          usr_layout_filter_, prv_layout_filter_,
          memory::describe({engine_, usr_layout_filter_,
            {1, {kw, kh}, {oc, ic/g}}}),
          memory::describe({engine_, prv_layout_filter_,
            {1, {kw, kh}, {oc, ic/g}}}));

  bwd_bias_diff = boost::make_shared<MKL_DNNDiff<Dtype> >(
          layout_bias_, layout_bias_,
          memory::describe({engine_, layout_bias_, {1, {{oc}}, 1}}),
          memory::describe({engine_, layout_bias_, {1, {{oc}}, 1}}));

  // TODO:
  // convolution_bwd =

  // Names are for debugging purposes only. TODO: Consider removing this.
  fwd_bottom_data->name = "fwd_bottom_data   @ " + this->layer_param_.name();
  fwd_top_data   ->name = "fwd_top_data      @ " + this->layer_param_.name();
  fwd_filter_data->name = "fwd_filter_data   @ " + this->layer_param_.name();
  fwd_bias_data  ->name = "fwd_bias_data     @ " + this->layer_param_.name();
  bwd_top_diff   ->name = "bwd_top_diff      @ " + this->layer_param_.name();
  bwd_bottom_diff->name = "bwd_bottom_diff   @ " + this->layer_param_.name();
  bwd_filter_diff->name = "bwd_filter_diff   @ " + this->layer_param_.name();
  bwd_bias_diff  ->name = "bwd_bias_diff     @ " + this->layer_param_.name();
}


template <>
void MKL_DNNConvolutionLayer<double>::Forward_cpu(
  const vector<Blob<double>*>& bottom,
  const vector<Blob<double>*>& top) { NOT_IMPLEMENTED; }

template <>
void MKL_DNNConvolutionLayer<double>::Backward_cpu(
  const vector<Blob<double>*>& top,
  const vector<bool>& propagate_down,
  const vector<Blob<double>*>& bottom) { NOT_IMPLEMENTED; }


template <typename Dtype>
void MKL_DNNConvolutionLayer<Dtype>::Forward_cpu(
  const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {
  int n, g;
  int iw, ih, ic;
  int ow, oh, oc;

  g  = this->group_;
  n  = this->num_;
  iw = this->width_;
  ih = this->height_;
  ic = this->channels_/g;

  CHECK_EQ(bottom[0]->width()   , iw) << "Inclompatible shape of bottom";
  CHECK_EQ(bottom[0]->height()  , ih) << "Inclompatible shape of bottom";
  CHECK_EQ(bottom[0]->channels(), ic*g) << "Inclompatible shape of bottom";
  CHECK_EQ(bottom[0]->num()     , n) << "Inclompatible shape of bottom";

  ow = this->width_out_;
  oh = this->height_out_;
  oc = this->num_output_/g;
  CHECK_EQ(top[0]->width()   , ow)   << "Inclompatible shape of bottom";
  CHECK_EQ(top[0]->height()  , oh)   << "Inclompatible shape of bottom";
  CHECK_EQ(top[0]->channels(), oc*g) << "Inclompatible shape of bottom";
  CHECK_EQ(top[0]->num()     , n)    << "Inclompatible shape of bottom";

  Dtype* bottom_data = fwd_bottom_data->get_converted_prv(bottom[0], true);
  Dtype* filter_data = fwd_filter_data->get_converted_prv(this->blobs_[0].get(),
    true);
  Dtype* bias_data   = fwd_bias_data  ->get_converted_prv(this->blobs_[1].get(),
    true);
  Dtype* top_data;
  if (fwd_top_data->from_prv != nullptr) {
    top_data = fwd_top_data->prv_ptr();
    top[0]->set_prv_data(top_data, fwd_top_data, false);
  } else {
    top_data = top[0]->mutable_cpu_data();
  }

  if (this->bias_term_) {
    if (engine_ == engine::cpu) {
      // TODO: For groups > 1 API for engine::cpu is different
      execute({fwd_bottom_data->memory_prv(bottom_data),
               fwd_top_data->memory_prv(top_data),
               filters_[0](filter_data),
               biases_[0](bias_data),
               convolution_fwd_[0]}).wait();
    } else {
      CHECK_EQ(convolution_fwd_.size(), g);
      CHECK_EQ(biases_.size(), g);
      CHECK_EQ(filters_.size(), g);
      for (int i = 0; i < g; i++) {
        execute({fwd_bottom_data->memory_prv(bottom_data),
                fwd_top_data->memory_prv(top_data),
                filters_[i](filter_data + i* this->weight_offset_),
                biases_[i](bias_data + i*oc),
                convolution_fwd_[i]}).wait();
      }
    }
  } else {
    NOT_IMPLEMENTED;  // TODO: not supported currently
    for (int i = 0; i < g; i++) {
      execute({fwd_bottom_data->memory_prv(bottom_data),
              fwd_top_data->memory_prv(top_data),
              filters_[i](filter_data + i* this->weight_offset_),
              convolution_fwd_[i]}).wait();
    }
  }
}

template <typename Dtype>
void MKL_DNNConvolutionLayer<Dtype>::Backward_cpu(
  const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
  const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(MKL_DNNConvolutionLayer);
#else
template <typename Dtype>
void MKL_DNNConvolutionLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
  {NOT_IMPLEMENTED;}
template <typename Dtype>
void MKL_DNNConvolutionLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom)
  {NOT_IMPLEMENTED;}
#endif

INSTANTIATE_CLASS(MKL_DNNConvolutionLayer);
}  // namespace caffe
#endif  // #ifdef MKL_DNN_ENABLED

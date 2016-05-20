#ifdef MKL_DNN_ENABLED
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/mkl_dnn_layers.hpp"

// Uncomment to see where the layout conversions are done
//#undef DLOG
#ifndef DLOG
#include <iostream>
#define DLOG(x) std::cout
#endif

/*  TODO
 *  biast_term_ support
 *  backward support
 *  separate backwards for input, weight, bias
 *  1d, 3d conv
 */
using namespace neural;

namespace caffe {
template <typename Dtype>
MKL_DNNConvolutionLayer<Dtype>::MKL_DNNConvolutionLayer(
  const LayerParameter& param, neural::engine::type engine)
      : ConvolutionLayer<Dtype>(param), engine_(engine),
        fwd_bottom_data  (new MKL_DNNData<Dtype>(memory::format::bfyx_f32, memory::format::yxfb_f32)),
        fwd_top_data     (new MKL_DNNData<Dtype>(memory::format::bfyx_f32, memory::format::yxfb_f32)),
        fwd_filter_data  (new MKL_DNNData<Dtype>(memory::format::oiyx_f32, memory::format::oiyx_f32)),
        fwd_bias_data    (new MKL_DNNData<Dtype>(memory::format::x_f32,    memory::format::x_f32)),
        bwd_top_diff     (new MKL_DNNDiff<Dtype>(memory::format::bfyx_f32, memory::format::yxfb_f32)),
        bwd_bottom_diff  (new MKL_DNNDiff<Dtype>(memory::format::bfyx_f32, memory::format::yxfb_f32)),
        bwd_filter_diff  (new MKL_DNNDiff<Dtype>(memory::format::oiyx_f32, memory::format::oiyx_f32)),
        bwd_bias_diff    (new MKL_DNNDiff<Dtype>(memory::format::x_f32,    memory::format::x_f32))
        {
          if(engine_ == neural::engine::cpu) {
            fwd_bottom_data->layout_prv = memory::format::byxf_f32;
            bwd_bottom_diff->layout_prv = memory::format::byxf_f32;
            fwd_top_data   ->layout_prv = memory::format::byxf_f32;
            bwd_top_diff   ->layout_prv = memory::format::byxf_f32;
            fwd_filter_data->layout_prv = memory::format::os_yxi_sv16_f32;
            bwd_filter_diff->layout_prv = memory::format::os_yxi_sv16_f32;
          }
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
void MKL_DNNConvolutionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
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

  // Forward setup
  fwd_bottom_data->memory_usr = memory::create({engine_, fwd_bottom_data->layout_usr, {n, {ih, iw},  ic}});
  fwd_top_data   ->memory_usr = memory::create({engine_, fwd_top_data   ->layout_usr, {n, {oh, ow},  oc }});
  fwd_filter_data->memory_usr = memory::create({engine_, fwd_filter_data->layout_usr, {1, {kh, kw}, {oc, ic/g}}});
  fwd_bias_data  ->memory_usr = memory::create({engine_, fwd_bias_data  ->layout_usr, {1, {{oc}}, 1}});

  fwd_bottom_data->memory_prv = memory::create({engine_, fwd_bottom_data->layout_prv, {n, {ih, iw},  ic}});
  fwd_top_data   ->memory_prv = memory::create({engine_, fwd_top_data   ->layout_prv, {n, {oh, ow},  oc }});
  fwd_filter_data->memory_prv = memory::create({engine_, fwd_filter_data->layout_prv, {1, {kh, kw}, {oc, ic/g}}});
  fwd_bias_data  ->memory_prv = memory::create({engine_, fwd_bias_data  ->layout_prv, {1, {{oc}}, 1}});

  fwd_bottom_data->create_conversions();
  fwd_top_data   ->create_conversions();
  fwd_filter_data->create_conversions();
  fwd_bias_data  ->create_conversions();

  for (unsigned i=0; i<g; i++) {
    filters_.push_back(memory::create({engine_, fwd_filter_data->layout_prv, {1, {kh, kw}, {oc/g, ic/g}}}));
    biases_. push_back(memory::create({engine_, fwd_bias_data  ->layout_prv, {1, {{oc/g}}, 1}}));
    convolution_fwd_.push_back(
      convolution::create({engine_,
                           fwd_top_data->memory_prv,
                           {0, {0, 0}, i*(oc/g)},
                           {n, {oh, ow}, oc/g},
                           { fwd_bottom_data->memory_prv,
                             filters_[i],
                             biases_ [i] },
                           {0, {-pad_h_, -pad_w_}, static_cast<int>(i*(ic/g))},
                           {1, {stride_h_, stride_w_}, 1},
                           padding::zero}
                         ));
  }

/*
 * Backward by setup
 */

  bwd_bottom_diff->memory_usr = memory::create({engine_, bwd_bottom_diff->layout_usr, {n,  {ih, iw}, ic}});
  bwd_top_diff   ->memory_usr = memory::create({engine_, bwd_top_diff   ->layout_usr, {n,  {oh, ow}, oc}});
  bwd_filter_diff->memory_usr = memory::create({engine_, bwd_filter_diff->layout_usr, {oc, {kh, kw}, ic}});
  bwd_bias_diff  ->memory_usr = memory::create({engine_, bwd_bias_diff  ->layout_usr, {1,  {{oc}},    1}});

  bwd_bottom_diff->memory_prv = memory::create({engine_, bwd_bottom_diff->layout_prv, {n,  {ih, iw}, ic}});
  bwd_top_diff   ->memory_prv = memory::create({engine_, bwd_top_diff   ->layout_prv, {n,  {oh, ow}, oc}});
  bwd_filter_diff->memory_prv = memory::create({engine_, bwd_filter_diff->layout_prv, {oc, {kh, kw}, ic}});
  bwd_bias_diff  ->memory_prv = memory::create({engine_, bwd_bias_diff  ->layout_prv, {1,  {{oc}},    1}});

  bwd_bottom_diff->create_conversions();
  bwd_top_diff->create_conversions();
  bwd_filter_diff->create_conversions();
  bwd_bias_diff->create_conversions();

  // TODO:
#if 0
  convolution_bwd = convolution_backward::create({ engine_,
                                              std::vector<primitive>({bwd_bottom_diff->memory_prv,
                                                                      bwd_filter_diff->memory_prv,
                                                                      bwd_bias_diff->memory_prv}),
                                          //   {out_off_y, out_off_x, out_off_z, out_off_b},
                                          //   {out_siz_y, out_siz_x, out_siz_z, out_siz_b},
                                              {bwd_top_diff->memory_prv,
                                               fwd_bottom_data->memory_prv,
                                               fwd_filter_data->memory_prv,
                                               fwd_bias_data->memory_prv},
                                          //  {in_off_y, in_off_x, in_off_z, in_off_b},
                                               {1, {stride_h_, stride_w_}, 1},
                                              padding::zero
                                            });
#endif

  // Names are for debugging purposes only. TODO: Consider removing this.
  fwd_bottom_data    ->name = "fwd_bottom_data   @ " + this->layer_param_.name();
  fwd_top_data       ->name = "fwd_top_data      @ " + this->layer_param_.name();
  fwd_filter_data    ->name = "fwd_filter_data   @ " + this->layer_param_.name();
  fwd_bias_data      ->name = "fwd_bias_data     @ " + this->layer_param_.name();
  bwd_top_diff       ->name = "bwd_top_diff      @ " + this->layer_param_.name();
  bwd_bottom_diff    ->name = "bwd_bottom_diff   @ " + this->layer_param_.name();
  bwd_filter_diff    ->name = "bwd_filter_diff   @ " + this->layer_param_.name();
  bwd_bias_diff      ->name = "bwd_bias_diff     @ " + this->layer_param_.name();
}

// TODO: move this code to separate file
template <typename Dtype, bool is_diff>
void MKL_DNNMemoryDescriptor<Dtype, is_diff>::convert_from_prv(void* prv_ptr, void* cpu_ptr)
{
  CHECK(prv_ptr);
  CHECK(cpu_ptr);
  CHECK(this->from_prv != nullptr);

  DLOG(INFO) << "convert priv =>           "  << this->name << " =>"  << "\n";
  execute({memory_prv(prv_ptr), memory_usr(cpu_ptr), this->from_prv}).sync();
#ifdef CONVERSION_PRINT_DATA
  DLOG(INFO) << "Before conversion: ";
  for (auto i=0; i<this->prv_count(); i++)
    DLOG(INFO) << ((Dtype*)prv_ptr)[i] << " ";
  DLOG(INFO) << " \n";
  DLOG(INFO) << "After  conversion: ";
  for (auto i=0; i<this->prv_count(); i++)
    DLOG(INFO) << ((Dtype*)cpu_ptr)[i] << " ";
  DLOG(INFO) << " \n\n";
#endif
}

template <typename Dtype, bool is_diff>
Dtype* MKL_DNNMemoryDescriptor<Dtype, is_diff>::get_converted_prv(
  Blob<Dtype>* blob, bool set_prv_ptr, MKL_DNNMemoryDescriptor<Dtype, is_diff>* converted_in_fwd) {

  if (this->to_prv != nullptr)  // Checking is conversion is required
  {
    const Dtype* prv_ptr = is_diff ?  blob->prv_diff() : blob->prv_data();
    if(prv_ptr == NULL)
    {
      DLOG(INFO) << "convert      => priv                                => " << this->name << "\n";
      auto usr_ptr = is_diff ? (Dtype *) blob->cpu_diff() : (Dtype *) blob->cpu_data();
      execute({memory_usr(usr_ptr), memory_prv(this->prv_ptr), this->to_prv}).sync();

#ifdef CONVERSION_PRINT_DATA
      DLOG(INFO) << "Before conversion: ";
      for (auto i=0; i<blob->count(); i++)
        DLOG(INFO) << usr_ptr[i] << " ";
      DLOG(INFO) << " \n";
      DLOG(INFO) << "After  conversion: ";
      for (auto i=0; i<blob->count(); i++)
        DLOG(INFO) << this->prv_ptr[i] << " ";
      DLOG(INFO) << " \n\n";
#endif
      if (set_prv_ptr) {
        if(is_diff)
          blob->set_prv_diff(this->prv_ptr, get_shared_ptr(), true);
        else
          blob->set_prv_data(this->prv_ptr, get_shared_ptr(), true);
      }
      return this->prv_ptr;
    }
    else
    {
      // Make sure the layout is fine
      shared_ptr<PrvMemDescr> prv_mem_descriptor =
          is_diff ? (blob->get_prv_descriptor_diff()) : (blob->get_prv_descriptor_data());

      CHECK(prv_mem_descriptor != nullptr);
      CHECK(prv_mem_descriptor->get_descr_type() == PrvMemDescr::PRV_DESCR_MKL_DNN);

      shared_ptr<MKL_DNNMemoryDescriptor<Dtype, is_diff> > current_descr =
        boost::static_pointer_cast<MKL_DNNMemoryDescriptor<Dtype, is_diff> > (prv_mem_descriptor);

      if(current_descr->layout_prv != this->layout_prv) {
        if(converted_in_fwd != nullptr) {
          // hack for reusing previously done conversion
          if(converted_in_fwd->layout_prv == this->layout_prv) {
            DLOG(INFO) << "layout OK                 " << converted_in_fwd->name << " == " << this->name  << "\n";
            return converted_in_fwd->prv_ptr;
          }
        }
        DLOG(INFO) << "convert priv => priv      " << current_descr->name << " => " << this->name
                   << "  || layouts: " << current_descr->layout_prv  << " => " << this->layout_prv<< "\n";
        neural::primitive convert = reorder::create(reorder::arguments({engine::reference, current_descr->memory_prv, this->memory_prv}));
        execute({current_descr->memory_prv(current_descr->prv_ptr),
                 this->memory_prv(this->prv_ptr), convert}).sync();

        if (set_prv_ptr) {
          if(is_diff)
            blob->set_prv_diff(this->prv_ptr, get_shared_ptr(), true);
          else
            blob->set_prv_data(this->prv_ptr, get_shared_ptr(), true);
        }
        return this->prv_ptr;
      }
      else if(current_descr.get() != this) {
        DLOG(INFO) << "layout OK                 " << current_descr->name << " == " << this->name << "\n";
      }
    }

    return (Dtype*) prv_ptr;
  }

  return (is_diff ? (Dtype*) blob->cpu_diff() : (Dtype*) blob->cpu_data());
}

template <>
void MKL_DNNConvolutionLayer<double>::Forward_cpu(const vector<Blob<double>*>& bottom,
    const vector<Blob<double>*>& top) { NOT_IMPLEMENTED; }

template <>
void MKL_DNNConvolutionLayer<double>::Backward_cpu(const vector<Blob<double>*>& top,
        const vector<bool>& propagate_down,
    const vector<Blob<double>*>& bottom) { NOT_IMPLEMENTED; }


template <typename Dtype>
void MKL_DNNConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top)
{
  int n, g;
  int iw, ih, ic;
  int ow, oh, oc;

  g  = this->group_;
  n  = this->num_;
  iw = this->width_;
  ih = this->height_;
  ic = this->channels_/g;

  CHECK_EQ(bottom[0]->width()   , iw  ) << "Inclompatible shape of bottom with layer";
  CHECK_EQ(bottom[0]->height()  , ih  ) << "Inclompatible shape of bottom with layer";
  CHECK_EQ(bottom[0]->channels(), ic*g) << "Inclompatible shape of bottom with layer";
  CHECK_EQ(bottom[0]->num()     , n   ) << "Inclompatible shape of bottom with layer";

  ow = this->width_out_;
  oh = this->height_out_;
  oc = this->num_output_/g;
  CHECK_EQ(top[0]->width()   , ow)   << "Inclompatible shape of bottom with layer";
  CHECK_EQ(top[0]->height()  , oh)   << "Inclompatible shape of bottom with layer";
  CHECK_EQ(top[0]->channels(), oc*g) << "Inclompatible shape of bottom with layer";
  CHECK_EQ(top[0]->num()     , n)    << "Inclompatible shape of bottom with layer";

  CHECK_EQ(convolution_fwd_.size(), g);
  CHECK_EQ(filters_.size(), g);
  CHECK_EQ(biases_.size(), g);

  auto bottom_data   = fwd_bottom_data->get_converted_prv(bottom[0], true);
  float* filter_data = fwd_filter_data->get_converted_prv(this->blobs_[0].get(), true);
  Dtype* top_data;
  if (fwd_top_data->from_prv != nullptr) {
    top_data = fwd_top_data->prv_ptr;
    top[0]->set_prv_data(fwd_top_data->prv_ptr, fwd_top_data, false);
  } else {
    top_data = top[0]->mutable_cpu_data();
  }

  if(this->bias_term_) {
    float* bias_data   = fwd_bias_data  ->get_converted_prv(this->blobs_[1].get(), true);
    for(int i=0; i<g; i++) {
      execute({fwd_bottom_data->memory_prv(bottom_data), fwd_top_data->memory_prv(top_data),
              filters_[i](filter_data + i* this->weight_offset_), biases_[i](bias_data + i*oc),
              convolution_fwd_[i]}).sync();
    }
  } else {
    NOT_IMPLEMENTED;  // TODO, not supported currently
    for(int i=0; i<g; i++) {
      execute({fwd_bottom_data->memory_prv(bottom_data), fwd_top_data->memory_prv(top_data),
              filters_[i](filter_data + i* this->weight_offset_),
              convolution_fwd_[i]}).sync();
    }
  }
}

template <typename Dtype>
void MKL_DNNConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{
  NOT_IMPLEMENTED;  // This is WIP - not tested

  size_t n, g;
  size_t iw, ih, ic;
  size_t ow, oh, oc;

  g  = this->group_;
  n  = this->num_;
  iw = this->width_;
  ih = this->height_;
  ic = this->channels_/g;

  CHECK_EQ(bottom[0]->width()   , iw  ) << "Inclompatible shape of bottom with layer";
  CHECK_EQ(bottom[0]->height()  , ih  ) << "Inclompatible shape of bottom with layer";
  CHECK_EQ(bottom[0]->channels(), ic*g) << "Inclompatible shape of bottom with layer";
  CHECK_EQ(bottom[0]->num()     , n   ) << "Inclompatible shape of bottom with layer";

  ow = this->width_out_;
  oh = this->height_out_;
  oc = this->num_output_/g;
  CHECK_EQ(top[0]->width()   , ow  ) << "Inclompatible shape of bottom with layer";
  CHECK_EQ(top[0]->height()  , oh  ) << "Inclompatible shape of bottom with layer";
  CHECK_EQ(top[0]->channels(), oc*g) << "Inclompatible shape of bottom with layer";
  CHECK_EQ(top[0]->num()     , n   ) << "Inclompatible shape of bottom with layer";

  // TODO: why can't we do it separately?
  if (propagate_down[0] || this->param_propagate_down(0) || this->param_propagate_down(1))
  {
    auto top_diff    = bwd_top_diff   ->get_converted_prv(top[0], true);
    auto filter_data = fwd_filter_data->get_converted_prv(this->blobs_[0].get(), false);
    auto bottom_data = fwd_bottom_data->get_converted_prv(bottom[0], false);
    auto bias_data = fwd_bias_data->get_converted_prv(this->blobs_[1].get(), false);

    Dtype* bottom_diff;
    if (bwd_bottom_diff->from_prv != nullptr)
    {
      bottom[0]->set_prv_diff(bwd_bottom_diff->prv_ptr, bwd_bottom_diff, false);
      bottom_diff = bwd_bottom_diff->prv_ptr;
    }
    else
      bottom_diff = bottom[0]->mutable_cpu_diff();

    Dtype* filter_diff;
    if (bwd_filter_diff->from_prv != nullptr)
    {
      this->blobs_[0]->set_prv_diff(bwd_filter_diff->prv_ptr, bwd_filter_diff, false);
      filter_diff = bwd_filter_diff->prv_ptr;
    }
    else
      filter_diff = this->blobs_[0]->mutable_cpu_diff();

    Dtype* bias_diff;
    if (bwd_bias_diff->from_prv != nullptr) {
      this->blobs_[1]->set_prv_diff(bwd_bias_diff->prv_ptr, bwd_bias_diff, false);
      bias_diff = bwd_bias_diff->prv_ptr;
    }
    else
      bias_diff = this->blobs_[1]->mutable_cpu_diff();

    execute({
        bwd_top_diff->memory_prv(top_diff), fwd_bottom_data->memory_prv(bottom_data),
                fwd_filter_data->memory_prv(filter_data), fwd_bias_data->memory_prv(bias_data),  //inputs
        bwd_bottom_diff->memory_prv(bottom_diff), bwd_filter_diff->memory_prv(filter_diff), bwd_bias_diff->memory_prv(bias_diff), //outputs
        convolution_bwd
    }).sync();
  }
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
#endif // #ifdef MKL_DNN_ENABLED
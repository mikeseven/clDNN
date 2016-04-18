#ifdef USE_MKL_DNN
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/mkl_dnn_layers.hpp"

// Uncomment to see where the layout conversions are done
#undef DLOG
#ifndef DLOG
#include <iostream>
#define DLOG(x) std::cout
#endif

using namespace neural;

namespace caffe {
template <typename Dtype>
MKL_DNNConvolutionLayer<Dtype>::MKL_DNNConvolutionLayer(const LayerParameter& param, neural::engine::type engine)
      : ConvolutionLayer<Dtype>(param),
        fwd_bottom_data  (new MKL_DNNData<Dtype>()),
        fwd_top_data     (new MKL_DNNData<Dtype>()),
        fwd_filter_data  (new MKL_DNNData<Dtype>()),
        fwd_bias_data    (new MKL_DNNData<Dtype>()),
        bwd_top_diff     (new MKL_DNNDiff<Dtype>()),
        bwd_bottom_diff  (new MKL_DNNDiff<Dtype>()),
        bwd_filter_diff  (new MKL_DNNDiff<Dtype>()),
        bwd_bias_diff    (new MKL_DNNDiff<Dtype>()),
        engine_(engine) {}

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

  DLOG(INFO) << " input:   " << iw << " " << ih << " " << ic <<  " " << n << "\n";
  DLOG(INFO) << " output:  " << ow << " " << oh << " " << oc <<  " " << n << "\n";
  DLOG(INFO) << " weights: " << kw << " " << kh << " " << ic <<  " " << oc << "\n";

  // Forward setup
  CHECK_EQ(g, 1);
  fwd_bottom_data->memory_usr = memory::create({engine::cpu, memory::format::bfyx_f32, {n, ic, ih, iw}});
  fwd_top_data->memory_usr    = memory::create({engine::cpu, memory::format::bfyx_f32, {n, oc, oh, ow}});
  fwd_filter_data->memory_usr = memory::create({engine::cpu, memory::format::bfyx_f32, {oc, ic/g, kh, kw}});
  fwd_bias_data->memory_usr   = memory::create({engine::cpu, memory::format::x_f32, {oc}});
  fwd_bias_data->layout_usr   = memory::format::x_f32;

  fwd_bottom_data->memory = memory::create({engine::cpu, memory::format::yxfb_f32, {ih, iw, ic, n}});
  fwd_top_data->memory    = memory::create({engine::cpu, memory::format::yxfb_f32, {oh, ow, oc, n}});
  fwd_filter_data->memory = memory::create({engine::cpu, memory::format::yxfb_f32, {kh, kw, ic/g, oc}});
  fwd_bias_data->memory   = memory::create({engine::cpu, memory::format::x_f32, {oc}});
  fwd_bias_data->layout_int = memory::format::x_f32;

  fwd_bottom_data->create_conversions();
  fwd_top_data->create_conversions();
  fwd_filter_data->create_conversions();
  fwd_bias_data->create_conversions();

  //TODO: support for g>1
  std::vector<unsigned> top_sizes_g = {oh, ow, oc/g, n};
  convolution_fwd = convolution::create( {engine_,
                                        fwd_top_data->memory,
                                        {0,0,0,0},
                                        top_sizes_g,
                                        fwd_bottom_data->memory,
                                        {-pad_h_, -pad_w_, 0, 0},
                                        {stride_h_, stride_w_, 1, 1},
                                        fwd_filter_data->memory,
                                        fwd_bias_data->memory,
                                        padding::zero}
                                      );
/*
 * Backward by setup
 */

  bwd_bottom_diff->memory_usr = memory::create({engine::cpu, memory::format::bfyx_f32, {n, ic, ih, iw}});
  bwd_top_diff->memory_usr    = memory::create({engine::cpu, memory::format::bfyx_f32, {n, oc, oh, ow}});
  bwd_filter_diff->memory_usr = memory::create({engine::cpu, memory::format::bfyx_f32, {oc, ic/g, kh, kw}});
  bwd_bias_diff->memory_usr   = memory::create({engine::cpu, memory::format::x_f32, {oc}});
  bwd_bias_diff->layout_usr  = memory::format::x_f32;

  bwd_bottom_diff->memory = memory::create({engine::cpu, memory::format::yxfb_f32, {ih, iw, ic, n}});
  bwd_top_diff->memory    = memory::create({engine::cpu, memory::format::yxfb_f32, {oh, ow, oc, n}});
  bwd_filter_diff->memory = memory::create({engine::cpu, memory::format::yxfb_f32, {kh, kw, ic/g, oc}});
  bwd_bias_diff->memory   = memory::create({engine::cpu, memory::format::x_f32, {oc}});
  bwd_bias_diff->layout_int = memory::format::x_f32;

  bwd_bottom_diff->create_conversions();
  bwd_top_diff->create_conversions();
  bwd_filter_diff->create_conversions();
  bwd_bias_diff->create_conversions();

  // TODO: support for groups
  convolution_bwd = convolution_backward::create({ engine_,
                                              std::vector<primitive>({bwd_bottom_diff->memory,
                                                                      bwd_filter_diff->memory,
                                                                      bwd_bias_diff->memory}),
                                          //   {out_off_y, out_off_x, out_off_z, out_off_b},
                                          //   {out_siz_y, out_siz_x, out_siz_z, out_siz_b},
                                              {bwd_top_diff->memory,
                                               fwd_bottom_data->memory,
                                               fwd_filter_data->memory,
                                               fwd_bias_data->memory},
                                          //  {in_off_y, in_off_x, in_off_z, in_off_b},
                                              {stride_h_, stride_w_, 1, 1},
                                              padding::zero
                                            });


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

template <typename Dtype, bool is_diff>
void MKL_DNNMemoryDescriptor<Dtype, is_diff>::convert_from_prv(void* prv_ptr, void* cpu_ptr)
{
  CHECK(prv_ptr);
  CHECK(cpu_ptr);
  CHECK(this->from_internal != nullptr);

  DLOG(INFO) << "convert priv =>           "  << this->name << " =>"  << "\n";

      DLOG(INFO) << "Before conversion: ";
      for (auto i=0; i<this->prv_count(); i++)
        DLOG(INFO) << ((Dtype*)prv_ptr)[i] << " ";
      DLOG(INFO) << " \n";
      execute({memory(prv_ptr), memory_usr(cpu_ptr), this->from_internal});

      DLOG(INFO) << "After  conversion: ";
      for (auto i=0; i<this->prv_count(); i++)
        DLOG(INFO) << ((Dtype*)cpu_ptr)[i] << " ";
      DLOG(INFO) << " \n\n";

}

template <typename Dtype, bool is_diff>
Dtype* MKL_DNNMemoryDescriptor<Dtype, is_diff>::get_converted_prv(
  Blob<Dtype>* blob, bool set_prv_ptr, MKL_DNNMemoryDescriptor<Dtype, is_diff>* converted_in_fwd) {
  if (this->to_internal != nullptr)
  {
    const Dtype* prv_ptr = is_diff ?  blob->prv_diff() : blob->prv_data();
    if(prv_ptr == NULL)
    {
      DLOG(INFO) << "convert      => priv                                => " << this->name << "\n";
      auto usr_ptr = is_diff ? (Dtype *) blob->cpu_diff() : (Dtype *) blob->cpu_data();
      execute({memory_usr(usr_ptr), memory(this->internal_ptr), this->to_internal});

      DLOG(INFO) << "Before conversion: ";
      for (auto i=0; i<blob->count(); i++)
        DLOG(INFO) << usr_ptr[i] << " ";
       DLOG(INFO) << " \n";
      DLOG(INFO) << "After  conversion: ";
      for (auto i=0; i<blob->count(); i++)
        DLOG(INFO) << this->internal_ptr[i] << " ";
       DLOG(INFO) << " \n\n";

      if (set_prv_ptr) {
        if(is_diff)
          blob->set_prv_diff(this->internal_ptr, get_shared_ptr(), true);
        else
          blob->set_prv_data(this->internal_ptr, get_shared_ptr(), true);
      }
      return this->internal_ptr;
    }
    else
    {
      // Make sure the layout is fine
      shared_ptr<PrvMemDescr> prv_mem_descriptor =
          is_diff ? (blob->get_prv_descriptor_diff()) : (blob->get_prv_descriptor_data());

      CHECK(prv_mem_descriptor->get_descr_type() == PrvMemDescr::PRV_DESCR_MKL_DNN);

      shared_ptr<MKL_DNNMemoryDescriptor<Dtype, is_diff> > current_descr =
        boost::static_pointer_cast<MKL_DNNMemoryDescriptor<Dtype, is_diff> > (prv_mem_descriptor);

      if(current_descr->layout_int != this->layout_int) {
        if(converted_in_fwd != nullptr)
        {
          // hack for reusing previously done conversion
          if(converted_in_fwd->layout_int == this->layout_int)
          {
            DLOG(INFO) << "layout OK                 " << converted_in_fwd->name << " == " << this->name  << "\n";
            return converted_in_fwd->internal_ptr;
          }
        }
        DLOG(INFO) << "convert priv => priv      " << current_descr->name << " => " << this->name  << "\n";

        neural::primitive convert = reorder::create(reorder::arguments({engine::reference, current_descr->memory, this->memory}));
        execute({current_descr->memory(current_descr->internal_ptr),
                 this->memory(this->internal_ptr), convert});

        if (set_prv_ptr) {
          if(is_diff)
            blob->set_prv_diff(this->internal_ptr, get_shared_ptr(), true);
          else
            blob->set_prv_data(this->internal_ptr, get_shared_ptr(), true);
        }
        return this->internal_ptr;
      }
      else if(current_descr.get() != this) {
        DLOG(INFO) << "layout OK                 " << current_descr->name << " == " << this->name;
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

  CHECK(bottom[0]->width()    == iw &&
        bottom[0]->height()   == ih &&
        bottom[0]->channels() == ic*g &&
        bottom[0]->num()      == n) << "Inclompatible shape of bottom with layer";

  ow = this->width_out_;
  oh = this->height_out_;
  oc = this->num_output_/g;
  CHECK(top[0]->width()    == ow &&
        top[0]->height()   == oh &&
        top[0]->channels() == oc*g &&
        top[0]->num()      == n) << "Inclompatible shape of bottom with layer";


  auto bottom_data = fwd_bottom_data->get_converted_prv(bottom[0], true); //temporary false
  auto filter_data = fwd_filter_data->get_converted_prv(this->blobs_[0].get(), true);
  auto bias_data   = fwd_bias_data  ->get_converted_prv(this->blobs_[1].get(), true);
  Dtype* top_data;
  if (fwd_top_data->from_internal != nullptr)
  {
    top_data = fwd_top_data->internal_ptr;
    top[0]->set_prv_data(fwd_top_data->internal_ptr, fwd_top_data, false);
  }
  else
    top_data = top[0]->mutable_cpu_data();

  // for one group
  CHECK_EQ(g,1);
  execute({fwd_bottom_data->memory(bottom_data), fwd_top_data->memory(top_data),
          fwd_filter_data->memory(filter_data), fwd_bias_data->memory(bias_data),
          convolution_fwd});

}

template <typename Dtype>
void MKL_DNNConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{
  size_t n, g;
  size_t iw, ih, ic;
  size_t ow, oh, oc;

  g  = this->group_;
  n  = this->num_;
  iw = this->width_;
  ih = this->height_;
  ic = this->channels_/g;

  CHECK(bottom[0]->width()    == iw &&
        bottom[0]->height()   == ih &&
        bottom[0]->channels() == ic*g &&
        bottom[0]->num()      == n) << "Incompatible shape of bottom with layer";

  ow = this->width_out_;
  oh = this->height_out_;
  oc = this->num_output_/g;
  CHECK(top[0]->width()    == ow &&
        top[0]->height()   == oh &&
        top[0]->channels() == oc*g &&
        top[0]->num()      == n) << "Incompatible shape of bottom with layer";

  // TODO: why can't we do it separately?
  if (propagate_down[0] || this->param_propagate_down(0) || this->param_propagate_down(1))
  {
    auto top_diff    = bwd_top_diff   ->get_converted_prv(top[0], true);
    auto filter_data = fwd_filter_data->get_converted_prv(this->blobs_[0].get(), false);
    auto bottom_data = fwd_bottom_data->get_converted_prv(bottom[0], false);
    auto bias_data = fwd_bias_data->get_converted_prv(this->blobs_[1].get(), false);

    Dtype* bottom_diff;
    if (bwd_bottom_diff->from_internal != nullptr)
    {
      bottom[0]->set_prv_diff(bwd_bottom_diff->internal_ptr, bwd_bottom_diff, false);
      bottom_diff = bwd_bottom_diff->internal_ptr;
    }
    else
      bottom_diff = bottom[0]->mutable_cpu_diff();

    Dtype* filter_diff;
    if (bwd_filter_diff->from_internal != nullptr)
    {
      this->blobs_[0]->set_prv_diff(bwd_filter_diff->internal_ptr, bwd_filter_diff, false);
      filter_diff = bwd_filter_diff->internal_ptr;
    }
    else
      filter_diff = this->blobs_[0]->mutable_cpu_diff();

    Dtype* bias_diff;
    if (bwd_bias_diff->from_internal != nullptr) {
      this->blobs_[1]->set_prv_diff(bwd_bias_diff->internal_ptr, bwd_bias_diff, false);
      bias_diff = bwd_bias_diff->internal_ptr;
    }
    else
      bias_diff = this->blobs_[1]->mutable_cpu_diff();

    execute({
        bwd_top_diff->memory(top_diff), fwd_bottom_data->memory(bottom_data),
                fwd_filter_data->memory(filter_data), fwd_bias_data->memory(bias_data),  //inputs
        bwd_bottom_diff->memory(bottom_diff), bwd_filter_diff->memory(filter_diff), bwd_bias_diff->memory(bias_diff), //outputs
        convolution_bwd
    });
  }
}

#ifdef CPU_ONLY
STUB_GPU(MKL_DNNConvolutionLayer);
#endif

INSTANTIATE_CLASS(MKL_DNNConvolutionLayer);
}  // namespace caffe
#endif // #ifdef USE_MKL_DNN
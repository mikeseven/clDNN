 //todo remove
#include "multidimensional_counter.h"
#include "neural.h"

#include <fstream>
#include <ostream>

template<typename T> //todo remove
void save_1d_data(neural::vector<uint32_t> sizes, std::string filename, T* data) {
    std::ofstream file;
    file.open( filename + ".txt" );

    for( uint32_t x = 0; x < sizes.spatial[0] ; ++x ) {
        file << data[ x ] << std::endl;
    }
    file.close();
}
template<typename T> //todo remove
void save_2d_data( neural::vector<uint32_t> sizes, std::string filename, T* data ) {
    std::ofstream file;
    file.open( filename + ".txt" );

    ndimensional::calculate_idx_obselote<uint32_t> calculate_idx_obselote( sizes.raw );

    std::vector<uint32_t> current_pos(3);
    for( uint32_t batch = 0; batch < sizes.batch[0] ; ++batch ){
        current_pos[ 0 ] = batch;
        file << "n: " << batch << std::endl;
        for( uint32_t x = 0; x < sizes.spatial[0]; ++x ) {
            current_pos[ 2 ] = x;
            file << data[ calculate_idx_obselote(current_pos) ] << "\t";
        }
        file << std::endl;
    }
    file.close();
}
template<typename T> //todo remove
void save_4d_data( neural::vector<uint32_t> sizes, std::string filename, T* data) {
    std::ofstream file;
    file.open( filename + ".txt" );

    ndimensional::calculate_idx_obselote<uint32_t> calculate_idx_obselote( sizes.raw );

    std::vector<uint32_t> current_pos(4);
    for( uint32_t batch = 0; batch < sizes.batch[0] ; ++batch ){
        current_pos[ 0 ] = batch;
        for( uint32_t z = 0; z < sizes.feature[0] ; ++z ){
            current_pos[ 1 ] = z;
            file << "n\\z: " << batch << "\\" << z << std::endl;
            for( uint32_t y = 0; y < sizes.spatial[0]; ++y ) {
                current_pos[ 2 ] = y;
                for( uint32_t x = 0; x < sizes.spatial[1]; ++x ) {
                    current_pos[ 3 ] = x;

                    file << data[ calculate_idx_obselote(current_pos) ] << "\t";
                }
                file << std::endl;
            }
        }
    }
    file.close();
}
template<typename T> //todo remove
void save_data( std::string filename, const neural::memory_obselote& mem ) {
    T* data = static_cast<T*>(mem.pointer);
    auto sizes = mem.argument.size;

    switch(mem.argument.format){
        case neural::memory_obselote::format::x_f32    :
            save_1d_data(sizes, filename, data);
            return;
            break;
        case neural::memory_obselote::format::xb_f32   :
            save_2d_data(sizes, filename, data);
            return;
            break;
        case neural::memory_obselote::format::yxfb_f32 :
            break;
        default                               : throw std::runtime_error("unknown data format");
    }
    save_4d_data(sizes, filename, data);
}

template<typename T> //todo remove
void save_data( std::string filename, const neural::primitive& mem ) {
    try{
        save_data<T>( filename, mem.as<const neural::memory_obselote&>() );
    } catch (...) {
        std::cout << "Primitive to save is not memory_obselote.";
    }
}


//save_data(input_buffer_size, "in_b", input, input_memory_arg.format); //todo remove
//save_data(output_buffer_size, "out_b", output, output_memory_arg.format); //todo remove
//save_data(output_buffer_size, "out_a", output, output_memory_arg.format); //todo remove
//
//for(int i = 0; i < conv_size_y*conv_size_x*conv_size_z*conv_size_b; ++i) //todo remove
//    weight_buffer[i]  = i - 2;
//for(int i = 0; i < input_y*input_x*input_z*input_b; ++i) //todo remove
//     in_buffer[i]  = 1.0f*i - 1.0f*input_x*input_b/2;
//for(int i = 0; i < output_y*output_x*output_z*output_b; ++i) //todo remove
//    out_buffer[i] = -999;

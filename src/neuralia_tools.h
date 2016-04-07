 //todo remove
#include "multidimensional_counter.h"
#include "neural.h"

#include <fstream>
#include <ostream>

template<typename T> //todo remove
void save_1d_data( uint32_t len, std::string filename, T* data) {
    std::ofstream file;
    file.open( filename + ".txt" );

    for( uint32_t x = 0; x < len; ++x ) {
        file << data[ x ] << std::endl;
    }
    file.close();
}
template<typename T> //todo remove
void save_2d_data( std::vector<uint32_t> size, std::string filename, T* data, std::vector<uint32_t> yxfb_pos ) {
    std::ofstream file;
    file.open( filename + ".txt" );

    ndimensional::calculate_idx<uint32_t> calculate_idx( size );

    std::vector<uint32_t> current_pos(2);
    for( uint32_t batch = 0; batch < size[ yxfb_pos[3] ]; ++batch ){
        current_pos[ yxfb_pos[3] ] = batch;
        file << "n: " << batch << std::endl;
        for( uint32_t x = 0; x < size[ yxfb_pos[1] ]; ++x ) {
            current_pos[ yxfb_pos[1] ] = x;
            file << data[ calculate_idx(current_pos) ] << "\t";
        }
        file << std::endl;
    }
    file.close();
}
template<typename T> //todo remove
void save_4d_data( std::vector<uint32_t> size, std::string filename, T* data, std::vector<uint32_t> yxfb_pos ) {
    std::ofstream file;
    file.open( filename + ".txt" );

    ndimensional::calculate_idx<uint32_t> calculate_idx( size );

    std::vector<uint32_t> current_pos(4);
    for( uint32_t batch = 0; batch < size[ yxfb_pos[3] ]; ++batch ){
        current_pos[ yxfb_pos[3] ] = batch;
        for( uint32_t z = 0; z < size[ yxfb_pos[2] ]; ++z ){
            current_pos[ yxfb_pos[2] ] = z;
            file << "n\\z: " << batch << "\\" << z << std::endl;
            for( uint32_t y = 0; y < size[ yxfb_pos[0] ]; ++y ) {
                current_pos[ yxfb_pos[0] ] = y;
                for( uint32_t x = 0; x < size[ yxfb_pos[1] ]; ++x ) {
                    current_pos[ yxfb_pos[1] ] = x;

                    file << data[ calculate_idx(current_pos) ] << "\t";
                }
                file << std::endl;
            }
        }
    }
    file.close();
}
template<typename T> //todo remove
void save_data( std::string filename, const neural::memory& mem ) {
    std::vector<uint32_t> size = mem.argument.size;
    T* data = static_cast<T*>(mem.pointer);        
    std::vector<uint32_t> yxfb_pos(4);

    switch(mem.argument.format){
        case neural::memory::format::x_f32    : 
            save_1d_data(size[0], filename, data);
            return;
            break;
        case neural::memory::format::xb_f32   : yxfb_pos = {0, 0, 0, 1};
            save_2d_data(size, filename, data, yxfb_pos);
            return;
            break;
        case neural::memory::format::yxfb_f32 : yxfb_pos = {0, 1, 2, 3}; break;
        default                               : throw std::runtime_error("unknown data format");
    }
    save_4d_data(size, filename, data, yxfb_pos);
}

template<typename T> //todo remove
void save_data( std::string filename, const neural::primitive& mem ) {
    try{
        save_data<T>( filename, mem.as<const neural::memory&>() );
    } catch (...) {
        std::cout << "Primitive to save is not memory.";
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
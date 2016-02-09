#include "neural.h"

namespace neural {

primitive memory::create(memory::arguments arg){
        return new memory(arg);
}

}
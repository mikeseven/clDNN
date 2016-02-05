#include "neural.h"

namespace neural {

primitive memory::create(arguments arg){
        return new memory(arg);
}

}
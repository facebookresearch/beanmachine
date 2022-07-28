//
// Created by Steffi Stumpos on 7/19/22.
//

#ifndef PAIC2_WORLDSPEC_H
#define PAIC2_WORLDSPEC_H
#include <string>

namespace paic2 {

    // The compiler knows that the world has certain members and the layout of those members, it just doesn't know what those
    // members are called. This is the configuration for the world fields.
    class WorldSpec {
    public:
        WorldSpec(){}
        void set_print_name(std::string name){ _print_name = name; }
        std::string print_name()const{return _print_name;}
    private:
        std::string _print_name;
    };
}

#endif //PAIC2_WORLDSPEC_H

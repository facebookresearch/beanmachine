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
        WorldSpec():_world_size(0){}
        void set_print_name(std::string name){ _print_name = name; }
        void set_world_name(std::string name) { _world_name = name; }
        void set_world_size(int size){
            if(size < 0){
                _world_size = 0;
            } else {
                _world_size = size;
            }
        }
        std::string print_name()const{return _print_name;}
        std::string world_name()const{return _world_name;}
        int world_size()const{return _world_size; }
    private:
        std::string _print_name;
        std::string _world_name;
        int _world_size;
    };
}

#endif //PAIC2_WORLDSPEC_H

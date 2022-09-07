//
// Created by Steffi Stumpos on 7/15/22.
//

#ifndef PAIC2_PAICAST_H
#define PAIC2_PAICAST_H

#include <vector>
#include <string>
#include <iostream>
#include "llvm/ADT/ArrayRef.h"
#include <pybind11/pybind11.h>

namespace paic2 {

    class Location {
    public:
        Location(unsigned int l, unsigned int c):_line(l), _col(c){}
        unsigned int getLine() const {return _line;}
        unsigned int getCol() const {return _col;}
    private:
        unsigned int _line;
        unsigned int _col;
    };

    class PythonFunction {
    public:
        PythonFunction(Location location, const std::string &name):_name(name), _location(std::move(location)) {}
        virtual ~PythonFunction() = default;
        const Location &loc() { return _location; }
        std::string getName() const { return _name; }
    private:
        Location _location;
        std::string _name;
    };

    class PythonModule {
    public:
        static void bind(pybind11::module &m);
        PythonModule(std::vector<std::shared_ptr<PythonFunction>> functions):_functions(functions){}
        virtual ~PythonModule() = default;
        llvm::ArrayRef<std::shared_ptr<PythonFunction>> getFunctions() { return _functions; }
    private:
        std::vector<std::shared_ptr<PythonFunction>> _functions;
    };
}

#endif //PAIC2_PAICAST_H

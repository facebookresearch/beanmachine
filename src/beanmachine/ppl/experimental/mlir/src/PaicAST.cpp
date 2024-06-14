//
// Created by Steffi Stumpos on 6/12/22.
//

#include "PaicAST.h"
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);
using ExpList = std::vector<std::shared_ptr<paic_mlir::Expression>, std::allocator<std::shared_ptr<paic_mlir::Expression>>>;
PYBIND11_MAKE_OPAQUE(ExpList);
using NodeList = std::vector<std::shared_ptr<paic_mlir::Node>, std::allocator<std::shared_ptr<paic_mlir::Node>>>;
PYBIND11_MAKE_OPAQUE(NodeList);
using ParamList = std::vector<std::shared_ptr<paic_mlir::ParamNode>, std::allocator<std::shared_ptr<paic_mlir::ParamNode>>>;
PYBIND11_MAKE_OPAQUE(ParamList);
using FunctionList = std::vector<std::shared_ptr<paic_mlir::PythonFunction>, std::allocator<std::shared_ptr<paic_mlir::PythonFunction>>>;
PYBIND11_MAKE_OPAQUE(FunctionList);

template<typename T>
void bind_vector(pybind11::module &m, const char* name){
    py::class_<std::vector<T, std::allocator<T>>>(m, name)
            .def(py::init<>())
            .def("pop_back", &std::vector<T>::pop_back)
                    /* There are multiple versions of push_back(), etc. Select the right ones. */
            .def("push_back", (void(std::vector<T>::*)(const T &)) &std::vector<T>::push_back)
            .def("back", (T & (std::vector<T>::*) ()) & std::vector<T>::back)
            .def("__len__", [](const std::vector<T> &v) { return v.size(); })
            .def("__iter__",
                    [](std::vector<T> &v) { return py::make_iterator(v.begin(), v.end()); },
                    py::keep_alive<0, 1>());
}


void paic_mlir::Node::bind(pybind11::module &m) {
    py::enum_<NodeKind>(m, "NodeKind").value("Constant", Constant).export_values();
    py::class_<Location>(m, "Location").def(py::init<int,int>());
    py::class_<Type>(m, "Type").def(py::init<std::string>());
    bind_vector<std::shared_ptr<paic_mlir::Node>>(m, "NodeList");
    bind_vector<std::shared_ptr<paic_mlir::ParamNode>>(m, "ParamList");
    bind_vector<std::shared_ptr<paic_mlir::PythonFunction>>(m, "FunctionList");
    bind_vector<std::shared_ptr<paic_mlir::Expression>>(m, "ExpList");

    // NODES
    py::class_<Node, std::shared_ptr<Node>>(m, "Node")
            .def(py::init<Location, NodeKind>())
            .def("loc", &Node::loc);
    py::class_<Expression, std::shared_ptr<Expression>, Node>(m, "Expression")
            .def(py::init<Location,NodeKind,Type>()).def("type", &paic_mlir::Expression::getType);
    py::class_<ConstNode<float>, std::shared_ptr<ConstNode<float>>, Expression>(m, "FloatNode").def(py::init<Location, float>());

    py::class_<DeclareValNode, std::shared_ptr<DeclareValNode>, Node>(m, "DeclareValNode")
            .def(py::init<Location,std::string, NodeKind, Type>())
            .def("type", &paic_mlir::DeclareValNode::getType)
            .def("name", &paic_mlir::DeclareValNode::getPyName);

    py::class_<ParamNode, std::shared_ptr<ParamNode>, DeclareValNode>(m, "ParamNode")
            .def(py::init<Location,std::string, Type>()).def("name", &paic_mlir::DeclareValNode::getPyName).def("type", &paic_mlir::DeclareValNode::getType);

    py::class_<CallNode, std::shared_ptr<CallNode>, Expression>(m, "CallNode")
            .def(py::init<Location,const std::string &,std::vector<std::shared_ptr<Expression>>,Type>());

    py::class_<VarNode, std::shared_ptr<VarNode>, DeclareValNode>(m, "VarNode")
            .def(py::init<Location, std::string, Type,std::shared_ptr<Expression>>());

    py::class_<GetValNode, std::shared_ptr<GetValNode>, Expression>(m, "GetValNode")
            .def(py::init<Location, std::string, Type>());

    py::class_<BlockNode, std::shared_ptr<BlockNode>, Node>(m, "BlockNode")
            .def(py::init<Location, NodeList>());

    py::class_<ReturnNode, std::shared_ptr<ReturnNode>, Node>(m, "ReturnNode")
            .def(py::init<Location, std::shared_ptr<Expression>>());

    m.def("make_block_ptr", [](Location l, NodeList n) {
        return std::make_shared<BlockNode>(l,n); }
        );
    py::class_<PythonFunction, std::shared_ptr<PythonFunction>, Node>(m, "PythonFunction")
            .def(py::init<Location, const std::string &, Type,ParamList,std::shared_ptr<BlockNode>>());

    py::class_<PythonModule, std::shared_ptr<PythonModule>>(m, "PythonModule").def(py::init<FunctionList>());
}
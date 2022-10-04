//
// Created by Steffi Stumpos on 7/15/22.
//
#include "PaicAST.h"
#include "pybind_utils.h"
namespace py = pybind11;

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);
using ExpList = std::vector<std::shared_ptr<paic2::Expression>, std::allocator<std::shared_ptr<paic2::Expression>>>;
PYBIND11_MAKE_OPAQUE(ExpList);
using NodeList = std::vector<std::shared_ptr<paic2::Node>, std::allocator<std::shared_ptr<paic2::Node>>>;
PYBIND11_MAKE_OPAQUE(NodeList);
using ParamList = std::vector<std::shared_ptr<paic2::ParamNode>, std::allocator<std::shared_ptr<paic2::ParamNode>>>;
PYBIND11_MAKE_OPAQUE(ParamList);
using FunctionList = std::vector<std::shared_ptr<paic2::PythonFunction>, std::allocator<std::shared_ptr<paic2::PythonFunction>>>;
PYBIND11_MAKE_OPAQUE(FunctionList);

void paic2::Node::bind(pybind11::module &m) {
    py::class_<Location>(m, "Location").def(py::init<int,int>());
    py::enum_<TypeKind>(m, "TypeKind")
            .value("Primitive", paic2::TypeKind::Primitive)
            .value("World", paic2::TypeKind::World).export_values();
    py::class_<Type, std::shared_ptr<Type>>(m, "Type").def(py::init<TypeKind>());

    py::enum_<PrimitiveCode>(m, "PrimitiveCode")
            .value("Void", paic2::PrimitiveCode::Void)
            .value("Double", paic2::PrimitiveCode::Double)
            .value("Float", paic2::PrimitiveCode::Float).export_values();
    py::class_<PrimitiveType, std::shared_ptr<PrimitiveType>, Type>(m, "PrimitiveType")
            .def(py::init<PrimitiveCode>());
    py::class_<WorldType, std::shared_ptr<WorldType>, Type>(m, "WorldType")
            .def(py::init<PrimitiveCode, int>());

    bind_vector<std::shared_ptr<paic2::Node>>(m, "NodeList");
    bind_vector<std::shared_ptr<paic2::ParamNode>>(m, "ParamList");
    bind_vector<std::shared_ptr<paic2::PythonFunction>>(m, "FunctionList");
    bind_vector<std::shared_ptr<paic2::Expression>>(m, "ExpList");

    // NODES
    py::class_<Node, std::shared_ptr<Node>>(m, "Node")
            .def(py::init<Location, NodeKind>())
            .def("loc", &Node::loc);

    py::class_<Expression, std::shared_ptr<Expression>, Node>(m, "Expression")
            .def(py::init<Location,NodeKind,std::shared_ptr<Type>>()).def("type", &paic2::Expression::getType);

    py::class_<FloatConstNode, std::shared_ptr<FloatConstNode>, Expression>(m, "FloatNode")
            .def(py::init<Location, float>());

    py::class_<DeclareValNode, std::shared_ptr<DeclareValNode>, Node>(m, "DeclareValNode")
            .def(py::init<Location,std::string, NodeKind, std::shared_ptr<Type>>())
            .def("type", &paic2::DeclareValNode::getType)
            .def("name", &paic2::DeclareValNode::getPyName);

    py::class_<ParamNode, std::shared_ptr<ParamNode>, DeclareValNode>(m, "ParamNode")
            .def(py::init<Location,std::string, std::shared_ptr<Type>>())
            .def("name", &paic2::DeclareValNode::getPyName)
            .def("type", &paic2::DeclareValNode::getType);

    py::class_<CallNode, std::shared_ptr<CallNode>, Expression>(m, "CallNode")
            .def(py::init<Location, const std::string &,std::vector<std::shared_ptr<Expression>>,std::shared_ptr<Expression>,std::shared_ptr<Type>>())
            .def(py::init<Location, const std::string &,std::vector<std::shared_ptr<Expression>>,std::shared_ptr<Type>>());

    py::class_<VarNode, std::shared_ptr<VarNode>, DeclareValNode>(m, "VarNode")
            .def(py::init<Location, std::string, std::shared_ptr<Type>,std::shared_ptr<Expression>>());

    py::class_<GetValNode, std::shared_ptr<GetValNode>, Expression>(m, "GetValNode")
            .def(py::init<Location, std::string, std::shared_ptr<Type>>());

    py::class_<BlockNode, std::shared_ptr<BlockNode>, Node>(m, "BlockNode")
            .def(py::init<Location, NodeList>());

    py::class_<ReturnNode, std::shared_ptr<ReturnNode>, Node>(m, "ReturnNode")
            .def(py::init<Location, std::shared_ptr<Expression>>())
            .def("value", &paic2::ReturnNode::getValue);

    py::class_<PythonFunction, std::shared_ptr<PythonFunction>, Node>(m, "PythonFunction")
            .def(py::init<Location, const std::string &, std::shared_ptr<Type>,ParamList,std::shared_ptr<BlockNode>>());

    py::class_<PythonModule, std::shared_ptr<PythonModule>>(m, "PythonModule").def(py::init<FunctionList>());
}
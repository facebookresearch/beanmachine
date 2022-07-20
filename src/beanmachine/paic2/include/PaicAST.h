//
// Created by Steffi Stumpos on 7/15/22.
//

#ifndef PAIC2_PAICAST_H
#define PAIC2_PAICAST_H

#include <vector>
#include <string>
#include <iostream>
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
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

    enum NodeKind {
        Variable,
        Parameter,
        Constant,
        Call,
        Function,
        Block,
        Return,
        GetVal
    };

    class Type {
    public:
        Type(llvm::StringRef name):_name(name){}
        llvm::StringRef getName() const { return _name; }
    private:
        std::string _name;
    };

    class Node {
    public:
        static void bind(pybind11::module &m);
        Node(Location location, NodeKind kind):_location(std::move(location)), _kind(kind) {}
        virtual ~Node() = default;
        const Location &loc() { return _location; }
        const NodeKind getKind() const {return _kind;}
    private:
        Location _location;
        NodeKind _kind;
    };

    class DeclareValNode : public Node {
    public:
        DeclareValNode(Location loc, llvm::StringRef name, NodeKind kind, Type type):Node(std::move(loc), kind), _name(name),_type(std::move(type)){}
        std::string getPyName() { return _name; }
        llvm::StringRef getName() const { return _name; }
        const Type &getType() { return _type; }
    private:
        std::string _name;
        Type _type;
    };

    class ParamNode : public DeclareValNode {
    public:
        ParamNode(Location loc, llvm::StringRef name, Type type): DeclareValNode(std::move(loc), name, Parameter, std::move(type)){}
    };

    class Expression : public Node {
    public:
        Expression(Location loc,NodeKind kind, Type type):Node(std::move(loc), kind),_type(std::move(type)){}
        const Type &getType() { return _type; }
    private:
        Type _type;
    };

    class VarNode : public DeclareValNode {
    public:
        VarNode(Location loc, llvm::StringRef name, Type type,
                std::shared_ptr<Expression> initVal): DeclareValNode(std::move(loc), name, Variable, std::move(type)),_rhs(std::move(initVal)){}
        std::shared_ptr<Expression> getInitVal() { return _rhs; }
        static bool classof(const Node*c) { return c->getKind() == paic2::NodeKind::Variable; }
    private:
        std::shared_ptr<Expression> _rhs;
    };

    class GetValNode : public Expression {
    public:
        GetValNode(Location loc, llvm::StringRef name, Type type)
                : Expression(std::move(loc), NodeKind::GetVal, std::move(type)), _name(name) {}

        std::string getPyName() { return _name; }
        llvm::StringRef getName() const { return _name; }
    private:
        std::string _name;
    };

    class CallNode : public Expression {
    public:
        CallNode(Location loc,
                 const std::string &callee,
                 std::vector<std::shared_ptr<Expression>> args,
                 Type type)
                : Expression(std::move(loc), NodeKind::Call, std::move(type)), _function(callee),
                  args(args), _receiver(nullptr) {}
        CallNode(Location loc,
                 const std::string &callee,
                 std::vector<std::shared_ptr<Expression>> args,
                 std::shared_ptr<Expression> receiver,
                 Type type)
                : Expression(std::move(loc), NodeKind::Call, std::move(type)), _function(callee),
                  args(args), _receiver(receiver) {}

        llvm::StringRef getCallee() { return _function; }
        std::shared_ptr<Expression> getReceiver() { return _receiver; }
        llvm::ArrayRef<std::shared_ptr<Expression>> getArgs() { return args; }
        static bool classof(const Node*c) { return c->getKind() == paic2::NodeKind::Call; }
    private:
        std::string _function;
        std::shared_ptr<Expression> _receiver;
        std::vector<std::shared_ptr<Expression>> args;
    };

    class BlockNode : public Node {
    public:
        BlockNode(Location location, std::vector<std::shared_ptr<Node>> args):Node(std::move(location), Block), _children(std::move(args)){}
        virtual ~BlockNode(){}
        llvm::ArrayRef<std::shared_ptr<Node>> getChildren() { return _children; }
    private:
        std::vector<std::shared_ptr<Node>> _children;
    };

    template<typename T>
    class ConstNode : public Expression {
    public:
        ConstNode(Location location, std::string type_name, T value): Expression(location, Constant, Type(type_name)), _value(value){}
        T getValue() { return _value; }
    private:
        T _value;
    };

    class FloatConstNode : public ConstNode<float> {
    public:
        FloatConstNode(Location location, float value): ConstNode<float>(location, "float", Constant){}
    };

    class ReturnNode : public Node {
    public:
        ReturnNode(Location location, std::shared_ptr<Expression> value): Node(std::move(location), Return), _value(value){}
        std::shared_ptr<Expression> getValue() { return _value;}
        static bool classof(const Node*c) { return c->getKind() == paic2::NodeKind::Return; }
    private:
        std::shared_ptr<Expression> _value;
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

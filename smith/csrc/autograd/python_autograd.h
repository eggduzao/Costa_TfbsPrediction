#ifndef THP_AUTOGRAD_H
#define THP_AUTOGRAD_H
#include <smith/csrc/utils/pythoncapi_compat.h>

PyObject* THPAutograd_initExtension(PyObject* _unused, PyObject* unused);
void THPAutograd_initFunctions();

namespace smith::autograd {

PyMethodDef* python_functions();

}

#include <smith/csrc/autograd/python_engine.h>
#include <smith/csrc/autograd/python_function.h>
#include <smith/csrc/autograd/python_variable.h>

#endif

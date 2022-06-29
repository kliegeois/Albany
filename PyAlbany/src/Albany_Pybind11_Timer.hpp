#ifndef PYALBANY_TIMER_H
#define PYALBANY_TIMER_H

#include "Teuchos_StackedTimer.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using RCP_StackedTimer = Teuchos::RCP<Teuchos::StackedTimer>;
using RCP_Time = Teuchos::RCP<Teuchos::Time>;

RCP_Time createRCPTime(const std::string name);

void pyalbany_time(pybind11::module &m);

#endif

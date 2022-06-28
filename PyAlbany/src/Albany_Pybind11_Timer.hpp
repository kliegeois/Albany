#ifndef PYALBANY_TIMER_H
#define PYALBANY_TIMER_H

#include "Teuchos_StackedTimer.hpp"

using RCP_StackedTimer = Teuchos::RCP<Teuchos::StackedTimer>;
using RCP_Time = Teuchos::RCP<Teuchos::Time>;

RCP_Time createRCPTime(const std::string name) {
    return Teuchos::rcp<Teuchos::Time>(new Teuchos::Time(name));
}

#endif

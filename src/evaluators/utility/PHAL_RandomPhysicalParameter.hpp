//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_RANDOMPHYSICALPARAMETER_HPP
#define PHAL_RANDOMPHYSICALPARAMETER_HPP

#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_MDField.hpp"
#include "Phalanx_config.hpp"
#include "Teuchos_ParameterList.hpp"

#include "PHAL_AlbanyTraits.hpp"
#include "PHAL_SharedParameter.hpp"

namespace PHAL {
///
/// RandomPhysicalParameter
///
template<typename EvalT, typename Traits, typename ParamNameEnum, ParamNameEnum ParamName>
class RandomPhysicalParameter : public SharedParameter<EvalT, Traits, ParamNameEnum, ParamName>
{
 public:
  typedef typename EvalT::ScalarT   ScalarT;
  typedef ParamNameEnum             EnumType;

  RandomPhysicalParameter (const Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl)
  {
    if(p.isParameter("Is random") && p.get<bool>("Is random"))
    {
      is_random = true;
      param_name   = p.get<std::string>("Parameter Name");
      param_as_field = PHX::MDField<ScalarT,Dim>(param_name,dl->shared_param);

      // Never actually evaluated, but creates the evaluation tag
      this->addEvaluatedField(param_as_field);

      // Sacado-ized parameter
      Teuchos::RCP<ParamLib> paramLib = p.get<Teuchos::RCP<ParamLib>>("Parameter Library");
      this->setName("Shared Parameter " + param_name + PHX::print<EvalT>());

      const Teuchos::ParameterList* paramsList = p.get<const Teuchos::ParameterList*>("Parameters List");

      // Find the parameter in the Paramter list,
      // register as a Sacado Parameter and set the Nominal value
      bool nominalValueSet = false;
      if((paramsList != NULL) && paramsList->isParameter("Number Of Parameters"))
      {
        int n = paramsList->get<int>("Number Of Parameters");
        for (int i=0; (nominalValueSet==false) && i<n; ++i)
        {
          const Teuchos::ParameterList& pvi = paramsList->sublist(Albany::strint("Parameter",i));
          std::string parameterType = "Scalar";
          if(pvi.isParameter("Type"))
            parameterType = pvi.get<std::string>("Type");
          if (parameterType == "Distributed")
            break; // Pointless to check the remaining parameters as they are all distributed

          if (parameterType == "Scalar") {
            if (pvi.get<std::string>("Name")==param_name)
            {
              this->registerSacadoParameter(param_name, paramLib);
              if (pvi.isParameter("Nominal Value")) {
              double nom_val = pvi.get<double>("Nominal Value");
              value = nom_val;
              nominalValueSet = true;
            }
            break;
            }
          }
          else { //"Vector"
            int m = pvi.get<int>("Dimension");
            for (int j=0; j<m; ++j)
            {
              const Teuchos::ParameterList& pj = pvi.sublist(Albany::strint("Scalar",j));
              if (pj.get<std::string>("Name")==param_name)
              {
                this->registerSacadoParameter(param_name, paramLib);
                if (pj.isParameter("Nominal Value")) {
                  double nom_val = pj.get<double>("Nominal Value");
                  value = nom_val;
                  nominalValueSet = true;
                }
                break;
              }
            }
          }
        }
      }

      if(!nominalValueSet) 
        value = p.get<double>("Default Nominal Value");

      dummy = 0;
    }
    else {
      is_random = false;
      SharedParameter<EvalT, Traits, ParamNameEnum, ParamName>::SharedParameter(p, dl);
    }
  }

  void postRegistrationSetup(typename Traits::SetupData d, PHX::FieldManager<Traits>& fm)
  {
    if (is_random) {
      this->utils.setFieldData(param_as_field,fm);
      d.fill_field_dependencies(this->dependentFields(),this->evaluatedFields(),false);
    }
    else {
      SharedParameter<EvalT, Traits, ParamNameEnum, ParamName>::postRegistrationSetup(p, fm);
    }
  }

  static ScalarT getValue ()
  {
    if (is_random) {
      return value;
    }
    else {
      return SharedParameter<EvalT, Traits, ParamNameEnum, ParamName>::getValue();
    }
  }

  ScalarT& getValue(const std::string &n)
  {
    if (is_random) {
      if (n==param_name)
        return value;

      return dummy;
    }
    else {
      return SharedParameter<EvalT, Traits, ParamNameEnum, ParamName>::getValue(n);
    }
  }

  void evaluateFields(typename Traits::EvalData d)
  {
    if (is_random) {
      param_as_field(0) = value;
    }
    else {
      SharedParameter<EvalT, Traits, ParamNameEnum, ParamName>::evaluateFields(d);
    }
  }

protected:

  static ScalarT              value;
  static ScalarT              dummy;
  static std::string          param_name;
  static double               nominal_value;
  static bool                 is_parameter;
  static bool                 is_random;

  PHX::MDField<ScalarT,Dim>   param_as_field;
};

template<typename EvalT, typename Traits, typename ParamNameEnum, ParamNameEnum ParamName>
typename EvalT::ScalarT RandomPhysicalParameter<EvalT,Traits,ParamNameEnum,ParamName>::value;

template<typename EvalT, typename Traits, typename ParamNameEnum, ParamNameEnum ParamName>
typename EvalT::ScalarT RandomPhysicalParameter<EvalT,Traits,ParamNameEnum,ParamName>::dummy;

template<typename EvalT, typename Traits, typename ParamNameEnum, ParamNameEnum ParamName>
std::string RandomPhysicalParameter<EvalT,Traits,ParamNameEnum,ParamName>::param_name;

}  // Namespace PHAL

#endif  // PHAL_READSTATEFIELD_HPP

#ifndef PHAL_SHARED_PARAMETER_VECTOR_HPP
#define PHAL_SHARED_PARAMETER_VECTOR_HPP 1

#include "PHAL_Dimension.hpp"
#include "Albany_SacadoTypes.hpp"
#include "Albany_Utils.hpp"

#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Sacado_ParameterAccessor.hpp"

//IKT 6/3/2020 TODO: implement support for vector parameters, which is not available currently.

namespace PHAL
{

template<typename EvalT, typename Traits, typename ParamNameEnum, ParamNameEnum ParamName>
class SharedParameterVector : public PHX::EvaluatorWithBaseImpl<Traits>,
                        public PHX::EvaluatorDerived<EvalT, Traits>,
                        public Sacado::ParameterAccessor<EvalT, SPL_Traits>
{
public:

  typedef typename EvalT::ScalarT   ScalarT;
  typedef ParamNameEnum             EnumType;

  SharedParameterVector (const Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl)
  {
    param_name   = p.get<std::string>("Parameter Name");
    vecDim       = p.get<int>("Vector Dimension");
    dl->shared_param_vec = Teuchos::rcp(new PHX::MDALayout<Dim>(vecDim));
    param_as_field = PHX::MDField<ScalarT,Dim>(param_name, dl->shared_param_vec);

    value = new ScalarT[vecDim];

    // Never actually evaluated, but creates the evaluation tag
    this->addEvaluatedField(param_as_field);

    // Sacado-ized parameter
    Teuchos::RCP<ParamLib> paramLib = p.get<Teuchos::RCP<ParamLib>>("Parameter Library");
    this->setName("Shared Parameter Vector " + param_name + PHX::print<EvalT>());

    const Teuchos::ParameterList* paramsList = p.get<const Teuchos::ParameterList*>("Parameters List");

    for (int i_vec=0; i_vec<vecDim; ++i_vec) {
      std::string param_name_i = Albany::strint(param_name,i_vec);
      // Find the parameter in the Paramter list,
      // register as a Sacado Parameter and set the Nominal value
      bool nominalValueSet = false;
      log_parameter = false;
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
            if (pvi.get<std::string>("Name")==param_name_i)
            {
              this->registerSacadoParameter(param_name_i, paramLib);
              if (pvi.isParameter("Nominal Value")) {
                double nom_val = pvi.get<double>("Nominal Value");
                value[i_vec] = nom_val;
                nominalValueSet = true;
              }
              if (pvi.isParameter("Log Of Physical Parameter")) {
                log_parameter = pvi.get<bool>("Log Of Physical Parameter");
              }
            break;
            }
          }
          else { //"Vector"
            int m = pvi.get<int>("Dimension");
            for (int j=0; j<m; ++j)
            {
              const Teuchos::ParameterList& pj = pvi.sublist(Albany::strint("Scalar",j));
              if (pj.get<std::string>("Name")==param_name_i)
              {
                this->registerSacadoParameter(param_name_i, paramLib);
                if (pj.isParameter("Nominal Value")) {
                  double nom_val = pj.get<double>("Nominal Value");
                  value[i_vec] = nom_val;
                  nominalValueSet = true;
                }
                if (pj.isParameter("Log Of Physical Parameter")) {
                  log_parameter = pj.get<bool>("Log Of Physical Parameter");
                }
                break;
              }
            }
          }
        }
      }

      if(!nominalValueSet) 
        value[i_vec] = p.get<double>("Default Nominal Value");

    }
    dummy = 0;
  }

  void postRegistrationSetup(typename Traits::SetupData d, PHX::FieldManager<Traits>& fm)
  {
    this->utils.setFieldData(param_as_field,fm);
    d.fill_field_dependencies(this->dependentFields(),this->evaluatedFields(),false);
  }

  static ScalarT getValue ()
  {
    return value[0];
  }

  static ScalarT getValue (int i)
  {
    return value[i];
  }

  ScalarT& getValue(const std::string &n)
  {
    if (n==param_name)
      return value[0];

    return dummy;
  }

  void evaluateFields(typename Traits::EvalData /*d*/)
  {
    for (int i_vec=0; i_vec<vecDim; ++i_vec) {
      if (log_parameter) {
        param_as_field(i_vec) = std::exp(value[i_vec]);
      } else {
        param_as_field(i_vec) = value[i_vec];
      }
    }
  }

protected:

  static ScalarT*             value;
  static ScalarT              dummy;
  static std::string          param_name;
  static bool                 log_parameter;
  static int                  vecDim;

  PHX::MDField<ScalarT,Dim>   param_as_field;
};

template<typename EvalT, typename Traits, typename ParamNameEnum, ParamNameEnum ParamName>
typename EvalT::ScalarT* SharedParameterVector<EvalT,Traits,ParamNameEnum,ParamName>::value;

template<typename EvalT, typename Traits, typename ParamNameEnum, ParamNameEnum ParamName>
typename EvalT::ScalarT SharedParameterVector<EvalT,Traits,ParamNameEnum,ParamName>::dummy;

template<typename EvalT, typename Traits, typename ParamNameEnum, ParamNameEnum ParamName>
std::string SharedParameterVector<EvalT,Traits,ParamNameEnum,ParamName>::param_name;

template<typename EvalT, typename Traits, typename ParamNameEnum, ParamNameEnum ParamName>
bool SharedParameterVector<EvalT,Traits,ParamNameEnum,ParamName>::log_parameter;

template<typename EvalT, typename Traits, typename ParamNameEnum, ParamNameEnum ParamName>
int SharedParameterVector<EvalT,Traits,ParamNameEnum,ParamName>::vecDim;

} // Namespace PHAL

#endif // PHAL_SHARED_PARAMETER_VECTOR_HPP

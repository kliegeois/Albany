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

template<typename EvalT, typename Traits>
class SharedParameterVector : public PHX::EvaluatorWithBaseImpl<Traits>,
                        public PHX::EvaluatorDerived<EvalT, Traits>,
                        public Sacado::ParameterAccessor<EvalT, SPL_Traits>
{
public:

  typedef typename EvalT::ScalarT   ScalarT;

  SharedParameterVector (const Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl)
  {
    param_name   = p.get<std::string>("Parameter Name");
    vecDim       = p.get<int>("Vector Dimension");
    param_as_field = PHX::MDField<ScalarT,Dim>(param_name, rcp(new MDALayout<Dim>(vecDim)));

    value = ScalarT[vecDim];

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
        value = p.get<double>("Default Nominal Value");

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
    return value;
  }

  ScalarT& getValue(const std::string &n)
  {
    if (n==param_name)
      return value;

    return dummy;
  }

  void evaluateFields(typename Traits::EvalData /*d*/)
  {
    if (log_parameter) {
      param_as_field(0) = std::exp(value);
    } else {
      param_as_field(0) = value;
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

template<typename EvalT, typename Traits>
typename EvalT::ScalarT* SharedParameterVector<EvalT,Traits>::value;

template<typename EvalT, typename Traits>
typename EvalT::ScalarT SharedParameterVector<EvalT,Traits>::dummy;

template<typename EvalT, typename Traits>
std::string SharedParameterVector<EvalT,Traits>::param_name;

template<typename EvalT, typename Traits>
bool SharedParameterVector<EvalT,Traits>::log_parameter;

} // Namespace PHAL

#endif // PHAL_SHARED_PARAMETER_VECTOR_HPP

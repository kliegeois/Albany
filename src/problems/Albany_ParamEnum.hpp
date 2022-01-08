#ifndef ALBANY_PARAM_ENUM_HPP
#define ALBANY_PARAM_ENUM_HPP

#include <string>

namespace Albany
{

enum class ParamEnum
{
  Kappa_x      = 0, //For thermal problem
  Kappa_y      = 1, //For thermal problem
  Kappa_z      = 2, //For thermal problem
  Theta_0      = 3,
  Theta_1      = 4,
  Coeff_0      = 5,
  Coeff_1      = 6,
  Coeff_2      = 7,
  Coeff_3      = 8,
  Coeff_4      = 9,
  Coeff_5      = 10,
  Coeff_6      = 11,
  Coeff_7      = 12,
  Coeff_8      = 13,
  Coeff_9      = 14,
};

namespace ParamEnumName
{
  static const std::string kappa_x       = "kappa_x Parameter"; //For thermal problem
  static const std::string kappa_y       = "kappa_y Parameter"; //For thermal problem
  static const std::string kappa_z       = "kappa_z Parameter"; //For thermal problem
  static const std::string theta_0       = "Theta 0"; 
  static const std::string theta_1       = "Theta 1";
  static const std::string coeff_0       = "Coefficient 0";
  static const std::string coeff_1       = "Coefficient 1";
  static const std::string coeff_2       = "Coefficient 2";
  static const std::string coeff_3       = "Coefficient 3";
  static const std::string coeff_4       = "Coefficient 4";
  static const std::string coeff_5       = "Coefficient 5";
  static const std::string coeff_6       = "Coefficient 6";
  static const std::string coeff_7       = "Coefficient 7";
  static const std::string coeff_8       = "Coefficient 8";
  static const std::string coeff_9       = "Coefficient 9";

} // ParamEnum

} // Namespace Albany

#endif // ALBANY_PARAM_ENUM_HPP

#define NO_IMPORT_ARRAY
#include "numpy_include.hpp"

template< typename TYPE >
int NumPy_TypeCode();

template<>
int NumPy_TypeCode< bool >()
{
  return NPY_BOOL;
}

template<>
int NumPy_TypeCode< char >()
{
  return NPY_BYTE;
}

template<>
int NumPy_TypeCode< unsigned char >()
{
  return NPY_UBYTE;
}

template<>
int NumPy_TypeCode< short >()
{
  return NPY_SHORT;
}

template<>
int NumPy_TypeCode< unsigned short >()
{
  return NPY_USHORT;
}

template<>
int NumPy_TypeCode< int >()
{
  return NPY_INT;
}

template<>
int NumPy_TypeCode< unsigned int >()
{
  return NPY_UINT;
}

template<>
int NumPy_TypeCode< long >()
{
  return NPY_LONG;
}

template<>
int NumPy_TypeCode< unsigned long >()
{
  return NPY_ULONG;
}

template<>
int NumPy_TypeCode< long long >()
{
  return NPY_LONGLONG;
}

template<>
int NumPy_TypeCode< unsigned long long >()
{
  return NPY_ULONGLONG;
}

template<>
int NumPy_TypeCode< float >()
{
  return NPY_FLOAT;
}

template<>
int NumPy_TypeCode< double >()
{
  return NPY_DOUBLE;
}

template<>
int NumPy_TypeCode< long double >()
{
  return NPY_LONGDOUBLE;
}

template<>
int NumPy_TypeCode< std::complex< float > >()
{
  return NPY_CFLOAT;
}

template<>
int NumPy_TypeCode< std::complex< double > >()
{
  return NPY_CDOUBLE;
}

template<>
int NumPy_TypeCode< std::complex< long double > >()
{
  return NPY_CLONGDOUBLE;
}

template<>
int NumPy_TypeCode< char * >()
{
  return NPY_STRING;
}

template<>
int NumPy_TypeCode< std::string >()
{
  return NPY_STRING;
}
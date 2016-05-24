#pragma once

#include <algorithm>
#include <iterator>

#include <Eigen/Core>

namespace alps {
  namespace fastupdate {

    //return permutation from time-ordering (1 or -1)
    template <typename InputIterator>
    int permutation(InputIterator begin, InputIterator end) {
      using std::swap;
      typedef typename std::iterator_traits<InputIterator>::value_type my_value_type;

      std::vector<my_value_type> values;
      std::copy(begin, end, std::back_inserter(values));

      const int N = values.size();
      int perm = 1;
      while (true) {
        bool exchanged = false;
        for (int i=0; i<N-1; ++i) {
          if ( !(values[i]<values[i+1]) ) {
            swap(values[i], values[i+1]);
            perm *= -1;
            exchanged = true;
          }
        }
        if (!exchanged) break;
      }
      return perm;
    }

    template<typename Derived>
    inline typename Derived::RealScalar max_abs_coeff(const Eigen::MatrixBase<Derived>& mat) {
      typedef typename Derived::RealScalar RealScalar;
      const int rows = mat.rows();
      const int cols = mat.cols();

      RealScalar result = 0.0;
      for (int j=0; j<cols; ++j) {
        for (int i=0; i<rows; ++i) {
          result = std::max(result, std::abs(mat(i,j)));
        }
      }

      return result;
    }

    //Compute the determinant of a matrix avoiding underflow and overflow
    //Note: This make a copy of the matrix.
    template<typename Derived>
    typename Derived::Scalar
    safe_determinant(const Eigen::MatrixBase<Derived>& mat) {
      typedef typename Derived::RealScalar RealScalar;
      assert(mat.rows()==mat.cols());
      const int N = mat.rows();
      if (N==0) {
        return 1.0;
      }
      Eigen::Matrix<typename Derived::Scalar,Eigen::Dynamic,Eigen::Dynamic> mat_copy(mat);
      const RealScalar max_coeff = mat_copy.cwiseAbs().maxCoeff();
      if (max_coeff==0.0) {
        return 0.0;
      }
      mat_copy /= max_coeff;
      return mat_copy.determinant()*std::pow(max_coeff, 1.*N);
    }

    //Compute the determinant of a matrix avoiding underflow and overflow
    //Note: This make a copy of the matrix.
    template<typename Derived>
    typename Derived::Scalar
    safe_determinant_eigen_block(const Eigen::Block<const Derived>& mat) {
      typedef typename Derived::Scalar Scalar;
      typedef typename Derived::RealScalar RealScalar;

      assert(mat.rows()==mat.cols());
      const int N = mat.rows();
      if (N==0) {
        return 1.0;
      }
      Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> mat_copy(mat);
      const RealScalar max_coeff = mat_copy.cwiseAbs().maxCoeff();
      if (max_coeff==0.0) {
        return 0.0;
      }
      mat_copy /= max_coeff;
      return mat_copy.determinant()*std::pow(max_coeff, 1.*N);
    }

    //Compute the inverse of a matrix avoiding underflow and overflow
    //Note: This make a copy of the matrix.
    template<typename Derived>
    inline
    void
    safe_invert_in_place(Eigen::MatrixBase<Derived>& mat) {
      typedef typename Derived::RealScalar RealScalar;

      const int N = mat.rows();
      const RealScalar max_coeff = mat.cwiseAbs().maxCoeff();

      Eigen::Matrix<typename Derived::Scalar,Eigen::Dynamic,Eigen::Dynamic> mat_copy = mat/max_coeff;
      mat = mat_copy.inverse()/max_coeff;
    }

    //Compute the inverse of a matrix avoiding underflow and overflow
    //Note: This make a copy of the matrix.
    template<typename Derived>
    Eigen::Matrix<typename Derived::Scalar, Eigen::Dynamic, Eigen::Dynamic>
    safe_inverse(const Eigen::MatrixBase<Derived>& mat) {
      typedef typename Derived::RealScalar RealScalar;

      const int N = mat.rows();
      const RealScalar max_coeff = mat.cwiseAbs().maxCoeff();

      Eigen::Matrix<typename Derived::Scalar,Eigen::Dynamic,Eigen::Dynamic> mat_copy = mat/max_coeff;
      return mat_copy.inverse()/max_coeff;
    }

    //Compute the inverse of a matrix avoiding underflow and overflow
    //Note: This make a copy of the matrix.
    template<typename Derived>
    Eigen::Matrix<typename Derived::Scalar, Eigen::Dynamic, Eigen::Dynamic>
    safe_inverse(const Eigen::Block<const Derived>& mat) {
      typedef typename Derived::RealScalar RealScalar;

      const int N = mat.rows();
      const RealScalar max_coeff = mat.cwiseAbs().maxCoeff();

      Eigen::Matrix<typename Derived::Scalar,Eigen::Dynamic,Eigen::Dynamic> mat_copy = mat/max_coeff;
      return mat_copy.inverse()/max_coeff;
    }

    //Compute the inverse of a matrix avoiding underflow and overflow
    //Note: This make a copy of the matrix.
    template<typename Derived>
    Eigen::Matrix<typename Derived::Scalar, Eigen::Dynamic, Eigen::Dynamic>
    safe_inverse(Eigen::Block<Derived>& mat) {
      typedef typename Derived::RealScalar RealScalar;

      const int N = mat.rows();
      const RealScalar max_coeff = mat.cwiseAbs().maxCoeff();

      Eigen::Matrix<typename Derived::Scalar,Eigen::Dynamic,Eigen::Dynamic> mat_copy = mat/max_coeff;
      return mat_copy.inverse()/max_coeff;
    }


  }
}

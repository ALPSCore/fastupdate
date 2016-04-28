#include <complex>
#include <algorithm>
#include <limits>
#include <functional>

#include <boost/random.hpp>
#include <boost/math/special_functions/binomial.hpp>
#include <boost/random.hpp>
#include <boost/multi_array.hpp>
#include <boost/range/irange.hpp>

#include "gtest.h"

#include "fastupdate_formula.hpp"

template<typename Derived>
inline int num_cols(const Eigen::MatrixBase<Derived> &m) {
  return m.cols();
}

template<typename Derived>
inline int num_rows(const Eigen::MatrixBase<Derived> &m) {
  return m.rows();
}


template<class M>
void randomize_matrix(M& mat, size_t seed=100) {
  boost::random::mt19937 gen;
  boost::random::uniform_01<double> dist;
  gen.seed(seed);

  for (int j=0; j<num_cols(mat); ++j) {
    for (int i=0; i<num_rows(mat); ++i) {
      mat(i,j) = dist(gen);
    }
  }
}

//std::random_shuffle
class rs_shuffle : public std::unary_function<unsigned int, int unsigned> {
public:
  rs_shuffle(boost::mt19937 &gen) : gen_(gen) {};
  unsigned int operator()(unsigned int N) {
    boost::uniform_int<> dist(0,N-1);
    return dist(gen_);
  }

private:
  boost::random::mt19937 gen_;
};


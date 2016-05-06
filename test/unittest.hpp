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

#include "determinant_matrix.hpp"
#include "fastupdate_formula.hpp"
#include "util.hpp"

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

//for std::random_shuffle
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

/*creation operator class*/
class c_or_cdagger
{
public:
  typedef double itime_type;
  c_or_cdagger() : flavor_(-1), t_(-1.0) {};
  c_or_cdagger(int flavor, double t)
  {
    flavor_ = flavor;
    t_ = t;
  }
  virtual ~c_or_cdagger() {}

  double time() const {return t_;}
  int flavor() const {return flavor_;}

private:
  int flavor_;
  double t_;
};

inline double operator_time(const c_or_cdagger &op) {
  return op.time();
}

inline int operator_flavor(const c_or_cdagger &op) {
  return op.flavor();
}

inline bool operator<(const c_or_cdagger& op1, const c_or_cdagger& op2) {
  return operator_time(op1) < operator_time(op2);
}

class creator : public c_or_cdagger {
public:
  creator() : c_or_cdagger() {};
  creator(int flavor, double t) : c_or_cdagger(flavor, t) {};
};

class annihilator : public c_or_cdagger {
public:
  annihilator() : c_or_cdagger() {};
  annihilator(int flavor, double t) : c_or_cdagger(flavor, t) {};
};

//Interpolation of G0
template<typename T>
struct OffDiagonalG0 {
  OffDiagonalG0 (double beta, int n_flavor, const std::vector<double>& E, const boost::multi_array<T,2>& phase) : beta_(beta), n_flavor_(n_flavor), E_(E), phase_(phase) {}

  int nflavor() const {return n_flavor_;}

  T operator() (const annihilator& c_op, const creator& cdagg_op) const {
    const double dt = c_op.time()-cdagg_op.time();
    double dt_tmp = dt;
    if (dt_tmp > beta_) dt_tmp -= beta_;
    if (dt_tmp < 0) dt_tmp += beta_;

    if (c_op.flavor()==cdagg_op.flavor()) {
      const double E_tmp = E_[c_op.flavor()];
      return  -std::exp((beta_-dt_tmp)*E_tmp)/(1.0+std::exp(beta_*E_tmp));
    } else {
      return (-dt+0.5*beta_)/(2*beta_)*phase_[c_op.flavor()][cdagg_op.flavor()];
    }
  }

  double beta_;
  int n_flavor_;
  std::vector<double> E_;
  boost::multi_array<T,2> phase_;
};


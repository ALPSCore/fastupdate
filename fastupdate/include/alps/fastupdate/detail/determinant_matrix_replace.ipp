#include "../determinant_matrix.hpp"

namespace alps {
  namespace fastupdate {

    template<
      typename Scalar,
      typename GreensFunction,
      typename CdaggerOp,
      typename COp
    >
    Scalar
    DeterminantMatrix<Scalar,GreensFunction,CdaggerOp,COp>::try_replace_cdagg(
            const CdaggerOp& old_cdagg,
            const CdaggerOp& new_cdagg
    ) {
      check_state(waiting);
      state_ = try_replace_cdagg__called;
      const int nop = inv_matrix_.size1();
      new_cdagg_ = new_cdagg;
      old_cdagg_ = old_cdagg;

      //move the target row to the end
      const int pos = find_cdagg(old_cdagg);
      if (pos != nop) {
        swap_cdagg_op(pos, nop-1);
      }

      //compute the values of new elements
      G_j_n_.resize(nop, 1);
      for(int i=0; i<nop; ++i) {
        G_j_n_(i, 0) = gf_(c_ops_[i], new_cdagg);
      }

      det_rat_ = compute_det_ratio_relace_last_col(inv_matrix_, G_j_n_);
      return det_rat_;
    }

    template<
      typename Scalar,
      typename GreensFunction,
      typename CdaggerOp,
      typename COp
    >
    void
    DeterminantMatrix<Scalar,GreensFunction,CdaggerOp,COp>::perform_replace_cdagg() {
      check_state(try_replace_cdagg__called);
      state_ = waiting;
      const int nop = inv_matrix_.size1();
      compute_inverse_matrix_replace_last_col(inv_matrix_, G_j_n_, det_rat_);
      const int diff =
        std::abs(
              std::distance(cdagg_op_pos_.lower_bound(operator_time(old_cdagg_)), cdagg_op_pos_.end())-
              std::distance(cdagg_op_pos_.lower_bound(operator_time(new_cdagg_)), cdagg_op_pos_.end())
        );
      permutation_row_col_ *= (diff%2==0 ? 1 : -1);
      cdagg_ops_[nop-1] = new_cdagg_;
      cdagg_op_pos_.erase(operator_time(old_cdagg_));
      cdagg_op_pos_.insert(std::make_pair(operator_time(new_cdagg_), nop-1));
    }

    template<
      typename Scalar,
      typename GreensFunction,
      typename CdaggerOp,
      typename COp
    >
    void
    DeterminantMatrix<Scalar,GreensFunction,CdaggerOp,COp>::reject_replace_cdagg() {
      //do nothing
      check_state(try_replace_cdagg__called);
      state_ = waiting;
    }

  }
}

#include "determinant_matrix.hpp"

namespace alps {
  namespace fastupdate {

    template<
      typename Scalar,
      typename GreensFunction,
      typename CdaggerOp,
      typename COp
    >
    DeterminantMatrix<Scalar,GreensFunction,CdaggerOp,COp>::DeterminantMatrix(const GreensFunction& gf)
      : inv_matrix_(0,0),
        permutation_row_col_(1),
        gf_(gf)
    {
    }

    template<
      typename Scalar,
      typename GreensFunction,
      typename CdaggerOp,
      typename COp
    >
    Scalar
    DeterminantMatrix<Scalar,GreensFunction,CdaggerOp,COp>::try_remove_add(
      const std::vector<std::pair<CdaggerOp,COp> >& cdagg_c_rem,
      const std::vector<std::pair<CdaggerOp,COp> >& cdagg_c_add
    ) {
      const int nop_rem = cdagg_c_rem.size();
      const int nop_add = cdagg_c_add.size();
      const int nop = inv_matrix_.size1();
      const int nop_unchanged = nop - nop_rem;

      //move all rows and cols to be removed to the last
      rem_cols_.resize(nop_rem);
      rem_rows_.resize(nop_rem);
      for (int iop=0; iop<nop_rem; ++iop) {
        rem_cols_[iop] = find_cdagg(cdagg_c_rem[iop].first);
        rem_rows_[iop] = find_c(cdagg_c_rem[iop].second);
      }
      std::sort(rem_cols_.begin(), rem_cols_.end());
      std::sort(rem_rows_.begin(), rem_rows_.end());

      for (int swap=0; swap<nop_rem; ++swap) {
        swap_cols(rem_cols_[nop_rem-1-swap], nop-1-swap);
        swap_rows(rem_rows_[nop_rem-1-swap], nop-1-swap);
      }

      //Add new operators and compute new elements
      perm_rat_ = add_new_operators(cdagg_c_add.begin(), cdagg_c_add.end());

      //compute the values of new elements
      G_n_n_.resize(nop_add, nop_add);
      G_n_j_.resize(nop_add, nop_unchanged);
      G_j_n_.resize(nop_unchanged, nop_add);
      for(int i=0;i<nop_unchanged;++i) {
        for (int iv=0; iv<nop_add; ++iv) {
          G_n_j_(iv,i) = compute_g(nop+iv, i);
        }
      }
      for(int i=0;i<nop_unchanged;++i){
        for (int iv=0; iv<nop_add; ++iv) {
          G_j_n_(i,iv) = compute_g(i, nop+iv);
        }
      }
      for (int iv2=0; iv2<nop_add; ++iv2) {
        for (int iv = 0; iv < nop_add; ++iv) {
          G_n_n_(iv, iv2) = compute_g(nop + iv, nop + iv2);
        }
      }

      //numerical accuracy
      {
        std::cout << "acc " << G_n_n_.determinant() << " " << 1.0/G_n_n_.inverse().determinant() << std::endl;
      }

      replace_helper_
        = ReplaceHelper<Scalar,eigen_matrix_t,eigen_matrix_t,eigen_matrix_t>(inv_matrix_, G_j_n_, G_n_j_, G_n_n_);
      return static_cast<double>(perm_rat_)*replace_helper_.compute_det_ratio(inv_matrix_, G_j_n_, G_n_j_, G_n_n_);
    }

    template<
      typename Scalar,
      typename GreensFunction,
      typename CdaggerOp,
      typename COp
    >
    void
    DeterminantMatrix<Scalar,GreensFunction,CdaggerOp,COp>::perform_remove_add(
      const std::vector<std::pair<CdaggerOp,COp> >& cdagg_c_rem,
      const std::vector<std::pair<CdaggerOp,COp> >& cdagg_c_add
    ) {
      replace_helper_.compute_inverse_matrix(inv_matrix_, G_j_n_, G_n_j_, G_n_n_);
      permutation_row_col_ *= perm_rat_;
    }

    template<
      typename Scalar,
      typename GreensFunction,
      typename CdaggerOp,
      typename COp
    >
    void
    DeterminantMatrix<Scalar,GreensFunction,CdaggerOp,COp>::reject_remove_add(
      const std::vector<std::pair<CdaggerOp,COp> >& cdagg_c_rem,
      const std::vector<std::pair<CdaggerOp,COp> >& cdagg_c_add
    ) {
      /** TO BE IMPLEMENTED **/
    }

    template<
      typename Scalar,
      typename GreensFunction,
      typename CdaggerOp,
      typename COp
    >
    void
    DeterminantMatrix<Scalar,GreensFunction,CdaggerOp,COp>::swap_cols(int col1, int col2) {
      using std::swap;
      if (col1==col2) return;

      const itime_t t1 = operator_time(cdagg_ops_[col1]);
      const itime_t t2 = operator_time(cdagg_ops_[col2]);
      col_pos_[t1] = col2;
      col_pos_[t2] = col1;

      //Note we need to swap ROWS of the inverse matrix (not columns)
      inv_matrix_.swap_row(col1, col2);
      swap(cdagg_ops_[col1], cdagg_ops_[col2]);
      permutation_row_col_ *= -1;

      sanity_check();
    }

    template<
      typename Scalar,
      typename GreensFunction,
      typename CdaggerOp,
      typename COp
    >
    void
    DeterminantMatrix<Scalar,GreensFunction,CdaggerOp,COp>::swap_rows(int row1, int row2) {
      using std::swap;
      if (row1==row2) return;

      const itime_t t1 = operator_time(c_ops_[row1]);
      const itime_t t2 = operator_time(c_ops_[row2]);
      row_pos_[t1] = row2;
      row_pos_[t2] = row1;

      //Note we need to swap COLS of the inverse matrix (not rows)
      inv_matrix_.swap_col(row1, row2);
      swap(c_ops_[row1], c_ops_[row2]);
      permutation_row_col_ *= -1;

      sanity_check();
    }

    template<
      typename Scalar,
      typename GreensFunction,
      typename CdaggerOp,
      typename COp
    >
    int
    DeterminantMatrix<Scalar,GreensFunction,CdaggerOp,COp>::add_new_operators(
      typename std::vector<std::pair<CdaggerOp, COp> >::const_iterator begin,
      typename std::vector<std::pair<CdaggerOp, COp> >::const_iterator end
    ) {
      std::pair<typename std::map<itime_t,int>::iterator,bool> ret;
      itime_t time_new;
      int perm_diff = 0;
      for (typename std::vector<std::pair<CdaggerOp,COp> >::const_iterator it=begin; it!=end; ++it) {
        const int pos = cdagg_ops_.size();

        cdagg_ops_.push_back(it->first);
        time_new = operator_time(it->first);
        perm_diff += std::distance(col_pos_.lower_bound(time_new), col_pos_.end());
        ret = col_pos_.insert(std::make_pair(operator_time(it->first), pos));
        if(ret.second==false) {
          throw std::runtime_error("Something went wrong: cdagg already exists");
        }

        c_ops_.push_back(it->second);
        time_new = operator_time(it->second);
        perm_diff += std::distance(row_pos_.lower_bound(time_new), row_pos_.end());
        ret = row_pos_.insert(std::make_pair(operator_time(it->second), pos));
        if(ret.second==false) {
          throw std::runtime_error("Something went wrong: c operator already exists");
        }
      }

      //permutation_row_col_ = perm%2==0 ? permutation_row_col_ : -1*permutation_row_col_;
      sanity_check();
      return perm_diff%2==0 ? 1 : -1;
    }


    template<
      typename Scalar,
      typename GreensFunction,
      typename CdaggerOp,
      typename COp
    >
    void
    DeterminantMatrix<Scalar,GreensFunction,CdaggerOp,COp>::remove_new_operators(
      const std::vector<std::pair<CdaggerOp, COp> > &cdagg_c_add) {
      const int nop_add = cdagg_c_add.size();

      cdagg_ops_.resize(cdagg_ops_.size()-nop_add);
      c_ops_.resize(c_ops_.size()-nop_add);

      for (typename std::vector<std::pair<CdaggerOp,COp> >::iterator it=cdagg_c_add.begin(); it!=cdagg_c_add.end(); ++it) {
        row_pos_.erase(operator_time(it->first));
        col_pos_.erase(operator_time(it->second));
      }

      sanity_check();
    }

    template<
      typename Scalar,
      typename GreensFunction,
      typename CdaggerOp,
      typename COp
    >
    void
    DeterminantMatrix<Scalar,GreensFunction,CdaggerOp,COp>::sanity_check() const {
#ifndef NDEBUG
      const int num_ops = cdagg_ops_.size();
      const int mat_rank = inv_matrix_.size1();

      assert(cdagg_ops_.size()==c_ops_.size());
      assert(row_pos_.size()==num_ops);
      assert(col_pos_.size()==num_ops);

      for (typename std::map<itime_t,int>::const_iterator it=row_pos_.begin(); it!=row_pos_.end(); ++it) {
        assert(it->second<num_ops);
        assert(operator_time(c_ops_[it->second])==it->first);
      }

      for (typename std::map<itime_t,int>::const_iterator it=col_pos_.begin(); it!=col_pos_.end(); ++it) {
        assert(it->second<num_ops);
        assert(operator_time(cdagg_ops_[it->second])==it->first);
      }

      assert(permutation_row_col_ ==
               permutation(c_ops_.begin(), c_ops_.begin()+mat_rank)*
               permutation(cdagg_ops_.begin(), cdagg_ops_.begin()+mat_rank)
      );
#endif
    }


  }
}

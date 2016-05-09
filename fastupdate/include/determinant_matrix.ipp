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
    void
    DeterminantMatrix<Scalar,GreensFunction,CdaggerOp,COp>::swap_cdagg_op(int col1, int col2) {
      using std::swap;
      if (col1==col2) return;

      const itime_t t1 = operator_time(cdagg_ops_[col1]);
      const itime_t t2 = operator_time(cdagg_ops_[col2]);
      cdagg_op_pos_[t1] = col2;
      cdagg_op_pos_[t2] = col1;

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
    DeterminantMatrix<Scalar,GreensFunction,CdaggerOp,COp>::swap_c_op(int row1, int row2) {
      using std::swap;
      if (row1==row2) return;

      const itime_t t1 = operator_time(c_ops_[row1]);
      const itime_t t2 = operator_time(c_ops_[row2]);
      cop_pos_[t1] = row2;
      cop_pos_[t2] = row1;

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
        perm_diff += std::distance(cdagg_op_pos_.lower_bound(time_new), cdagg_op_pos_.end());
        ret = cdagg_op_pos_.insert(std::make_pair(operator_time(it->first), pos));
        if(ret.second==false) {
          throw std::runtime_error("Something went wrong: cdagg already exists");
        }

        c_ops_.push_back(it->second);
        time_new = operator_time(it->second);
        perm_diff += std::distance(cop_pos_.lower_bound(time_new), cop_pos_.end());
        ret = cop_pos_.insert(std::make_pair(operator_time(it->second), pos));
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
        cop_pos_.erase(operator_time(it->first));
        cdagg_op_pos_.erase(operator_time(it->second));
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
      assert(cop_pos_.size()==num_ops);
      assert(cdagg_op_pos_.size()==num_ops);

      for (typename std::map<itime_t,int>::const_iterator it=cop_pos_.begin(); it!=cop_pos_.end(); ++it) {
        assert(it->second<num_ops);
        assert(operator_time(c_ops_[it->second])==it->first);
      }

      for (typename std::map<itime_t,int>::const_iterator it=cdagg_op_pos_.begin(); it!=cdagg_op_pos_.end(); ++it) {
        assert(it->second<num_ops);
        assert(operator_time(cdagg_ops_[it->second])==it->first);
      }

      assert(permutation_row_col_ ==
               permutation(c_ops_.begin(), c_ops_.begin()+mat_rank)*
               permutation(cdagg_ops_.begin(), cdagg_ops_.begin()+mat_rank)
      );
#endif
    }

    template<
      typename Scalar,
      typename GreensFunction,
      typename CdaggerOp,
      typename COp
    >
    void
    DeterminantMatrix<Scalar,GreensFunction,CdaggerOp,COp>::rebuild_inverse_matrix() {
      const int pert_order = cdagg_ops_.size();
      assert(size()==pert_order);

      inv_matrix_.destructive_resize(pert_order, pert_order);
      for (int j=0; j<pert_order; ++j) {
        for (int i=0; i<pert_order; ++i) {
          inv_matrix_(i,j) = gf_(c_ops_[i], cdagg_ops_[j]);
        }
      }
      inv_matrix_.invert();
    }

    template<
      typename Scalar,
      typename GreensFunction,
      typename CdaggerOp,
      typename COp
    >
    Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>
    DeterminantMatrix<Scalar,GreensFunction,CdaggerOp,COp>::compute_inverse_matrix_time_ordered() {
      const int N = size();
      eigen_matrix_t inv_mat_ordered(N, N);

      //Look up the time-ordered set
      int col = 0;
      for (typename operator_map_t::const_iterator it_c=cop_pos_.begin(); it_c!=cop_pos_.end(); ++it_c) {
        int row = 0;
        for (typename operator_map_t::const_iterator it_cdagg=cdagg_op_pos_.begin(); it_cdagg!=cdagg_op_pos_.end(); ++it_cdagg) {
          inv_mat_ordered(row, col) = inv_matrix_(it_cdagg->second, it_c->second);
          ++row;
        }
        ++col;
      }

      return inv_mat_ordered;
    }
  }
}

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
      : state_(waiting),
        inv_matrix_(0,0),
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
    template<typename CdaggCIterator>
    DeterminantMatrix<Scalar,GreensFunction,CdaggerOp,COp>::DeterminantMatrix(
      const GreensFunction& gf,
      CdaggCIterator first,
      CdaggCIterator last
    )
      : state_(waiting),
        inv_matrix_(0,0),
        permutation_row_col_(1),
        gf_(gf)
    {
      try_add(first, last);
      perform_add();
    }

    template<
      typename Scalar,
      typename GreensFunction,
      typename CdaggerOp,
      typename COp
    >
    template<typename CdaggCIterator>
    bool
      DeterminantMatrix<Scalar,GreensFunction,CdaggerOp,COp>::insertion_possible(
      CdaggCIterator first,
      CdaggCIterator last) {

      int iop = 0;
      std::vector<itime_t> times_add(2*std::distance(first, last));
      for (CdaggCIterator it=first; it!=last; ++it) {
        const itime_t t1 = times_add[2*iop] = operator_time(it->first);
        const itime_t t2 = times_add[2*iop+1] = operator_time(it->second);

        if (exist(t1) || exist(t2)) {
          return false;
        }
        ++iop;
      }
      std::sort(times_add.begin(), times_add.end());
      if (boost::adjacent_find(times_add) != times_add.end()) {
        return false;
      }
      return true;
    };

    template<
      typename Scalar,
      typename GreensFunction,
      typename CdaggerOp,
      typename COp
    >
    template<typename CdaggCIterator>
    bool
    DeterminantMatrix<Scalar,GreensFunction,CdaggerOp,COp>::removal_insertion_possible(
      CdaggCIterator first_removal,
      CdaggCIterator last_removal,
      CdaggCIterator first_insertion,
      CdaggCIterator last_insertion
    ) {
      //check if removal is possible
      const int nop_rem = std::distance(first_removal, last_removal);
      std::vector<itime_t> times_rem(2*nop_rem);
      int iop = 0;
      for (CdaggCIterator it=first_removal; it!=last_removal; ++it) {
        bool tmp = exist(it->first) && exist(it->second);
        times_rem[2*iop] = operator_time(it->first);
        times_rem[2*iop+1] = operator_time(it->second);
        if (!tmp) {
          return false;
        }
        ++iop;
      }

      //check if insertion is possible.
      const int nop_add = std::distance(first_insertion, last_insertion);
      std::vector<itime_t> times_add(2*nop_add);
      iop = 0;
      for (CdaggCIterator it=first_insertion; it!=last_insertion; ++it) {
        const itime_t t1 = times_add[2*iop] = operator_time(it->first);
        const itime_t t2 = times_add[2*iop+1] = operator_time(it->second);
        bool tmp =
          (
            !exist(t1) || (std::find(times_rem.begin(), times_rem.end(), t1)!=times_rem.end())
          ) &&
          (
            !exist(t2) || (std::find(times_rem.begin(), times_rem.end(), t2)!=times_rem.end())
          );
        if (!tmp) {
          return false;
        }
        ++iop;
      }

      //check if there is no duplicate
      std::sort(times_rem.begin(), times_rem.end());
      std::sort(times_add.begin(), times_add.end());
      if (
        boost::adjacent_find(times_add) != times_add.end() ||
        boost::adjacent_find(times_rem) != times_rem.end()
        ) {
        return false;
      }

      return true;
    };

    template<
      typename Scalar,
      typename GreensFunction,
      typename CdaggerOp,
      typename COp
    >
    template<typename CdaggCIterator>
    bool
    DeterminantMatrix<Scalar,GreensFunction,CdaggerOp,COp>::removal_possible(
      CdaggCIterator first_removal,
      CdaggCIterator last_removal
    ) const {
      //check if removal is possible
      const int nop_rem = std::distance(first_removal, last_removal);
      std::vector<itime_t> times_rem(2*nop_rem);
      bool possible = true;
      int iop = 0;
      for (CdaggCIterator it=first_removal; it!=last_removal; ++it) {
        itime_t t1 = operator_time(it->first);
        itime_t t2 = operator_time(it->second);
        bool tmp = exist(t1) && exist(t2);
        times_rem[2*iop] = t1;
        times_rem[2*iop+1] = t2;
        if (!tmp) {
          possible = false;
          break;
        }
        ++iop;
      }
      if (!possible) {
        return false;
      }

      //check if there is no duplicate
      std::sort(times_rem.begin(), times_rem.end());
      if (boost::adjacent_find(times_rem)!=times_rem.end()) {
        return false;
      }

      return possible;
    };

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
    }

    template<
      typename Scalar,
      typename GreensFunction,
      typename CdaggerOp,
      typename COp
    >
    template<
      typename Iterator
    >
    int
    DeterminantMatrix<Scalar,GreensFunction,CdaggerOp,COp>::add_new_operators(
      Iterator first,
      Iterator last
    ) {
      std::pair<typename std::map<itime_t,int>::iterator,bool> ret;
      itime_t time_new;
      int perm_diff = 0;
      for (Iterator it=first; it!=last; ++it) {
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

      //sanity_check();
      return perm_diff%2==0 ? 1 : -1;
    }

    template<
      typename Scalar,
      typename GreensFunction,
      typename CdaggerOp,
      typename COp
    >
    int
    DeterminantMatrix<Scalar,GreensFunction,CdaggerOp,COp>::remove_last_operators(int num_operators_remove) {
      const int num_ops_remain = cdagg_ops_.size() - num_operators_remove;
      assert(num_ops_remain >= 0);
      if (num_ops_remain < 0) {
        throw std::logic_error("num_ops_remain < 0");
      }

      //Remove the last operators one by one
      int perm_diff = 0;
      for (int iop=0; iop<num_operators_remove; ++iop) {
        const itime_t t1 = operator_time(c_ops_.back());
        perm_diff += std::distance(cop_pos_.lower_bound(t1), cop_pos_.end());
        cop_pos_.erase(t1);

        const itime_t t2 = operator_time(cdagg_ops_.back());
        perm_diff += std::distance(cdagg_op_pos_.lower_bound(t2), cdagg_op_pos_.end());
        cdagg_op_pos_.erase(t2);

        c_ops_.pop_back();
        cdagg_ops_.pop_back();
      }

      return perm_diff%2==0 ? 1 : -1;
    }

    template<
      typename Scalar,
      typename GreensFunction,
      typename CdaggerOp,
      typename COp
    >
    void
    DeterminantMatrix<Scalar,GreensFunction,CdaggerOp,COp>::remove_excess_operators() {
      const int nop_rem = cdagg_op_pos_.size() - inv_matrix_.size1(); //number of excess operators to be removed
      const int offset = inv_matrix_.size1();

      //remove operators from std::map<operator_time,int>
      for (int iop=0; iop<nop_rem; ++iop) {
        cop_pos_.erase(operator_time(c_ops_[iop+offset]));
        cdagg_op_pos_.erase(operator_time(cdagg_ops_[iop+offset]));
      }

      cdagg_ops_.resize(offset);
      c_ops_.resize(offset);

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
      check_state(waiting);

      //print_operators();

      const int num_ops = cdagg_ops_.size();
      const int mat_rank = inv_matrix_.size1();

      assert(mat_rank==num_ops);
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

      for (int iop=0; iop<mat_rank; ++iop) {
        assert(find_cdagg(cdagg_ops_[iop])<mat_rank);
        assert(find_c(c_ops_[iop])<mat_rank);
      }
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
      check_state(waiting);

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
    ResizableMatrix<Scalar>
    DeterminantMatrix<Scalar,GreensFunction,CdaggerOp,COp>::build_matrix() {
      check_state(waiting);

      const int pert_order = cdagg_ops_.size();
      assert(size()==pert_order);

      ResizableMatrix<Scalar> matrix(pert_order, pert_order);
      for (int j=0; j<pert_order; ++j) {
        for (int i=0; i<pert_order; ++i) {
          matrix(i,j) = gf_(c_ops_[i], cdagg_ops_[j]);
        }
      }
      return matrix;
    }

    template<
      typename Scalar,
      typename GreensFunction,
      typename CdaggerOp,
      typename COp
    >
    Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>
    DeterminantMatrix<Scalar,GreensFunction,CdaggerOp,COp>::compute_inverse_matrix_time_ordered() {
      check_state(waiting);

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

    template<
      typename Scalar,
      typename GreensFunction,
      typename CdaggerOp,
      typename COp
    >
    void
    DeterminantMatrix<Scalar,GreensFunction,CdaggerOp,COp>::print_operators() const {
      const int N = size();

      for (int iop=0; iop<cdagg_ops_.size(); ++iop) {
        std::cout << "operator at row/col " << iop << " " << operator_time(c_ops_[iop]) << " " << operator_time(cdagg_ops_[iop]) << std::endl;
      }
    }

  }
}

namespace alps {
  namespace fastupdate {

    template<
      typename Scalar,
      typename GreensFunction,
      typename CdaggerOp,
      typename COp
    >
    template<typename CdaggCIterator>
    Scalar
    DeterminantMatrix<Scalar,GreensFunction,CdaggerOp,COp>::try_remove_add(
      CdaggCIterator cdagg_c_rem_first,
      CdaggCIterator cdagg_c_rem_last,
      CdaggCIterator cdagg_c_add_first,
      CdaggCIterator cdagg_c_add_last
    ) {
      check_state(waiting);
      state_ = try_rem_add_called;

      const int nop_rem = std::distance(cdagg_c_rem_first, cdagg_c_rem_last);
      const int nop_add = std::distance(cdagg_c_add_first, cdagg_c_add_last);
      const int nop = inv_matrix_.size1();
      const int nop_unchanged = nop - nop_rem;

      if (!removal_insertion_possible(
        cdagg_c_rem_first, cdagg_c_rem_last, cdagg_c_add_first, cdagg_c_add_last)
        ) {
        return 0.0;
      }

      //move all rows and cols to be removed to the last
      if (nop_rem>0) {
        rem_cols_.resize(nop_rem);
        rem_rows_.resize(nop_rem);
        CdaggCIterator it = cdagg_c_rem_first;
        for (int iop=0; iop<nop_rem; ++iop) {
          rem_cols_[iop] = find_cdagg(it->first);
          rem_rows_[iop] = find_c(it->second);
          ++it;
        }
        std::sort(rem_cols_.begin(), rem_cols_.end());
        std::sort(rem_rows_.begin(), rem_rows_.end());

        for (int swap=0; swap<nop_rem; ++swap) {
          swap_cdagg_op(rem_cols_[nop_rem-1-swap], nop-1-swap);
          swap_c_op(rem_rows_[nop_rem-1-swap], nop-1-swap);
        }
      }

      //Remove the last operators and add new operators
      perm_rat_ = remove_last_operators(nop_rem);
      perm_rat_ *= add_new_operators(cdagg_c_add_first, cdagg_c_add_last);

      //compute the values of new elements
      G_n_n_.resize(nop_add, nop_add);
      G_n_j_.resize(nop_add, nop_unchanged);
      G_j_n_.resize(nop_unchanged, nop_add);
      for(int i=0;i<nop_unchanged;++i) {
        for (int iv=0; iv<nop_add; ++iv) {
          G_n_j_(iv,i) = compute_g(nop_unchanged+iv, i);
        }
      }
      for(int i=0;i<nop_unchanged;++i){
        for (int iv=0; iv<nop_add; ++iv) {
          G_j_n_(i,iv) = compute_g(i, nop_unchanged+iv);
        }
      }
      for (int iv2=0; iv2<nop_add; ++iv2) {
        for (int iv = 0; iv < nop_add; ++iv) {
          G_n_n_(iv, iv2) = compute_g(nop_unchanged + iv, nop_unchanged + iv2);
        }
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
    DeterminantMatrix<Scalar,GreensFunction,CdaggerOp,COp>::perform_remove_add() {
      replace_helper_.compute_inverse_matrix(inv_matrix_, G_j_n_, G_n_j_, G_n_n_);
      check_state(try_rem_add_called);
      state_ = waiting;

      permutation_row_col_ *= perm_rat_;
      sanity_check();
    }

    template<
      typename Scalar,
      typename GreensFunction,
      typename CdaggerOp,
      typename COp
    >
    template<typename CdaggCIterator>
    void
    DeterminantMatrix<Scalar,GreensFunction,CdaggerOp,COp>::reject_remove_add(
      CdaggCIterator cdagg_c_rem_first,
      CdaggCIterator cdagg_c_rem_last,
      CdaggCIterator cdagg_c_add_first,
      CdaggCIterator cdagg_c_add_last
    ) {
      check_state(try_rem_add_called);
      state_ = waiting;

      //then the last operators
      remove_last_operators(std::distance(cdagg_c_add_first, cdagg_c_add_last));

      //then insert the removed operators back
      add_new_operators(cdagg_c_rem_first, cdagg_c_rem_last);
      sanity_check();
    }
  }
}

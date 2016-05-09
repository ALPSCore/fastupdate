namespace alps {
  namespace fastupdate {

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
      if (nop_rem>0) {
        rem_cols_.resize(nop_rem);
        rem_rows_.resize(nop_rem);
        for (int iop=0; iop<nop_rem; ++iop) {
          rem_cols_[iop] = find_cdagg(cdagg_c_rem[iop].first);
          rem_rows_[iop] = find_c(cdagg_c_rem[iop].second);
        }
        std::sort(rem_cols_.begin(), rem_cols_.end());
        std::sort(rem_rows_.begin(), rem_rows_.end());

        for (int swap=0; swap<nop_rem; ++swap) {
          swap_cdagg_op(rem_cols_[nop_rem-1-swap], nop-1-swap);
          swap_c_op(rem_rows_[nop_rem-1-swap], nop-1-swap);
        }
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
      /*
      {

        std::cout << "N " << G_n_n_.rows() << std::endl;
        std::cout << "acc1 " << G_n_n_.determinant() << " " << 1.0/G_n_n_.inverse().determinant() << std::endl;
        eigen_matrix_t inv = G_n_n_;
        safe_invert_in_place(inv);
        std::cout << "acc3 " << safe_determinant(G_n_n_) << " " << 1.0/safe_determinant(inv) << std::endl;
      }
      */

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
      throw std::runtime_error("Not implemented");
    }
  }
}

namespace alps {
  namespace fastupdate {

    template<
      typename Scalar,
      typename GreensFunction,
      typename CdaggerOp,
      typename COp
    >
    Scalar
    DeterminantMatrix<Scalar,GreensFunction,CdaggerOp,COp>::try_add(
      const std::vector<std::pair<CdaggerOp,COp> >& cdagg_c_add
    ) {
      const int nop_add = cdagg_c_add.size();
      const int nop = inv_matrix_.size1();

      //Add new operators and compute new elements
      perm_rat_ = add_new_operators(cdagg_c_add.begin(), cdagg_c_add.end());

      //compute the values of new elements
      G_n_n_.resize(nop_add, nop_add);
      G_n_j_.resize(nop_add, nop);
      G_j_n_.resize(nop, nop_add);
      for(int i=0;i<nop;++i) {
        for (int iv=0; iv<nop_add; ++iv) {
          G_n_j_(iv,i) = compute_g(nop+iv, i);
        }
      }
      for(int i=0;i<nop; ++i){
        for (int iv=0; iv<nop_add; ++iv) {
          G_j_n_(i,iv) = compute_g(i, nop+iv);
        }
      }
      for (int iv2=0; iv2<nop_add; ++iv2) {
        for (int iv = 0; iv < nop_add; ++iv) {
          G_n_n_(iv, iv2) = compute_g(nop + iv, nop + iv2);
        }
      }

      return static_cast<double>(perm_rat_)*compute_det_ratio_up(G_j_n_, G_n_j_, G_n_n_, inv_matrix_);
    }

    template<
      typename Scalar,
      typename GreensFunction,
      typename CdaggerOp,
      typename COp
    >
    void
    DeterminantMatrix<Scalar,GreensFunction,CdaggerOp,COp>::perform_add(
      const std::vector<std::pair<CdaggerOp,COp> >& cdagg_c_add
    ) {
      compute_inverse_matrix_up(G_j_n_, G_n_j_, G_n_n_, inv_matrix_);
      permutation_row_col_ *= perm_rat_;
    }

    template<
      typename Scalar,
      typename GreensFunction,
      typename CdaggerOp,
      typename COp
    >
    void
    DeterminantMatrix<Scalar,GreensFunction,CdaggerOp,COp>::reject_add(
      const std::vector<std::pair<CdaggerOp,COp> >& cdagg_c_add
    ) {
      remove_new_operators(cdagg_c_add);
    }

  }
}

#include "unittest.hpp"


TEST(FastUpdate, BlockMatrixAdd)
{
  using namespace alps::fastupdate;

  typedef double Scalar;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> eigen_matrix_t;

  std::vector<size_t> N_list, M_list;
  N_list.push_back(0);
  N_list.push_back(10);
  N_list.push_back(2);

  M_list.push_back(10);
  M_list.push_back(20);
  M_list.push_back(4);

  for (size_t n=0; n<N_list.size(); ++n) {
    for (size_t m=0; m<M_list.size(); ++m) {
      const size_t N = N_list[n];
      const size_t M = M_list[m];

      eigen_matrix_t A(N,N), B(N,M), C(M,N), D(M,M);
      eigen_matrix_t E(N,N), F(N,M), G(M,N), H(M,M);
      ResizableMatrix<Scalar> invA(N,N), BigMatrix(N+M, N+M, 0);//, invBigMatrix2(N+M, N+M, 0);

      randomize_matrix(A, 100);//100 is a seed
      randomize_matrix(B, 200);
      randomize_matrix(C, 300);
      randomize_matrix(D, 400);
      if (N>0) {
        invA = A.inverse();
      } else {
        invA.destructive_resize(0,0);
      }

      copy_block(A,0,0,BigMatrix,0,0,N,N);
      copy_block(B,0,0,BigMatrix,0,N,N,M);
      copy_block(C,0,0,BigMatrix,N,0,M,N);
      copy_block(D,0,0,BigMatrix,N,N,M,M);

      const Scalar det_rat = compute_det_ratio_up<Scalar>(B, C, D, invA);
      ASSERT_TRUE(std::abs(det_rat-determinant(BigMatrix)/A.determinant())<1E-8)
                << "N=" << N << " M=" << M << " " << std::abs(det_rat-determinant(BigMatrix)) << "/" << std::abs(det_rat)<<"="
                << std::abs(det_rat-determinant(BigMatrix)/A.determinant());

      const Scalar det_rat2 = alps::fastupdate::compute_inverse_matrix_up(B, C, D, invA);
      ASSERT_TRUE(std::abs(det_rat-det_rat2)<1E-8) << "N=" << N << " M=" << M;
      ASSERT_TRUE(norm_square(inverse(BigMatrix)-invA)<1E-8) << "N=" << N << " M=" << M;
    }
  }
}


void select_rows_removed(unsigned int seed, int N, int M, std::vector<int>& rows_removed, std::vector<int>& rows_remain) {
  boost::mt19937 gen(seed);
  rows_removed.resize(N+M);
  rows_remain.resize(N);
  for (int i=0; i<N+M; ++i) {
    rows_removed[i] = i;
  }
  rs_shuffle rs(gen);
  std::random_shuffle(rows_removed.begin(), rows_removed.end(), rs);
  for (int i=0; i<N; ++i) {
    rows_remain[i] = rows_removed[i+M];
  }
  rows_removed.resize(M);
  std::sort(rows_removed.begin(), rows_removed.end());
  std::sort(rows_remain.begin(), rows_remain.end());
}

TEST(FastUpdate, BlockMatrixDown)
{
  using namespace alps::fastupdate;
  typedef double Scalar;
  //typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> matrix_t;

  std::vector<int> N_list, M_list;
  N_list.push_back(10);
  M_list.push_back(10);
  M_list.push_back(20);
  M_list.push_back(30);

  for (int n=0; n<N_list.size(); ++n) {
    for (int m=0; m<M_list.size(); ++m) {
      const int N = N_list[n];
      const int M = M_list[m];

      ResizableMatrix<Scalar> G(N+M, N+M, 0.0), invG(N+M, N+M, 0.0);//G, G^{-1}
      ResizableMatrix<Scalar> Gprime(N, N, 0.0);//G'

      randomize_matrix(G, 100);//100 is a seed
      invG = inverse(G);

      //which rows and cols are to be removed
      std::vector<int> rows_removed(N+M), rows_remain(N), cols_removed(N+M), cols_remain(N);
      select_rows_removed(919, N, M, rows_removed, rows_remain);
      select_rows_removed(119, N, M, cols_removed, cols_remain);

      for (int j=0; j<N; ++j) {
        for (int i=0; i<N; ++i) {
          Gprime(i,j) = G(rows_remain[i], cols_remain[j]);
        }
      }

      //testing compute_det_ratio_down
      Scalar det_rat = compute_det_ratio_down(M, rows_removed, cols_removed, invG);
      ASSERT_TRUE(std::abs(det_rat-Gprime.determinant()/G.determinant())<1E-8) << "N=" << N << " M=" << M;

      //update G^{-1} to G'^{-1}
      ResizableMatrix<Scalar> invGprime_fastupdate(invG);
      compute_inverse_matrix_down(M, rows_removed, cols_removed, invGprime_fastupdate);

      //Note that remaining rows and cols may be swapped in the fastupdate
      ResizableMatrix<Scalar> invGprime = G;
      for (int s=0; s<M; ++s) {
        invGprime.swap_row(rows_removed[M-1-s], N+M-1-s);
        invGprime.swap_col(cols_removed[M-1-s], N+M-1-s);
      }
      invGprime.conservative_resize(N,N);
      invGprime.invert();
      ASSERT_TRUE(norm_square(invGprime-invGprime_fastupdate)<1E-8) << "N=" << N << " M=" << M;
    }
  }
}

/*
TEST(FastUpdate, BlockMatrixReplaceRowsCols) {
    std::vector<size_t> N_list, M_list;
    N_list.push_back(10);
    M_list.push_back(4);

    N_list.push_back(100);
    M_list.push_back(50);

    N_list.push_back(100);
    M_list.push_back(49);

    for (int n = 0; n < N_list.size(); ++n) {
        for (int m = 0; m < M_list.size(); ++m) {
            const int N = N_list[n];
            const int M = M_list[m];

            typedef alps::numeric::matrix<double> matrix_t;

            matrix_t BigMatrix(N + M, N + M, 0), invBigMatrix(N + M, N + M, 0);
            std::vector<std::pair<int, int> > swap_list;

            randomize_matrix(BigMatrix, 100);//100 is a seed
            invBigMatrix = inverse(BigMatrix);

            //which rows and cols are to be replaced
            std::vector<int> rows_replaced(N + M);
            for (int i = 0; i < N + M; ++i) {
                rows_replaced[i] = i;
            }
            std::random_shuffle(rows_replaced.begin(), rows_replaced.end());
            rows_replaced.resize(M);
            std::sort(rows_replaced.begin(), rows_replaced.end());

            swap_list.resize(M);
            for (int i=0; i<M; ++i) {
                swap_list[i] = std::pair<int,int>(rows_replaced[M-1-i], N+M-1-i);
            }

            matrix_t R(M, N), S(M, M), Q(N, M);
            randomize_matrix(R, 110);//100 is a seed
            randomize_matrix(S, 210);//100 is a seed
            randomize_matrix(Q, 310);//100 is a seed

            matrix_t BigMatrixReplaced(BigMatrix);
            replace_rows_cols(BigMatrixReplaced, Q, R, S, rows_replaced);

            //testing compute_det_ratio_down
            double det_rat = alps::numeric::determinant(BigMatrixReplaced)/determinant(BigMatrix);

            matrix_t invBigMatrix_fast(invBigMatrix), Mmat, inv_tSp, tPp, tQp, tRp, tSp;
            swap_cols_rows(invBigMatrix_fast, swap_list.begin(), swap_list.end());
            double det_rat_fast = compute_det_ratio_replace_rows_cols(invBigMatrix_fast, Q, R, S, Mmat, inv_tSp);
            //compute_inverse_matrix_replace_rows_cols(invBigMatrix_fast, Q, R, S, Mmat, inv_tSp, tPp, tQp, tRp, tSp);
            compute_inverse_matrix_replace_rows_cols(invBigMatrix_fast, Q, R, S, Mmat, inv_tSp);
            swap_cols_rows(invBigMatrix_fast, swap_list.rbegin(), swap_list.rend());
            ASSERT_TRUE(std::abs(det_rat-det_rat_fast)<1E-8);
            ASSERT_TRUE(alps::numeric::norm_square(inverse(BigMatrixReplaced)-invBigMatrix_fast)<1E-8);
        }
    }
}
*/


TEST(FastUpdate, BlockMatrixReplaceLastRowsColsWithDifferentSizes) {
  using namespace alps::fastupdate;
  typedef std::complex<double> Scalar;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> eigen_matrix_t;
  typedef ReplaceHelper<Scalar, eigen_matrix_t, eigen_matrix_t, eigen_matrix_t> helper_t;

  std::vector<int> N_list, M_list, Mold_list;
  N_list.push_back(0);
  M_list.push_back(4);
  Mold_list.push_back(5);

  N_list.push_back(2);
  M_list.push_back(0);
  Mold_list.push_back(5);

  N_list.push_back(2);
  M_list.push_back(5);
  Mold_list.push_back(0);

  N_list.push_back(100);
  M_list.push_back(40);
  Mold_list.push_back(50);

  N_list.push_back(100);
  M_list.push_back(49);
  Mold_list.push_back(20);

  //N_list.push_back(100);
  //M_list.push_back(100);
  //Mold_list.push_back(20);

  for (int n = 0; n < N_list.size(); ++n) {
    for (int m = 0; m < M_list.size(); ++m) {
      const int N = N_list[n];
      const int M = M_list[m];
      const int Mold = Mold_list[m];

      ResizableMatrix<Scalar> G(N + Mold, N + Mold, 0), invG(N + Mold, N + Mold, 0);

      randomize_matrix(G, 100);//100 is a seed
      invG = inverse(G);

      //New entries
      eigen_matrix_t R(M, N), S(M, M), Q(N, M);
      randomize_matrix(R, 110);//100 is a seed
      randomize_matrix(S, 210);//100 is a seed
      randomize_matrix(Q, 310);//100 is a seed

      ResizableMatrix<Scalar> Gprime(N+M, N+M);
      copy_block(G, 0, 0, Gprime, 0, 0, N, N);
      copy_block(R, 0, 0, Gprime, N, 0, M, N);
      copy_block(Q, 0, 0, Gprime, 0, N, N, M);
      copy_block(S, 0, 0, Gprime, N, N, M, M);

      //testing compute_det_ratio_down
      const Scalar det_rat = determinant(Gprime)/determinant(G);

      // construct a helper
      ResizableMatrix<Scalar> invGprime_fast(invG);//, Mmat, inv_tSp, tPp, tQp, tRp, tSp;
      helper_t helper(invGprime_fast, Q, R, S);

      // compute det ratio
      const Scalar det_rat_fast = helper.compute_det_ratio(invGprime_fast, Q, R, S);
      ASSERT_TRUE(std::abs(det_rat-det_rat_fast)/std::abs(det_rat)<1E-8);

      // update the inverse matrix
      helper.compute_inverse_matrix(invGprime_fast, Q, R, S);

      ASSERT_TRUE(norm_square(inverse(Gprime)-invGprime_fast)<1E-8);
    }
  }
}

TEST(DeterminantMatrix, AddRowsCols) {
  using namespace alps::fastupdate;
  typedef std::complex<double> Scalar;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> eigen_matrix_t;

  const int n_flavors = 3;
  const double beta = 1.0;
  typedef DeterminantMatrix<
    Scalar,
    OffDiagonalG0<Scalar>,
    creator,
    annihilator
  > determinant_matrix_t;
  const int seed = 122;
  boost::mt19937 gen(seed);
  boost::uniform_01<> unidist;
  rs_shuffle rs(gen);

  std::vector<double> E(n_flavors);
  boost::multi_array<Scalar,2> phase(boost::extents[n_flavors][n_flavors]);

  for (int i=0; i<n_flavors; ++i) {
    E[i] = 10*(double) (i+0.5);
  }
  for (int i=0; i<n_flavors; ++i) {
    for (int j=i; j<n_flavors; ++j) {
      //phase[i][j] = std::exp(std::complex<double>(0.0, 1.*i*(2*j+1.0)));
      //phase[j][i] = std::conj(phase[i][j]);
      phase[i][j] = 1.0;
      phase[j][i] = std::conj(phase[i][j]);
    }
  }

  determinant_matrix_t det_mat(OffDiagonalG0<Scalar>(beta, n_flavors, E, phase));

  std::vector<std::pair<creator,annihilator> > init_ops;
  for (int i=0; i<4; ++i) {
    init_ops.push_back(
      std::make_pair(
        creator(n_flavors*unidist(gen), unidist(gen)*beta),
        annihilator(n_flavors*unidist(gen), unidist(gen)*beta)
      )
    );
  }

  const Scalar det_rat = det_mat.try_add(
    init_ops
  );

  det_mat.perform_add(
    init_ops
  );

  const Scalar det_init = det_mat.compute_determinant();
  ASSERT_TRUE(std::abs(det_init-det_rat)/std::abs(det_rat)<1E-8);

  //check inverse matrix
  eigen_matrix_t inv_mat_init = det_mat.compute_inverse_matrix_time_ordered();
  det_mat.rebuild_inverse_matrix();
  eigen_matrix_t inv_mat_init_rebuild = det_mat.compute_inverse_matrix_time_ordered();
  ASSERT_TRUE((inv_mat_init-inv_mat_init_rebuild).squaredNorm()/inv_mat_init.squaredNorm()<1E-8);

  /*
   * Now we remove some operators and add some operators
   */
  std::vector<std::pair<creator,annihilator> > ops_add;
  for (int i=0; i<4; ++i) {
    ops_add.push_back(
      std::make_pair(
        creator(n_flavors*unidist(gen), unidist(gen)*beta),
        annihilator(n_flavors*unidist(gen), unidist(gen)*beta)
      )
    );
  }

  const Scalar det_rat2 = det_mat.try_add(
    ops_add
  );

  det_mat.perform_add(
    ops_add
  );

  const Scalar det2 = det_mat.compute_determinant();
  std::cout << "det_rat " << det2/det_init << " " << det_rat2 << std::endl;

  //check inverse matrix
  eigen_matrix_t inv_mat_fu = det_mat.compute_inverse_matrix_time_ordered();
  det_mat.rebuild_inverse_matrix();
  eigen_matrix_t inv_mat_rebuild = det_mat.compute_inverse_matrix_time_ordered();
  std::cout << "inv " << inv_mat_fu << std::endl;
  std::cout << "inv " << inv_mat_rebuild << std::endl;
  ASSERT_TRUE((inv_mat_fu-inv_mat_rebuild).squaredNorm()/inv_mat_rebuild.squaredNorm()<1E-8);
  ASSERT_TRUE(std::abs(det2/det_init-det_rat2)/std::abs(det_rat2)<1E-8);
}

/*
TEST(FastUpdate, DeterminantMatrix) {
  using namespace alps::fastupdate;
  typedef std::complex<double> Scalar;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> eigen_matrix_t;

  const int n_flavors = 3;
  const double beta = 1.0;
  typedef DeterminantMatrix<
    Scalar,
    OffDiagonalG0<Scalar>,
    creator,
    annihilator
  > determinant_matrix_t;
  const int seed = 122;
  boost::mt19937 gen(seed);
  boost::uniform_01<> unidist;
  rs_shuffle rs(gen);

  std::vector<double> E(n_flavors);
  boost::multi_array<Scalar,2> phase(boost::extents[n_flavors][n_flavors]);

  for (int i=0; i<n_flavors; ++i) {
    E[i] = 10*(double) (i+0.5);
  }
  for (int i=0; i<n_flavors; ++i) {
    for (int j=i; j<n_flavors; ++j) {
      //phase[i][j] = std::exp(std::complex<double>(0.0, 1.*i*(2*j+1.0)));
      //phase[j][i] = std::conj(phase[i][j]);
      phase[i][j] = 1.0;
      phase[j][i] = std::conj(phase[i][j]);
    }
  }

  determinant_matrix_t det_mat(OffDiagonalG0<Scalar>(beta, n_flavors, E, phase));

  std::vector<std::pair<creator,annihilator> > init_ops;
  for (int i=0; i<2; ++i) {
    init_ops.push_back(
      std::make_pair(
        creator(n_flavors*unidist(gen), unidist(gen)*beta),
        annihilator(n_flavors*unidist(gen), unidist(gen)*beta)
      )
    );
  }

  const Scalar det_rat = det_mat.try_add(
    //std::vector<std::pair<creator,annihilator> >(),
    init_ops
  );

  det_mat.perform_add(
    //std::vector<std::pair<creator,annihilator> >(),
    init_ops
  );

  const Scalar det_init = det_mat.compute_determinant();
  //std::cout << "det " << det_init << " " << det_rat << std::endl;
  ASSERT_TRUE(std::abs(det_init-det_rat)/std::abs(det_rat)<1E-8);

  //check inverse matrix
  eigen_matrix_t inv_mat_init = det_mat.compute_inverse_matrix_time_ordered();
  det_mat.rebuild_inverse_matrix();
  eigen_matrix_t inv_mat_init_rebuild = det_mat.compute_inverse_matrix_time_ordered();
  ASSERT_TRUE((inv_mat_init-inv_mat_init_rebuild).squaredNorm()/inv_mat_init.squaredNorm()<1E-8);

  std::vector<std::pair<creator,annihilator> > ops_add;
  for (int i=0; i<2; ++i) {
    ops_add.push_back(
      std::make_pair(
        creator(n_flavors*unidist(gen), unidist(gen)*beta),
        annihilator(n_flavors*unidist(gen), unidist(gen)*beta)
      )
    );
  }

  //std::vector<std::pair<creator,annihilator> > ops_rem(init_ops);
  //std::random_shuffle(ops_rem.begin(), ops_rem.end(), rs);
  //ops_rem.resize(init_ops.size()/2);
  //ops_rem.resize(0);

  const Scalar det_rat2 = det_mat.try_add(
    ops_add
  );

  det_mat.perform_add(
    ops_add
  );

  const Scalar det2 = det_mat.compute_determinant();
  //std::cout << det2 << " " << det_init << " " << det_rat2 << std::endl;
  std::cout << "det_rat " << det2/det_init << " " << det_rat2 << std::endl;

  //check inverse matrix
  eigen_matrix_t inv_mat_fu = det_mat.compute_inverse_matrix_time_ordered();
  det_mat.rebuild_inverse_matrix();
  eigen_matrix_t inv_mat_rebuild = det_mat.compute_inverse_matrix_time_ordered();
  std::cout << "inv " << inv_mat_fu << std::endl;
  std::cout << "inv " << inv_mat_rebuild << std::endl;
  ASSERT_TRUE((inv_mat_fu-inv_mat_rebuild).squaredNorm()/inv_mat_rebuild.squaredNorm()<1E-8);

  ASSERT_TRUE(std::abs(det2/det_init-det_rat2)/std::abs(det_rat2)<1E-8);
  //std::vector<std::pair<creator,annihilator> > ops_rem;
  //ops_rem.push_back(
  //std::make_pair(creator(0, 0.1*beta), annihilator(0, 0.2*beta))
  //);

}
*/

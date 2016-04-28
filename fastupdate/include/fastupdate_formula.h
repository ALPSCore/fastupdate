/**
 * Fast-update formula based on block matrix representation
 * Copyright (C) 2016 by Hiroshi Shinaoka <h.shinaoka@gmail.com>
 *
 */
#pragma once

#include "resizable_matrix.hpp"

namespace alps {
  namespace fastupdate {

    template<typename Derived>
    inline int num_cols(const Eigen::MatrixBase<Derived> &m) {
      return m.cols();
    }

    template<typename Derived>
    inline int num_rows(const Eigen::MatrixBase<Derived> &m) {
      return m.rows();
    }


    /**
     * Compute the determinant ratio with addition rows and cols
     * We implement equations in Appendix B.1.1 of Luitz's thesis.
     * https://opus.bibliothek.uni-wuerzburg.de/files/6408/thesis_luitz.pdf
     *
     * @param B right top block of the new matrix
     * @param C left bottom block of the new matrix
     * @param D right bottom block of the new matrix
     * @param invA inverse of the currrent matrix
     */
    template<typename Scalar, typename Derived>
    Scalar
    compute_det_ratio_up(
      const Eigen::MatrixBase<Derived> &B,
      const Eigen::MatrixBase<Derived> &C,
      const Eigen::MatrixBase<Derived> &D,
      const ResizableMatrix<Scalar> &invA) {
      const size_t N = invA.size1();
      const size_t M = D.rows();

      assert(M > 0);

      assert(num_rows(invA) == num_cols(invA));
      assert(num_rows(B) == N && num_cols(B) == M);
      assert(num_rows(C) == M && num_cols(C) == N);
      assert(num_rows(D) == M && num_cols(D) == M);

      if (N == 0) {
        return D.determinant();
      } else {
        //compute H
        return (D - C * invA.block() * B).determinant();
      }
    }

    /**
     * Update the inverse matrix by adding rows and cols
     * We implement equations in Appendix B.1.1 of Luitz's thesis.
     * https://opus.bibliothek.uni-wuerzburg.de/files/6408/thesis_luitz.pdf
     *
     * @param B right top block of the new matrix
     * @param C left bottom block of the new matrix
     * @param D right bottom block of the new matrix
     * @param invA inverse of the currrent matrix. invA is resized automatically.
     */
    template<typename Scalar, typename Derived>
    Scalar
    compute_inverse_matrix_up(
      const Eigen::MatrixBase<Derived> &B,
      const Eigen::MatrixBase<Derived> &C,
      const Eigen::MatrixBase<Derived> &D,
      ResizableMatrix<Scalar> &invA) {
      typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> eigen_matrix_t;
      typedef Eigen::Block<eigen_matrix_t> block_t;

      const int N = num_rows(invA);
      const int M = num_rows(D);

      assert(M > 0);
      assert(num_rows(invA) == num_cols(invA));
      assert(num_rows(B) == N && num_cols(B) == M);
      assert(num_rows(C) == M && num_cols(C) == N);
      assert(num_rows(D) == M && num_cols(D) == M);

      if (N == 0) {
        invA = D.inverse();
        return D.determinant();
      } else {
        //I don't know how expensive to allocate temporary objects C_invA, H, invA_B, F.
        //We could keep them as static objects or members of a class.

        //compute H
        const eigen_matrix_t C_invA = C * invA.block();
        const eigen_matrix_t H = (D - C_invA * B).inverse();

        //compute F
        const eigen_matrix_t invA_B = invA.block() * B;
        const eigen_matrix_t F = -invA_B * H;

        invA.conservative_resize(N + M, N + M);//this keeps the contents in the left corner of invA

        //compute G
        invA.block(N, 0, M, N) = -H * C_invA;

        //compute E
        invA.block(0, 0, N, N) -= invA_B * invA.block(N, 0, M, N);

        invA.block(0, N, N, M) = F;
        invA.block(N, N, M, M) = H;

        return 1. / H.determinant();
      }
    }

    /**
     * Compute the determinant ratio for the removal of rows and cols
     * We implement equations in Appendix B.1.1 of Luitz's thesis.
     * https://opus.bibliothek.uni-wuerzburg.de/files/6408/thesis_luitz.pdf
     *
     * For a certain matrix G, its inverse is denoted by G^{-1}
     * Let us consider removing several rows and columns in G.
     * The resultant matrix is G'.
     * As mentioned below, some of rows and columns in G' are exchanged.
     * In this function, we compute |G'|/|G|, which includes the sign change due to the permutations of rows and columns.
     * Note that swapping rows/columns in a matrix corresponds to
     * swapping the corresponding columns/rows in its inverse, respectively.
     * (G: row <=> G^-1: column)
     *
     * @param num_rows_cols_removed number of rows and cols to be removed in G
     * @param rows_removed positions of rows to be removed in G (not G^{-1}). The first num_rows_cols_removed elements are referred.
     * @param cols_removed positions of cols to be removed in G (not G^{-1}). The first num_rows_cols_removed elements are referred.
     * @param invG inverse of the currrent matrix: G^{-1}
     */
    template<class Scalar>
    Scalar
    compute_det_ratio_down(
      const int num_rows_cols_removed,
      const std::vector<int> &rows_removed,
      const std::vector<int> &cols_removed,
      const ResizableMatrix<Scalar> &invG) {
      typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> eigen_matrix_t;

      const int NpM = num_rows(invG);
      const int M = num_rows_cols_removed;
      assert(num_cols(invG) == NpM);
      assert(rows_removed.size() >= M);
      assert(M > 0);

      //Note: if rows_removed==cols_removed, there is no sign change.
      // Thus, we count the difference from this: perm;
      eigen_matrix_t H(M, M);
      unsigned long perm = 0;
      for (int j = 0; j < M; ++j) {
        perm += std::abs(rows_removed[j] - cols_removed[j]);
        for (int i = 0; i < M; ++i) {
          assert(cols_removed[i] < NpM);
          assert(rows_removed[j] < NpM);
          H(i, j) = invG(cols_removed[i], rows_removed[j]);//Note: rows and cols are inverted.
        }
      }

      return perm % 2 == 0 ? H.determinant() : -H.determinant();
    }

    /**
     * Update the inverse matrix for the removal of rows and cols
     * We implement equations in Appendix B.1.1 of Luitz's thesis.
     * https://opus.bibliothek.uni-wuerzburg.de/files/6408/thesis_luitz.pdf
     *
     * The actual procedure is the following.
     * First, we move all rows and cols to be removed to the last (step1).
     * Then, we remove them (step2).
     * On exit, the positions of some remaining rows and cols are exchanged in step1.
     *
     * @param num_rows_cols_removed number of rows and cols to be removed in G
     * @param rows_removed positions of rows to be removed in G (not G^{-1}). The first num_rows_cols_removed elements are referred.
     * @param cols_removed positions of cols to be removed in G (not G^{-1}). The first num_rows_cols_removed elements are referred.
     * @param invG G^{-1}. invG will be resized and updated to G'^{-1}.
     * @param swapped_rows a list of pairs of rows in G swapped in step 1
     * @param swapped_cols a list of pairs of cols in G swapped in step 1
     */
    template<class Scalar>
    void
    compute_inverse_matrix_down(
      const int num_rows_cols_removed,
      const std::vector<int> &rows_removed,
      const std::vector<int> &cols_removed,
      ResizableMatrix<Scalar> &invG
    ) {
      typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> eigen_matrix_t;

      const int NpM = num_rows(invG);
      const int M = num_rows_cols_removed;
      const int N = NpM - M;
      assert(num_cols(invG) == NpM);
      assert(rows_removed.size() >= M);
      assert(cols_removed.size() >= M);
      assert(M > 0);
      assert(NpM >= M);

      if (NpM < M) {
        throw std::logic_error("N should not be negative!");
      }

      if (M == 0) {
        throw std::logic_error("M should be larger than 0!");
      }

#ifndef NDEBUG
      //make sure the indices are in ascending order.
      for (int idel = 0; idel < M - 1; ++idel) {
        assert(rows_removed[idel] < rows_removed[idel + 1]);
        assert(cols_removed[idel] < cols_removed[idel + 1]);
      }
#endif

      //Step 1: move rows and cols to be removed to the end.
      for (int idel = 0; idel < M; ++idel) {
        //Note: If we swap two rows in G, this corresponds to swapping the corresponding COLUMNS in G^{-1}
        invG.swap_col(rows_removed[M - 1 - idel], NpM - 1 - idel);
        invG.swap_row(cols_removed[M - 1 - idel], NpM - 1 - idel);
      }

      //Step 2: update the inverse matrix and shrink it.
      if (N > 0) {
        //E -= F*H^{-1}*G
        invG.block(0, 0, N, N) -=
          invG.block(0, N, N, M) *
          invG.block(N, N, M, M).inverse() *
          invG.block(N, 0, M, N);
      }
      invG.conservative_resize(N, N);
    }

    //Implementing Ye-Hua Lie and Lei Wang (2015): Eqs. (17)-(26) before taking the limit of tS->0
    template<typename Scalar, typename M0, typename M1, typename M2>
    class ReplaceHelper {
    public:
      ReplaceHelper(ResizableMatrix<Scalar>& invG, const M0& R, const M1& Q, const M2& S);
      Scalar compute_det_ratio();
      void compute_inverse_matrix();

    private:
      typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> eigen_matrix_t;
      typedef Eigen::Block<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> > block_t;

      ResizableMatrix<Scalar>& invG_;
      const M0& Q_;
      const M1& R_;
      const M2& S_;
      const int N_, M_, M_old_;

      eigen_matrix_t Mmat_, inv_tSp_;
    };
  }
}

#include "fastupdate_formula.ipp"

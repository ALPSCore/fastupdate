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
      const alps::ResizableMatrix<Scalar> &invA) {
      const size_t N = num_rows(invA);
      const size_t M = num_rows(D);

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
      alps::ResizableMatrix<Scalar> &invA) {
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
      const alps::ResizableMatrix<Scalar> &invG) {
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
      alps::ResizableMatrix<Scalar> &invG,
      std::vector<std::pair<int, int> > &swapped_rows,
      std::vector<std::pair<int, int> > &swapped_cols
    ) {
      typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> eigen_matrix_t;

      const int NpM = num_rows(invG);
      const int M = num_rows_cols_removed;
      const int N = NpM - M;
      assert(num_cols(invG) == NpM);
      assert(rows_cols_removed.size() >= M);
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
        assert(rows_cols_removed[idel] < rows_cols_removed[idel + 1]);
      }
#endif

      //Step 1: move rows and cols to be removed to the end.
      swapped_rows.resize(M);
      swapped_cols.resize(M);
      for (int idel = 0; idel < M; ++idel) {
        //Note: If we swap two rows in G, this corresponds to swapping the corresponding COLUMNS in G^{-1}
        invG.swap_cols(rows_removed[M - 1 - idel], NpM - 1 - idel);
        invG.swap_rows(cols_removed[M - 1 - idel], NpM - 1 - idel);
        swapped_rows[idel] = std::pair<int, int>(rows_removed[M - 1 - idel], NpM - 1 - idel);
        swapped_cols[idel] = std::pair<int, int>(cols_removed[M - 1 - idel], NpM - 1 - idel);
      }

      //Step 2: update the inverse matrix and shrink it.
      if (N == 0) {
        invG.resize(0, 0);
      } else {
        //E -= F*H^{-1}*G
        invG.block(0, 0, N, N) -=
          invBigmat.block(0, N, N, M) *
          invG.block(N, N, M, M).inverse() *
          invG.block(N, 0, M, N);
        invG.conservative_resize(N, N);
      }
    }

/*
template<class T>
T
compute_inverse_matrix_replace_row_col(alps::ResizableMatrix<T>& invG, const alps::ResizableMatrix<T> Dr, const alps::ResizableMatrix<T> Dc, int m,
    bool assume_intermediate_state_is_singular=false) {
    typedef alps::ResizableMatrix<T> matrix_t;
    const double eps = 1E-10;

    const int N = num_cols(invG);
    const int Nm1 = N-1;
    assert(num_rows(invG)==N);
    assert(num_rows(Dr)==1);
    assert(num_cols(Dr)==N);
    assert(num_rows(Dc)==N);
    assert(num_cols(Dc)==1);
    assert(Dr(0,m)==Dc(m,0));

    //original G^{-1}
    matrix_t tP(Nm1, Nm1), tQ(Nm1, 1), tR(1, Nm1);
    invG.swap_cols(m,N-1);
    invG.swap_rows(m,N-1);
    copy_block(invG, 0, 0, tP, 0, 0, Nm1, Nm1);
    copy_block(invG, 0, Nm1, tQ, 0, 0, Nm1, 1);
    copy_block(invG, Nm1, 0, tR, 0, 0, 1, Nm1);
    const T tS = invG(N-1,N-1);

    //new G
    matrix_t Q(Dc), R(Dr);
    const T S = Dc(m,0);
    Q.swap_rows(m,N-1); Q.resize(Nm1,1);
    R.swap_cols(m,N-1); R.resize(1,Nm1);

    //compute lambda
    T lambda = tS*(S-mygemm(R,mygemm(tP,Q))(0,0)) +  mygemm(mygemm(R,tQ),mygemm(tR,Q))(0,0);
    const T tSp = tS/lambda;
    //std::cout << "lamba split " << tS*(S-mygemm(R,mygemm(tP,Q))(0,0)) << " " <<  mygemm(mygemm(R,tQ),mygemm(tR,Q))(0,0) << std::endl;
    //std::cout << "debug tS/tSp " << tS << " " << tSp << std::endl;
    if (assume_intermediate_state_is_singular || (std::abs(tS)<eps && std::abs(tSp)<eps) ) {
        const double R_tQ = mygemm(R, tQ)(0,0);
        const double tR_Q = mygemm(tR, Q)(0,0);
        matrix_t tQp(tQ); tQp /= R_tQ;
        matrix_t tRp(tR); tRp /= tR_Q;

        matrix_t tPp(tP);
        tPp += -mygemm(tQp,mygemm(R,tP))-mygemm(mygemm(tP,Q),tRp)+mygemm(mygemm(tQp,mygemm(R,mygemm(tP,Q))),tRp);

        copy_block(tPp, 0, 0, invG, 0, 0, Nm1, Nm1);
        copy_block(tQp, 0, 0, invG, 0, Nm1, Nm1, 1);
        copy_block(tRp, 0, 0, invG, Nm1, 0, 1, Nm1);
        invG(N-1,N-1) = tSp;

        invG.swap_cols(m,N-1);
        invG.swap_rows(m,N-1);

        return lambda;
    } else {
        matrix_t Mmat = mygemm(tQ,tR);
        Mmat *= -1/tS;
        Mmat += tP;

        const matrix_t MQ = mygemm(Mmat, Q);
        const matrix_t RM = mygemm(R, Mmat);
        const matrix_t tQp = (-tS/lambda)*MQ;
        const matrix_t tRp = (-tS/lambda)*RM;
        const matrix_t tPp = Mmat+(tS/lambda)*mygemm(MQ, RM);

        copy_block(tPp, 0, 0, invG, 0, 0, Nm1, Nm1);
        copy_block(tQp, 0, 0, invG, 0, Nm1, Nm1, 1);
        copy_block(tRp, 0, 0, invG, Nm1, 0, 1, Nm1);
        invG(N-1,N-1) = tSp;

        invG.swap_cols(m,N-1);
        invG.swap_rows(m,N-1);

        return lambda;
    }
}

//Assuming the intermediate state is singular, one replaces a row and a column.
template<class T>
T
compute_inverse_matrix_replace_row_col2(alps::ResizableMatrix<T>& invG, const alps::ResizableMatrix<T>& Dr, const alps::ResizableMatrix<T>& Dc, int m,
                                       bool compute_only_det_rat) {
    typedef alps::ResizableMatrix<T> matrix_t;
    const double eps = 1E-10;

    const int N = num_cols(invG);
    const int Nm1 = N-1;
    assert(num_rows(invG)==N);
    assert(num_rows(Dr)==1);
    assert(num_cols(Dr)==N);
    assert(num_rows(Dc)==N);
    assert(num_cols(Dc)==1);
    assert(Dr(0,m)==Dc(m,0));

    //original G^{-1}
    matrix_t tQ(Nm1, 1), tR(1, Nm1);
    invG.swap_cols(m,N-1);
    invG.swap_rows(m,N-1);
    copy_block(invG, 0, Nm1, tQ, 0, 0, Nm1, 1);
    copy_block(invG, Nm1, 0, tR, 0, 0, 1, Nm1);
    const T tS = invG(N-1,N-1);

    //new G
    matrix_t Q(Dc), R(Dr);
    Q.swap_rows(m,N-1); Q.resize(Nm1,1);
    R.swap_cols(m,N-1); R.resize(1,Nm1);

    const T R_tQ = mygemm(R, tQ)(0,0);
    const T tR_Q = mygemm(tR, Q)(0,0);
    const T lambda = R_tQ*tR_Q;

    if (compute_only_det_rat) {
        invG.swap_cols(m,N-1);
        invG.swap_rows(m,N-1);
        return lambda;
    }

    matrix_t tP(Nm1, Nm1);
    copy_block(invG, 0, 0, tP, 0, 0, Nm1, Nm1);

    matrix_t tQp(tQ); tQp /= R_tQ;
    matrix_t tRp(tR); tRp /= tR_Q;

    matrix_t tPp(tP);
    tPp += -mygemm(tQp,mygemm(R,tP))-mygemm(mygemm(tP,Q),tRp)+mygemm(mygemm(tQp,mygemm(R,mygemm(tP,Q))),tRp);

    copy_block(tPp, 0, 0, invG, 0, 0, Nm1, Nm1);
    copy_block(tQp, 0, 0, invG, 0, Nm1, Nm1, 1);
    copy_block(tRp, 0, 0, invG, Nm1, 0, 1, Nm1);
    const T tSp = tS/lambda;
    invG(N-1,N-1) = tSp;

    invG.swap_cols(m,N-1);
    invG.swap_rows(m,N-1);

    return lambda;
}


template<class T>
T
compute_inverse_matrix_replace_rows_cols_succesive(alps::ResizableMatrix<T>& invG, const alps::ResizableMatrix<T>& Q, const alps::ResizableMatrix<T>& R, const alps::ResizableMatrix<T>& S,
                                                    const std::vector<int>& rows_cols, bool assume_intermediate_state_is_singular=false) {
    typedef alps::ResizableMatrix<T> matrix_t;

    const int N = num_cols(invG);
    const int M = rows_cols.size();
    //const int Nm1 = N - 1;
    assert(num_rows(invG)==N);
    assert(num_rows(Q)==N-M && num_cols(Q)==M);
    assert(num_rows(R)==M && num_cols(R)==N-M);
    assert(num_rows(S)==M && num_cols(S)==M);
#ifndef NDEBUG
    for(int im=0; im<rows_cols.size()-1; ++im) {
        assert(rows_cols[im]<rows_cols[im+1]);
    }
#endif

    matrix_t Dr(1,N), Dc(N,1);
    //std::vector<int> idx_in_R(M);
    std::vector<bool> is_included(N,false);
    for (int im=0; im<M; ++im) {
        is_included[rows_cols[im]] = true;
    }
    //std::vector<int> rows_cols_rest(N-M);
    //int idx = 0;
    //for (int i=0; i<N; ++i) {
        //if (!is_included[i]) {
            //rows_cols_rest[idx] = i;
            //++idx;
        //}
    //}

    T lambda = 1.;
    for (int im=0; im<M; ++im) {
        int idx=0, idx2=0;
        for (int i=0; i<N; ++i) {
            if (!is_included[i]) {
                Dr(0,i) = R(im,idx);
                Dc(i,0) = Q(idx,im);
                ++idx;
            } else {
                Dr(0,i) = S(im,idx2);
                Dc(i,0) = S(idx2,im);
                ++idx2;
            }
        }
        assert(idx==N-M);
        assert(idx2==M);

        double rtmp = compute_inverse_matrix_replace_row_col(invG, Dr, Dc, rows_cols[im], assume_intermediate_state_is_singular);
        //std::cout << " im = " << im << " " << rtmp << std::endl;
        lambda *= rtmp;
    }
    return lambda;
}

template<class T>
T
compute_inverse_matrix_replace_single_row_col(alps::ResizableMatrix<T>& invG, const alps::ResizableMatrix<T>& Q, const alps::ResizableMatrix<T>& R, const alps::ResizableMatrix<T>& S,
                                                   const std::vector<int>& rows_cols, bool compute_only_det) {
    typedef alps::ResizableMatrix<T> matrix_t;

    const int N = num_cols(invG);
    const int M = rows_cols.size();
    if (M!=1)
        throw std::logic_error("Error in compute_inverse_matrix_replace_single_row_col");

    assert(num_rows(invG)==N);
    assert(num_rows(Q)==N-M && num_cols(Q)==M);
    assert(num_rows(R)==M && num_cols(R)==N-M);
    assert(num_rows(S)==M && num_cols(S)==M);

    matrix_t Dr(1,N), Dc(N,1);
    std::vector<bool> is_included(N,false);
    for (int im=0; im<M; ++im) {
        is_included[rows_cols[im]] = true;
    }

    T lambda = 1.;
    for (int im=0; im<M; ++im) {
        int idx=0, idx2=0;
        for (int i=0; i<N; ++i) {
            if (!is_included[i]) {
                Dr(0,i) = R(im,idx);
                Dc(i,0) = Q(idx,im);
                ++idx;
            } else {
                Dr(0,i) = S(im,idx2);
                Dc(i,0) = S(idx2,im);
                ++idx2;
            }
        }
        assert(idx==N-M);
        assert(idx2==M);

        T rtmp = compute_inverse_matrix_replace_row_col2(invG, Dr, Dc, rows_cols[im], compute_only_det);
        lambda *= rtmp;
    }
    return lambda;
}


template<class T>
void
replace_rows_cols(alps::ResizableMatrix<T>& A,
                  const alps::ResizableMatrix<T>& Q, const alps::ResizableMatrix<T>& R, const alps::ResizableMatrix<T>& S,
                  const std::vector<int>& rows_cols) {
    using namespace alps::numeric;
    typedef matrix<T> matrix_t;

    const int NpM = num_cols(A);
    const int M = rows_cols.size();
    const int N = NpM-M;

    std::vector<std::pair<int,int> > swap_list(M);
    for (int i=0; i<M; ++i) {
        A.swap_cols(rows_cols[M-1-i], NpM-1-i);
        A.swap_rows(rows_cols[M-1-i], NpM-1-i);
        swap_list[i] = std::pair<int,int>(rows_cols[M-1-i], NpM-1-i);
    }

    copy_block(Q, 0, 0, A, 0, N, N, M);
    copy_block(R, 0, 0, A, N, 0, M, N);
    copy_block(S, 0, 0, A, N, N, M, M);

    for(std::vector<std::pair<int,int> >::reverse_iterator it=swap_list.rbegin(); it!=swap_list.rend(); ++it) {
        A.swap_cols(it->first, it->second);
        A.swap_rows(it->first, it->second);
    }
}

template<class T>
void generate_indices(const std::vector<T>& rows_cols, int N, int M, std::vector<T>& rows_cols_rest) {
    assert(rows_cols.size()==M);
    const int NpM = N+M;

    std::vector<bool> is_included(N+M,false);
    for (int im=0; im<M; ++im) {
        is_included[rows_cols[im]] = true;
    }

    rows_cols_rest.resize(N);
    int idx = 0;
    for (int i=0; i<NpM; ++i) {
        if (!is_included[i]) {
            rows_cols_rest[idx]  = i;
            ++idx;
        }
    }
    assert(idx==N);
}

template<class T>
void
replace_rows_cols_respect_ordering(alps::ResizableMatrix<T>& A,
                  const alps::ResizableMatrix<T>& Q, const alps::ResizableMatrix<T>& R, const alps::ResizableMatrix<T>& S,
                  const std::vector<int>& rows_cols) {

    const int NpM = num_cols(A);
    const int M = rows_cols.size();
    const int N = NpM-M;

    std::vector<bool> is_included(N+M,false);
    for (int im=0; im<M; ++im) {
        is_included[rows_cols[im]] = true;
    }
    std::vector<int> pos_N(N), pos_M(M);
    int idx_N=0, idx_M=0;
    for (int i=0; i<NpM; ++i) {
        if (is_included[i]) {
            pos_M[idx_M] = i;
            ++idx_M;
        } else {
            pos_N[idx_N] = i;
            ++idx_N;
        }
    }
    assert(idx_N==N && idx_M==M);

    for (int j=0; j<M; ++j) {
        for (int i=0; i<M; ++i) {
            A(pos_M[i],pos_M[j]) =  S(i,j);
        }
    }
    for (int j=0; j<M; ++j) {
        for (int i=0; i<N; ++i) {
            A(pos_N[i],pos_M[j]) =  Q(i,j);
            A(pos_M[j],pos_N[i]) =  R(j,i);
        }
    }
}

//Implementing Ye-Hua Lie and Lei Wang (2015): Eqs. (17)-(26) without taking the limit of tS->0
//T = double or complex<double>
template<class T>
T
compute_det_ratio_replace_rows_cols(const alps::ResizableMatrix<T>& invBigMat,
                               const alps::ResizableMatrix<T>& Q, const alps::ResizableMatrix<T>& R, const alps::ResizableMatrix<T>& S,
                               alps::ResizableMatrix<T>& Mmat, alps::ResizableMatrix<T>& inv_tSp) {
    //const std::vector<int>& rows_cols, const std::vector<std::pair<int,int> >& swap_list, alps::ResizableMatrix<T>& Mmat, alps::ResizableMatrix<T>& inv_tSp) {
    using namespace alps::numeric;
    typedef matrix<T> matrix_t;

    static matrix_t invtS_tR, inv_tS, MQ, RMQ, ws1, ws2;

    const int N = num_cols(R);
    const int M = num_rows(R);
    const int M_old = num_cols(invBigMat)-N;

    if (N==0) {
      ws1.destructive_resize(M,M);
      ws2.destructive_resize(M_old,M_old);
      return determinant_no_copy(S, ws1)*determinant_no_copy(invBigMat, ws2);
    }

    assert(N>0);
    assert(num_cols(invBigMat)==num_rows(invBigMat));
    assert(num_rows(R)==M && num_cols(R)==N);
    assert(num_rows(Q)==N && num_cols(Q)==M);
    assert(num_rows(S)==M && num_cols(S)==M);

    inv_tS.destructive_resize(M_old,M_old);
    invtS_tR.destructive_resize(M_old,N);
    MQ.destructive_resize(N,M);
    RMQ.destructive_resize(M,M);
    ws1.destructive_resize(M_old,M_old);
    ws2.destructive_resize(M,M);

    submatrix_view<T> tQ_view(invBigMat, 0, N, N, M_old);
    submatrix_view<T> tR_view(invBigMat, N, 0, M_old, N);
    submatrix_view<T> tS_view(invBigMat, N, N, M_old, M_old);

    //compute inv_tS
    copy_block(invBigMat, N, N, inv_tS, 0, 0, M_old, M_old);
    inverse_in_place(inv_tS);

    gemm(inv_tS, tR_view, invtS_tR);

    Mmat.destructive_resize(N,N);
    copy_block(invBigMat, 0, 0, Mmat, 0, 0, N, N);
    mygemm((T)-1.0, tQ_view, invtS_tR, (T) 1.0, Mmat);

    inv_tSp.destructive_resize(M,M);
    gemm(Mmat, Q, MQ);
    copy_block(S, 0, 0, inv_tSp, 0, 0, M, M);
    mygemm((T) -1.0, R, MQ, (T) 1.0, inv_tSp);
    return determinant_no_copy(tS_view, ws1)*determinant_no_copy(inv_tSp, ws2);
}

//Implementing Ye-Hua Lie and Lei Wang (2015): Eqs. (17)-(26) before taking the limit of tS->0
//T = double or complex<double>
template<class T>
void
compute_inverse_matrix_replace_rows_cols(alps::ResizableMatrix<T>& invBigMat,
                                    const alps::ResizableMatrix<T>& Q, const alps::ResizableMatrix<T>& R, const alps::ResizableMatrix<T>& S,
                                    const alps::ResizableMatrix<T>& Mmat, const alps::ResizableMatrix<T>& inv_tSp) {
    using namespace alps::numeric;
    typedef matrix<T> matrix_t;

    static matrix_t tmp_NM, tmp_MN;

    const int N = num_cols(R);
    const int M = num_rows(R);
    const int M_old = num_cols(invBigMat)-N;

    assert(num_cols(invBigMat)==num_rows(invBigMat));
    assert(num_rows(R)==M && num_cols(R)==N);
    assert(num_rows(Q)==N && num_cols(Q)==M);
    assert(num_rows(S)==M && num_cols(S)==M);

    if (N==0) {
        invBigMat.destructive_resize(M, M);
        my_copy_block(S, 0, 0, invBigMat, 0, 0, M, M);
        inverse_in_place(invBigMat);
        return;
    }

    assert(N>0);
    tmp_NM.destructive_resize(N,M);
    tmp_MN.destructive_resize(M,N);
    invBigMat.destructive_resize(N+M, N+M);
    submatrix_view<T> tPp_view(invBigMat,0,0,N,N);
    submatrix_view<T> tQp_view(invBigMat,0,N,N,M);
    submatrix_view<T> tRp_view(invBigMat,N,0,M,N);
    submatrix_view<T> tSp_view(invBigMat,N,N,M,M);

    //tSp
    my_copy_block(inv_tSp, 0, 0, tSp_view, 0, 0, M, M);
    inverse_in_place(tSp_view);

    //tQp
    gemm(Q,tSp_view,tmp_NM);
    mygemm((T)-1.0, Mmat, tmp_NM, (T) 0.0, tQp_view);

    //tRp
    gemm(tSp_view,R,tmp_MN);
    mygemm((T)-1.0, tmp_MN, Mmat, (T) 0.0, tRp_view);

    //tPp
    gemm(Mmat, Q, tmp_NM);
    my_copy_block(Mmat, 0, 0, tPp_view, 0, 0, N, N);
    mygemm((T)-1.0, tmp_NM, tRp_view, (T) 1.0, tPp_view);
}

template<class T>
T
compute_det_ratio_replace_rows_cols_safe(const alps::ResizableMatrix<T>& invBigMat,
                                    const alps::ResizableMatrix<T>& Q, const alps::ResizableMatrix<T>& R, const alps::ResizableMatrix<T>& S,
                                    alps::ResizableMatrix<T>& Mmat, alps::ResizableMatrix<T>& inv_tSp) {
    using namespace alps::numeric;
    typedef matrix<T> matrix_t;

    const int NpM = num_cols(invBigMat);
    const int M = num_rows(R);
    const int N = NpM-M;
    assert(N>0);
    assert(M==1);

    assert(num_cols(invBigMat)==num_rows(invBigMat));
    assert(num_rows(R)==M && num_cols(R)==N);
    assert(num_rows(Q)==N && num_cols(Q)==M);
    assert(num_rows(S)==M && num_cols(S)==M);

    matrix_t tP(N, N), tQ(N, M), tR(M, N), tS(M, M), invtS_tR(M,N,0.), tQ_invtS_tR(N,N,0.);

    copy_block(invBigMat, 0, 0, tP, 0, 0, N, N);
    copy_block(invBigMat, 0, N, tQ, 0, 0, N, M);
    copy_block(invBigMat, N, 0, tR, 0, 0, M, N);
    copy_block(invBigMat, N, N, tS, 0, 0, M, M);

    return (mygemm(tS, S-mygemm(R,mygemm(tP,Q)))+mygemm(mygemm(R,tQ),mygemm(tR,Q)))(0,0);

    //gemm(inverse(tS), tR, invtS_tR);
    //gemm(tQ, invtS_tR, tQ_invtS_tR);
    //Mmat = tP-tQ_invtS_tR;

    //matrix_t MQ(N,M,0.), RMQ(M,M,0.);//, inv_tSp(M,M);
    //gemm(Mmat, Q, MQ);
    //gemm(R, MQ, RMQ);
    //inv_tSp = S-RMQ;
    //return determinant(tS)*determinant(inv_tSp);
}

template<typename T>
T
compute_det_ratio_replace_diaognal_elements(alps::ResizableMatrix<T>& invBigMat, int num_elem_updated, const std::vector<int>& pos, const std::vector<T>& elems_diff, bool compute_only_det_rat) {
    using namespace alps::numeric;

    assert(invBigMat.num_cols()==invBigMat.num_rows());

    //work space (static!)
    static matrix<T> x, invC_plus_x, invC_plus_x_times_zx, yx;
    static std::vector<std::pair<int,int> > swap_list;

    const int N = invBigMat.num_cols();

    x.destructive_resize(num_elem_updated, num_elem_updated);
    for (int j=0; j<num_elem_updated; ++j) {
        for (int i=0; i<num_elem_updated; ++i) {
            assert(pos[i]>=0 && pos[i]<invBigMat.num_cols());
            assert(pos[j]>=0 && pos[j]<invBigMat.num_cols());
            x(i,j) = invBigMat(pos[i],pos[j])*elems_diff[j];
        }
        x(j,j) += (T) 1;
    }

    const T det_rat = determinant(x);
    if (compute_only_det_rat) {
        return det_rat;
    }

    //swap rows and cols to move all diagonal elements to be updated to the end rows and cols
    swap_list.resize(num_elem_updated);
    for (int i=0; i<num_elem_updated; ++i) {
        invBigMat.swap_cols(pos[num_elem_updated-1-i], N-1-i);
        invBigMat.swap_rows(pos[num_elem_updated-1-i], N-1-i);
        swap_list[i] = std::pair<int,int>(pos[num_elem_updated-1-i], N-1-i);
    }
    yx.destructive_resize(N, num_elem_updated);
    copy_block(invBigMat, 0, N-num_elem_updated, yx, 0, 0, N, num_elem_updated);

    submatrix_view<T> zx_view(invBigMat, N-num_elem_updated, 0,     num_elem_updated, N);
    invC_plus_x.destructive_resize(num_elem_updated, num_elem_updated);
    invC_plus_x_times_zx.destructive_resize(num_elem_updated, N);

    copy_block(invBigMat, N-num_elem_updated, N-num_elem_updated, invC_plus_x, 0, 0, num_elem_updated, num_elem_updated);
    for (int i=0; i<num_elem_updated; ++i) {
        invC_plus_x(i,i) += 1.0/elems_diff[i];
    }
    inverse_in_place(invC_plus_x);
    mygemm((T) 1.0, invC_plus_x, zx_view, (T) 0.0, invC_plus_x_times_zx);
    mygemm((T) -1.0, yx, invC_plus_x_times_zx, (T) 1.0, invBigMat);

    for(std::vector<std::pair<int,int> >::reverse_iterator it=swap_list.rbegin(); it!=swap_list.rend(); ++it) {
        invBigMat.swap_cols(it->first, it->second);
        invBigMat.swap_rows(it->first, it->second);
    }

    return det_rat;
}

*/

  }
}

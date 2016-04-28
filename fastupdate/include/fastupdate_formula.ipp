/**
 * Fast-update formula based on block matrix representation
 * Copyright (C) 2016 by Hiroshi Shinaoka <h.shinaoka@gmail.com>
 *
 */

#include "fastupdate_formula.h"

namespace alps {
  namespace fastupdate {

    /*
     * Definition of ReplaceHelper
     */
    template<typename Scalar, typename M0, typename M1, typename M2>
    ReplaceHelper<Scalar,M0,M1,M2>::ReplaceHelper(ResizableMatrix<Scalar>& invG, const M0& Q, const M1& R, const M2& S) :
    invG_(invG), Q_(Q), R_(R), S_(S), N_(num_cols(R)), M_(num_rows(R)), M_old_(num_cols(invG)-N_) {
      assert(num_cols(invG_)==num_rows(invG_));
      assert(num_rows(R_)==M_ && num_cols(R_)==N_);
      assert(num_rows(Q_)==N_ && num_cols(Q_)==M_);
      assert(num_rows(S_)==M_ && num_cols(S_)==M_);
    }

    template<typename Scalar, typename M0, typename M1, typename M2>
    Scalar ReplaceHelper<Scalar,M0,M1,M2>::compute_det_ratio() {
      if (N_ == 0) {
        return S_.determinant()*invG_.determinant();
      }

      block_t tP_view (invG_.block(0,  0,  N_,     N_     ));
      block_t tQ_view (invG_.block(0,  N_, N_,     M_old_ ));
      block_t tR_view (invG_.block(N_, 0,  M_old_, N_     ));
      block_t tS_view (invG_.block(N_, N_, M_old_, M_old_ ));

      //matrix M
      Mmat_ = tP_view - tQ_view * tS_view.inverse() * tR_view;

      //(tS')^{-1}
      inv_tSp_ = S_ - R_ * Mmat_ * Q_;

      return tS_view.determinant()*inv_tSp_.determinant();
    }

    template<typename Scalar, typename M0, typename M1, typename M2>
    void ReplaceHelper<Scalar,M0,M1,M2>::compute_inverse_matrix() {
      if (N_ == 0) {
        invG_.destructive_resize(M_, M_);
        invG_.block() = S_.inverse();
        return;
      }

      invG_.destructive_resize(N_ + M_, N_ + M_);
      block_t tPp_view(invG_.block(0,  0,  N_, N_));
      block_t tQp_view(invG_.block(0,  N_, N_, M_));
      block_t tRp_view(invG_.block(N_, 0,  M_, N_));
      block_t tSp_view(invG_.block(N_, N_, M_, M_));

      //tSp
      tSp_view = inv_tSp_.inverse();

      //tQp
      tQp_view = -Mmat_ * Q_ * tSp_view;

      //tRp
      tRp_view = -tSp_view * R_ * Mmat_;

      //tPp
      tPp_view = Mmat_ - Mmat_ * Q_ * tRp_view;
    }
  }
}

/**
 * Copyright (C) 2016 by Hiroshi Shinaoka <h.shinaoka@gmail.com>
 */
#pragma once

#include <boost/scoped_ptr.hpp>

#include <Eigen/Dense>

#include "determinant_matrix.hpp"
#include "fastupdate_formula.hpp"
#include "util.hpp"
#include "detail/clustering.hpp"

#include "determinant_matrix_base.hpp"

namespace alps {
  namespace fastupdate {

    /**
     * CdaggerOp and COp must have the following functionalities
     *   CdaggerOp::itime_type, COp::itime_type the type of time
     *
     *  Function itime_type operator_time(const CdaggerOp&) and operator_time(const COp&)
     *  Function int operator_flavor(const CdaggerOp&) and operator_flavor(const COp&)
     */
    template<
      typename Scalar,
      typename GreensFunction,
      typename CdaggerOp,
      typename COp
    >
    class DeterminantMatrixPartitioned
            : public DeterminantMatrixBase<
                    Scalar,
                    GreensFunction,
                    CdaggerOp,
                    COp,
                    DeterminantMatrixPartitioned<Scalar, GreensFunction, CdaggerOp, COp>
            >
    {
    private:
      typedef std::vector<CdaggerOp> cdagg_container_t;
      typedef std::vector<COp> c_container_t;
      typedef Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> eigen_matrix_t;
      typedef DeterminantMatrixBase<
              Scalar,
              GreensFunction,
              CdaggerOp,
              COp,
              DeterminantMatrixPartitioned<Scalar, GreensFunction, CdaggerOp, COp>
            > Base;

      enum State {
        waiting = 0,
        try_add_called = 1,
        try_rem_called = 2,
        try_rem_add_called = 3,
        try_replace_cdagg_called = 4,
        try_replace_c_called = 5
      };

      //for time-ordering in each sector
      template<typename T>
      struct CompareWithinSectors {
        bool operator()(const std::pair<int,T>& t1, const std::pair<int,T>& t2) const{
          if (t1.first == t2.first) {
            return operator_time(t1.second) < operator_time(t2.second);
          } else {
            return t1.first < t2.first;
          }
        }
      };

      //for time-ordering over sectors
      template<typename T>
      struct CompareOverSectors {
        bool operator()(const std::pair<int,T>& t1, const std::pair<int,T>& t2) const {
          return operator_time(t1.second) < operator_time(t2.second);
        }
      };

    public:
      DeterminantMatrixPartitioned (
        const GreensFunction& gf
      );

      template<typename CdaggCIterator>
      DeterminantMatrixPartitioned (
        const GreensFunction& gf,
        CdaggCIterator first,
        CdaggCIterator last
      );

      //size of matrix
      inline int size() const {return cdagg_ops_ordered_in_sectors_.size();};

      //Getter: costs O(N)
      inline const cdagg_container_t& get_cdagg_ops() const {
        return cdagg_ops_;
      }

      //Getter: costs O(N)
      inline const c_container_t& get_c_ops() const {
        return c_ops_;
      }

      /**
       * Compute determinant. This may suffer from overflow
       */
      inline Scalar compute_determinant() const {
        Scalar r = 1.0;
        for (int sector=0; sector<num_sectors_; ++sector) {
          r *= p_det_mat_[sector]->compute_determinant();
        }
        return (1.*permutation_)*r;
      }

      /**
       * Compute determinant as a product of Scalar
       */
      std::vector<Scalar> compute_determinant_as_product() const {
        std::vector<Scalar> r;
        for (int sector=0; sector<num_sectors_; ++sector) {
          const std::vector<Scalar>& vec = p_det_mat_[sector]->compute_determinant_as_product();
          std::copy(vec.begin(), vec.end(), std::back_inserter(r));
        }
        if (r.size() > 0) {
          r[0] *= permutation_;
        }
        return r;
      }

      /**
       * Compute inverse matrix. The rows and cols may not be time-ordered.
       */
      eigen_matrix_t compute_inverse_matrix() const {
        eigen_matrix_t inv(size(), size());
        inv.setZero();
        int offset = 0;
        for (int sector=0; sector<num_sectors_; ++sector) {
          int block_size = p_det_mat_[sector]->size();
          inv.block(offset, offset, block_size, block_size) = p_det_mat_[sector]->compute_inverse_matrix();
          offset += block_size;
        }
        return inv;
      }


      template<typename CdaggIterator, typename CIterator, typename CdaggIterator2, typename CIterator2>
      Scalar try_update(
        CdaggIterator  cdagg_rem_first,  CdaggIterator  cdagg_rem_last,
        CIterator      c_rem_first,      CIterator      c_rem_last,
        CdaggIterator2 cdagg_add_first,  CdaggIterator2 cdagg_add_last,
        CIterator2     c_add_first,      CIterator2     c_add_last
      );
      void perform_update();
      void reject_update();

      /**
       * Rebuild the matrix from scratch
       */
      ResizableMatrix<Scalar> build_matrix();

      /**
       * Rebuild the matrix from scratch
       */
      void rebuild_inverse_matrix() {
        for (int sector=0; sector<num_sectors_; ++sector) {
          p_det_mat_[sector]->rebuild_inverse_matrix();
        }
      }

    private:
      typedef DeterminantMatrix<Scalar,GreensFunction,CdaggerOp,COp> BlockMatrixType;

      /**
       * Try to remove some operators and add new operators
       * This function actually remove and insert operators in cdagg_ops_, c_ops_ but does not update the matrix.
       * After calling try_add(), either of perform_add() or reject_add() must be called.
       */
      template<typename CdaggCIterator>
      Scalar try_add(
        CdaggCIterator cdagg_c_add_first,
        CdaggCIterator cdagg_c_add_last
      );

      void perform_add() {perform_update();};

      void reject_add() {reject_update();};

      /**
       * Try to remove some operators and add new operators
       * This function actually remove and insert operators in cdagg_ops_, c_ops_ but does not update the matrix
       */
      template<typename CdaggCIterator>
      Scalar try_remove_add(
        CdaggCIterator cdagg_c_rem_first,
        CdaggCIterator cdagg_c_rem_last,
        CdaggCIterator cdagg_c_add_first,
        CdaggCIterator cdagg_c_add_last
      );

      /**
       *  Remove some operators and add new operators
       *  This function actually update the matrix
       */
      void perform_remove_add() {perform_update();};

      template<typename CdaggCIterator>
      void reject_remove_add(
        CdaggCIterator cdagg_c_rem_first,
        CdaggCIterator cdagg_c_rem_last,
        CdaggCIterator cdagg_c_add_first,
        CdaggCIterator cdagg_c_add_last
      ) {reject_update();};

      /**
       * Try to remove some operators
       * This function actually remove and insert operators in cdagg_ops_, c_ops_ but does not update the matrix
       */
      template<typename CdaggCIterator>
      Scalar try_remove(
        CdaggCIterator cdagg_c_rem_first,
        CdaggCIterator cdagg_c_rem_last
      );

      void perform_remove() {perform_update();};

      void reject_remove() {reject_update();};

      /**
       * Try to replace a creation operator
       */
      Scalar try_replace_cdagg(
              const CdaggerOp & old_cdagg,
              const CdaggerOp & new_cdagg
      );

      void perform_replace_cdagg() {perform_update();};

      void reject_replace_cdagg() {reject_update();};

      /**
       * Try to replace an annihilation operator
       */
      Scalar try_replace_c(
        const COp & old_c,
        const COp & new_c
      );

      void perform_replace_c() {perform_update();};

      void reject_replace_c() {reject_update();};


      State state_;

      int num_flavors_;                     //number of flavors
      int num_sectors_;                     //number of sectors
      std::vector<std::vector<int> > sector_members_;     //members of each sector
      std::vector<int> sector_belonging_to_; //remember to which sector each flavor belongs

      std::vector<boost::shared_ptr<BlockMatrixType> >  p_det_mat_;

      //permutation from a set that is time-ordered in each sector to a time-ordered set
      int permutation_;//1 or -1

      //a vector of creation and annihilation operators time-ordered in each sector
      //first element: sector
      //second element: operator
      std::vector<std::pair<int,CdaggerOp> > cdagg_ops_ordered_in_sectors_;
      std::vector<std::pair<int,COp> > c_ops_ordered_in_sectors_;
      //this is just a copy of second elements of cdagg_ops_ordered_in_sectors_ and c_ops_ordered_in_sectors_
      std::vector<CdaggerOp> cdagg_ops_;
      std::vector<COp> c_ops_;

      //for update
      int new_perm_;
      std::vector<std::pair<int,CdaggerOp> > cdagg_ops_work_;
      std::vector<std::pair<int,COp> > c_ops_work_;
      std::vector<std::vector<CdaggerOp> > cdagg_ops_add_, cdagg_ops_rem_;
      std::vector<std::vector<COp> > c_ops_add_, c_ops_rem_;

      void clear_work() {
        for (int sector=0; sector < num_sectors_; ++sector) {
          cdagg_ops_add_[sector].resize(0);
          cdagg_ops_rem_[sector].resize(0);
          c_ops_add_[sector].resize(0);
          c_ops_rem_[sector].resize(0);
        }
      }

      void init(const GreensFunction& gf);

      inline void check_state(State state) const {
        if (state_ != state) {
          throw std::logic_error("Error: the system is not in a correct state!");
        }
      }

      void sanity_check();

    };
  }
}

#include "determinant_matrix_partitioned.ipp"

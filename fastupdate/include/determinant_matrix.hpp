/**
 * Copyright (C) 2016 by Hiroshi Shinaoka <h.shinaoka@gmail.com>
 */
#pragma once

#include <set>

#include <boost/multi_index_container.hpp>
#include <boost/multi_index/sequenced_index.hpp>
#include <boost/multi_index/ordered_index.hpp>
#include <boost/multi_index/identity.hpp>
#include <boost/lambda/lambda.hpp>
#include <boost/lambda/if.hpp>
#include <boost/range/algorithm/adjacent_find.hpp>

#include <Eigen/Dense>

#include "fastupdate_formula.hpp"
#include "util.hpp"

namespace alps {
  namespace fastupdate {

    enum DeterminantMatrixState {
      waiting = 0,
      try_add_called = 1,
      try_rem_called = 2,
      try_rem_add_called = 3
    };

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
    class DeterminantMatrix {
    private:
      typedef typename CdaggerOp::itime_type itime_t;
      typedef std::vector<CdaggerOp> cdagg_container_t; //one can use range() with multi_index_container.
      typedef std::vector<COp> c_container_t; //one can use range() with multi_index_container.
      typedef std::map<itime_t,int> operator_map_t;

      typedef typename cdagg_container_t::iterator cdagg_it;
      typedef typename c_container_t::iterator c_it;
      BOOST_STATIC_ASSERT(boost::is_same<typename CdaggerOp::itime_type, typename COp::itime_type>::value);

      typedef Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> eigen_matrix_t;

    public:
      DeterminantMatrix(
        const GreensFunction& gf
      );

      template<typename CdaggCIterator>
      DeterminantMatrix(
        const GreensFunction& gf,
        CdaggCIterator first,
        CdaggCIterator last
      );

      //size of matrix
      inline int size() const {return inv_matrix_.size1();}

      //Getter
      inline const cdagg_container_t& get_cdagg_ops() const { return cdagg_ops_; }
      inline const c_container_t& get_c_ops() const { return c_ops_; }

      /**
       * Return a reference to the inverse matrix. The rows and columns may not be time-ordered.
       * The operators corresponding to the row and columns are given by get_cdagg_ops() and get_c_ops().
       * Note that the rows of the inverse matrix corresponds to the CREATION operators.
       */
      const ResizableMatrix<Scalar>& get_inverse_matrix_time_unordered() const {return inv_matrix_;}

      /**
       * Compute determinant. This may suffer from overflow
       */
      inline Scalar compute_determinant() const {
        return (1.*permutation_row_col_)/inv_matrix_.determinant();
      }

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

      void perform_add();

      void reject_add();

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
      void perform_remove_add();

      template<typename CdaggCIterator>
      void reject_remove_add(
        CdaggCIterator cdagg_c_rem_first,
        CdaggCIterator cdagg_c_rem_last,
        CdaggCIterator cdagg_c_add_first,
        CdaggCIterator cdagg_c_add_last
      );

      /**
       * Try to remove some operators
       * This function actually remove and insert operators in cdagg_ops_, c_ops_ but does not update the matrix
       */
      template<typename CdaggCIterator>
      Scalar try_remove(
        CdaggCIterator cdagg_c_rem_first,
        CdaggCIterator cdagg_c_rem_last
      );

      template<typename CdaggCIterator>
      void perform_remove(
        CdaggCIterator cdagg_c_rem_first,
        CdaggCIterator cdagg_c_rem_last
      );

      template<typename CdaggCIterator>
      void reject_remove(
        CdaggCIterator cdagg_c_rem_first,
        CdaggCIterator cdagg_c_rem_last
      );

      /**
       * Rebuild the matrix from scratch
       */
      ResizableMatrix<Scalar> build_matrix();

      /**
       * Rebuild the matrix from scratch
       */
      void rebuild_inverse_matrix();

      /**
       * Compute the inverse matrix for the time-ordered set of operators
       * This is only for debug.
       */
      eigen_matrix_t compute_inverse_matrix_time_ordered();

      void print_operators() const;

    private:
      DeterminantMatrixState state_;

      //inverse matrix
      ResizableMatrix<Scalar> inv_matrix_;

      //permutation of time-ordering of rows and cols
      int permutation_row_col_;//1 or -1

      //a vector of creation and annihilation operators in the order in which they appear in the rows and cols of the matrix
      cdagg_container_t cdagg_ops_;
      c_container_t c_ops_;

      const GreensFunction gf_;

      //key: the imaginary time of an operator, the index of row or col in the matrix
      operator_map_t cop_pos_, cdagg_op_pos_;

      //work space and helper
      int perm_rat_;
      std::vector<int> rem_cols_, rem_rows_;
      std::vector<std::pair<CdaggerOp,COp> > removed_op_pairs_;
      //std::vector<COp> removed_c_ops_;
      eigen_matrix_t G_n_n_, G_n_j_, G_j_n_;
      ReplaceHelper<Scalar,eigen_matrix_t,eigen_matrix_t,eigen_matrix_t> replace_helper_;

      /*
       * Private auxially functions
       */
    public:
      inline void check_state(DeterminantMatrixState state) const {
        if (state_ != state) {
          throw std::logic_error("Error: the system is not in a correct state!");
        }
      }

    private:

      /** swap cols of the matrix (and the rows of the inverse matrix)*/
      void swap_cdagg_op(int col1, int col2);

      /** swap rows of the matrix (and the cols of the inverse matrix)*/
      void swap_c_op(int row1, int row2);

      /** return if once can insert given operators. Note: duplicate members are not allowed in any configuration. */
      template<typename CdaggCIterator>
      bool insertion_possible(
        CdaggCIterator first,
        CdaggCIterator last
      );

      /** return if once can remove given operators, i.e, if they actually exist in the configuraiton */
      template<typename CdaggCIterator>
      bool removal_insertion_possible(
        CdaggCIterator first_removal,
        CdaggCIterator last_removal,
        CdaggCIterator first_insertion,
        CdaggCIterator last_insertion
      );

      /** return if once can remove given operators */
      template<typename CdaggCIterator>
      bool removal_possible(
        CdaggCIterator first_removal,
        CdaggCIterator last_removal
      ) const;

      /** add new operators and keeping the inverse matrix unchanged */
      template<typename Iterator>
      int add_new_operators(Iterator first,Iterator last);

      /** add operators and keeping the inverse matrix unchanged */
      int remove_last_operators(int num_operators_remove);

      /** remove excess operators, which were inserted by add_new_operators() */
      void remove_excess_operators();

      inline Scalar compute_g(int row, int col) const {return gf_(c_ops_[row], cdagg_ops_[col]); }

      /** return if there is an operator at a given time */
      inline bool exist(const CdaggerOp& cdagg) const {
        return exist(operator_time(cdagg));
      }

      /** return if there is an operator at a given time */
      inline bool exist(const COp& c) const {
        return exist(operator_time(c));
      }

      /** return if there is an operator at a given time */
      inline bool exist(itime_t time) const {
        return cop_pos_.find(time)!=cop_pos_.end() || cdagg_op_pos_.find(time)!=cdagg_op_pos_.end();
      }

      inline int find_cdagg(const CdaggerOp& cdagg) const {
        assert(cdagg_op_pos_.find(operator_time(cdagg))!=cdagg_op_pos_.end());
        return cdagg_op_pos_.at(operator_time(cdagg));
      }

      inline int find_c(const COp& c) const {
        assert(cop_pos_.find(operator_time(c))!=cop_pos_.end());
        return cop_pos_.at(operator_time(c));
      }

      void sanity_check() const;

    };
  }
}

#include "determinant_matrix.ipp"
#include "detail/determinant_matrix_add.ipp"
#include "detail/determinant_matrix_remove.ipp"
#include "detail/determinant_matrix_remove_add.ipp"

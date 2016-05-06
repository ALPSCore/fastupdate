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

#include <Eigen/Dense>

#include "fastupdate_formula.hpp"
#include "util.hpp"

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
    class DeterminantMatrix {
    private:
      typedef typename CdaggerOp::itime_type itime_t;
      typedef std::vector<CdaggerOp> cdagg_container_t; //one can use range() with multi_index_container.
      typedef std::vector<COp> c_container_t; //one can use range() with multi_index_container.
      //typedef boost::multi_index::multi_index_container<CdaggerOp> cdagg_container_t; //one can use range() with multi_index_container.
      //typedef boost::multi_index::multi_index_container<COp> c_container_t; //one can use range() with multi_index_container.

      typedef typename cdagg_container_t::iterator cdgg_it;
      typedef typename c_container_t::iterator c_it;
      BOOST_STATIC_ASSERT(boost::is_same<typename CdaggerOp::itime_type, typename COp::itime_type>::value);

      typedef Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> eigen_matrix_t;

    public:
      DeterminantMatrix(const GreensFunction& gf);

      /*
      template<typename CdaggIterator, typename CIterator>
      DeterminantMatrix(
        const GreensFunction& gf,
        CdaggIterator cdagg_begin,
        CdaggIterator cdagg_end,
        CIterator c_begin,
        CIterator c_end
      );
      */

      //size of matrix
      inline int size() const {return inv_matrix_.size1();}

      //Getter
      inline const cdagg_container_t& get_cdagg_ops() const { return cdagg_ops_; }
      inline const c_container_t& get_c_ops() const { return c_ops_; }

      /**
       * Compute determinant. This may suffer from overflow
       */
      inline Scalar compute_determinant() const {
        return (1.*permutation_row_col_)/inv_matrix_.determinant();
      }

      /**
       * Try to remove some operators and add new operators
       * This function actually remove and insert operators in cdagg_ops_, c_ops_ but does not update the matrix
       */
      Scalar try_remove_add(
        const std::vector<std::pair<CdaggerOp,COp> >& cdagg_c_rem,
        const std::vector<std::pair<CdaggerOp,COp> >& cdagg_c_add
      );

      /**
       *  Remove some operators and add new operators
       *  This function actually update the matrix
       */
      void perform_remove_add(
        const std::vector<std::pair<CdaggerOp,COp> >& cdagg_c_rem,
        const std::vector<std::pair<CdaggerOp,COp> >& cdagg_c_add
      );

      void reject_remove_add(
        const std::vector<std::pair<CdaggerOp,COp> >& cdagg_c_rem,
        const std::vector<std::pair<CdaggerOp,COp> >& cdagg_c_add
      );

      /**
       * Rebuild the matrix from scratch
       */
      void rebuild();

    private:
      //inverse matrix
      ResizableMatrix<Scalar> inv_matrix_;

      //permutation of time-ordering of rows and cols
      int permutation_row_col_;//1 or -1

      //a vector of creation and annihilation operators in the order in which they appear in the rows and cols of the matrix
      cdagg_container_t cdagg_ops_;
      c_container_t c_ops_;

      const GreensFunction gf_;

      //key: the imaginary time of an operator, the index of row or col in the matrix
      std::map<itime_t, int> row_pos_, col_pos_;

      //work space and helper
      int perm_rat_;
      std::vector<int> rem_cols_, rem_rows_;
      eigen_matrix_t G_n_n_, G_n_j_, G_j_n_;
      ReplaceHelper<Scalar,eigen_matrix_t,eigen_matrix_t,eigen_matrix_t> replace_helper_;

      //swap cols of the matrix (and the rows of the inverse matrix)
      void swap_cols(int col1, int col2);

      //swap rows of the matrix (and the cols of the inverse matrix)
      void swap_rows(int row1, int row2);

      int add_new_operators(
        typename std::vector<std::pair<CdaggerOp, COp> >::const_iterator begin,
        typename std::vector<std::pair<CdaggerOp, COp> >::const_iterator end);

      void remove_new_operators(const std::vector<std::pair<CdaggerOp,COp> >& cdagg_c_add);

      inline Scalar compute_g(int row, int col) const {return gf_(c_ops_[row], cdagg_ops_[col]); }

      inline int find_cdagg(const CdaggerOp& cdagg) const {
        assert(col_pos_.find(operator_time(cdagg))!=col_pos_.end());
        return col_pos_.at(operator_time(cdagg));
      }

      inline int find_c(const COp& c) const {
        assert(row_pos_.find(operator_time(c))!=row_pos_.end());
        return row_pos_.at(operator_time(c));
      }

      void sanity_check() const;

    };
  }
}

#include "determinant_matrix.ipp"

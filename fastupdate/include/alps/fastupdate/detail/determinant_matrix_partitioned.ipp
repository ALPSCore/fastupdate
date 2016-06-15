#include "../determinant_matrix_partitioned.hpp"

namespace alps {
  namespace fastupdate {

    template<
      typename Scalar,
      typename GreensFunction,
      typename CdaggerOp,
      typename COp
    >
    DeterminantMatrixPartitioned<Scalar, GreensFunction, CdaggerOp, COp>::DeterminantMatrixPartitioned(const GreensFunction &gf)
      : Base(gf),
        state_(waiting),
        num_flavors_(gf.num_flavors()),
        num_sectors_(-1),
        permutation_(1) {

      init(gf);
    }

    template<
      typename Scalar,
      typename GreensFunction,
      typename CdaggerOp,
      typename COp
    >
    template<typename CdaggCIterator>
    DeterminantMatrixPartitioned<Scalar, GreensFunction, CdaggerOp, COp>::DeterminantMatrixPartitioned(
      const GreensFunction &gf,
      CdaggCIterator first,
      CdaggCIterator last
    )
      :
        Base(gf),
        state_(waiting),
        num_flavors_(gf.num_flavors()),
        num_sectors_(-1),
        permutation_(1) {

      init(gf);

      std::vector<CdaggerOp> cdagg_ops;
      std::vector<COp> c_ops;
      for (CdaggCIterator it = first; it != last; ++it) {
        cdagg_ops.push_back(it->first);
        c_ops.    push_back(it->second);
      }
      const Scalar det_rat = try_update(
        (CdaggerOp*)NULL,  (CdaggerOp*)NULL,
        (COp*)NULL,        (COp*)NULL,
        cdagg_ops.begin(), cdagg_ops.end(),
        c_ops.begin(),     c_ops.end()
      );
      if (det_rat == 0.0) {
        throw std::runtime_error("Initial state is singular!");
      }
      perform_update();
    }

    //Partitioning of flavors
    template<
      typename Scalar,
      typename GreensFunction,
      typename CdaggerOp,
      typename COp
    >
    void DeterminantMatrixPartitioned<Scalar, GreensFunction, CdaggerOp, COp>::init(const GreensFunction &gf) {
      Clustering cl(num_flavors_);
      for (int flavor=0; flavor<num_flavors_; ++flavor) {
        for (int flavor2=0; flavor2<flavor; ++flavor2) {
          assert(gf.is_connected(flavor, flavor2)==gf.is_connected(flavor2, flavor));
          if (gf.is_connected(flavor, flavor2)) {
            cl.connect_vertices(flavor, flavor2);
          }
        }
      }
      cl.finalize_labeling();
      num_sectors_ = cl.get_num_clusters();
      sector_members_ = cl.get_cluster_members();
      sector_belonging_to_.resize(num_flavors_);
      for (int flavor=0; flavor<num_flavors_; ++flavor) {
        sector_belonging_to_[flavor] = cl.get_cluster_label(flavor);
      }

      //Sanity check
      for (int flavor=0; flavor<num_flavors_; ++flavor) {
        for (int flavor2 = 0; flavor2 < flavor; ++flavor2) {
          if (sector_belonging_to_[flavor] != sector_belonging_to_[flavor2]) {
            assert(!gf.is_connected(flavor, flavor2));
          }
        }
      }

      //Prepare DeterminantMatrix for each sector
      for (int sector=0; sector<num_sectors_; ++sector) {
        p_det_mat_.push_back(
          boost::shared_ptr<BlockMatrixType>(
            new BlockMatrixType(gf)
          )
        );
      }

      cdagg_ops_add_.resize(num_flavors_);
      cdagg_ops_rem_.resize(num_flavors_);
      c_ops_add_.resize(num_flavors_);
      c_ops_rem_.resize(num_flavors_);
    }

    template<typename Op>
    inline
    void
    remove_from_std_vector(std::vector<Op>& ops, const Op& val) {
      typename std::vector<Op>::iterator it_found = std::find(ops.begin(), ops.end(), val);
      if (it_found != ops.end()) {
        if (it_found != ops.end()-1) {
          std::swap(*it_found, *(ops.end()-1));
        }
        ops.pop_back();
      }
    }

    template<
      typename Scalar,
      typename GreensFunction,
      typename CdaggerOp,
      typename COp
    >
    template<typename CdaggIterator, typename CIterator, typename CdaggIterator2, typename CIterator2>
    Scalar
    DeterminantMatrixPartitioned<Scalar,GreensFunction,CdaggerOp,COp>::try_update(
      CdaggIterator  cdagg_rem_first,  CdaggIterator  cdagg_rem_last,
      CIterator      c_rem_first,      CIterator      c_rem_last,
      CdaggIterator2 cdagg_add_first,  CdaggIterator2 cdagg_add_last,
      CIterator2     c_add_first,      CIterator2     c_add_last
    ) {
      cdagg_ops_work_ = cdagg_ops_ordered_in_sectors_;
      c_ops_work_ = c_ops_ordered_in_sectors_;

      //Creation operators to be removed
      for (CdaggIterator it=cdagg_rem_first; it!=cdagg_rem_last; ++it) {
        const int sector = sector_belonging_to_[operator_flavor(*it)];
        cdagg_ops_rem_[sector].push_back(*it);
        remove_from_std_vector(cdagg_ops_work_, std::make_pair(sector,*it));
      }

      //Annihilation operators to be removed
      for (CIterator it=c_rem_first; it!=c_rem_last; ++it) {
        const int sector = sector_belonging_to_[operator_flavor(*it)];
        c_ops_rem_[sector].push_back(*it);
        remove_from_std_vector(c_ops_work_, std::make_pair(sector,*it) );
      }

      //Creation operators to be added
      for (CdaggIterator2 it=cdagg_add_first; it!=cdagg_add_last; ++it) {
        const int sector = sector_belonging_to_[operator_flavor(*it)];
        cdagg_ops_add_[sector].push_back(*it);
        cdagg_ops_work_.push_back(std::make_pair(sector, *it));
      }

      //Annihilation operators to be added
      for (CIterator2 it=c_add_first; it!=c_add_last; ++it) {
        const int sector = sector_belonging_to_[operator_flavor(*it)];
        c_ops_add_[sector].push_back(*it);
        c_ops_work_.push_back(std::make_pair(sector, *it));
      }

      //Count permutation
      detail::comb_sort(cdagg_ops_work_.begin(), cdagg_ops_work_.end(), CompareOverSectors<CdaggerOp>());
      detail::comb_sort(c_ops_work_.begin(),     c_ops_work_.end(),     CompareOverSectors<COp>());
      new_perm_ =
        detail::comb_sort(cdagg_ops_work_.begin(), cdagg_ops_work_.end(), CompareWithinSectors<CdaggerOp>())*
        detail::comb_sort(c_ops_work_.begin(),     c_ops_work_.end(),     CompareWithinSectors<COp>());

      //Second, compute determinant ratio from each sector
      Scalar det_rat = 1.0;
      for (int sector=0; sector<num_sectors_; ++sector) {
        det_rat *= p_det_mat_[sector]->try_update(
          cdagg_ops_rem_[sector].begin(), cdagg_ops_rem_[sector].end(),
          c_ops_rem_[sector].    begin(),     c_ops_rem_[sector].end(),
          cdagg_ops_add_[sector].begin(), cdagg_ops_add_[sector].end(),
          c_ops_add_[sector].    begin(),     c_ops_add_[sector].end()
        );
      }

      return det_rat*(1.*new_perm_/permutation_);
    }

    template<
            typename Scalar,
            typename GreensFunction,
            typename CdaggerOp,
            typename COp
    >
    void
    DeterminantMatrixPartitioned<Scalar,GreensFunction,CdaggerOp,COp>::perform_update() {
      for (int sector=0; sector<num_sectors_; ++sector) {
        p_det_mat_[sector]->perform_update();
      }
      std::swap(cdagg_ops_work_, cdagg_ops_ordered_in_sectors_);
      std::swap(c_ops_work_,     c_ops_ordered_in_sectors_);

      cdagg_ops_.resize(size());
      for (int i=0; i<size(); ++i) {
        cdagg_ops_[i] = cdagg_ops_ordered_in_sectors_[i].second;
      }

      c_ops_.resize(size());
      for (int i=0; i<size(); ++i) {
        c_ops_[i] = c_ops_ordered_in_sectors_[i].second;
      }

      permutation_ = new_perm_;
      clear_work();

      sanity_check();
    };

    template<
              typename Scalar,
              typename GreensFunction,
              typename CdaggerOp,
              typename COp
      >
    void
    DeterminantMatrixPartitioned<Scalar,GreensFunction,CdaggerOp,COp>::reject_update() {
      for (int sector=0; sector<num_sectors_; ++sector) {
        p_det_mat_[sector]->reject_update();
      }
      clear_work();
    };

    template<
      typename Scalar,
      typename GreensFunction,
      typename CdaggerOp,
      typename COp
    >
    void
    DeterminantMatrixPartitioned<Scalar,GreensFunction,CdaggerOp,COp>::sanity_check() {
#ifndef NDEBUG
      const int N = c_ops_ordered_in_sectors_.size();
      for (int i=0; i<N-1; ++i) {
        assert(cdagg_ops_ordered_in_sectors_[i].first==c_ops_ordered_in_sectors_[i].first);
        if (cdagg_ops_ordered_in_sectors_[i].first==cdagg_ops_ordered_in_sectors_[i+1].first) {
          assert(cdagg_ops_ordered_in_sectors_[i].second < cdagg_ops_ordered_in_sectors_[i+1].second);
          assert(c_ops_ordered_in_sectors_[i].second < c_ops_ordered_in_sectors_[i+1].second);
        }
      }

      int pert_order = 0;
      for (int sector=0; sector<num_sectors_; ++sector) {
        pert_order += p_det_mat_[sector]->size();
      }
      assert(size()==pert_order);

      detail::comb_sort(cdagg_ops_ordered_in_sectors_.begin(), cdagg_ops_ordered_in_sectors_.end(), CompareOverSectors<CdaggerOp>());
      detail::comb_sort(c_ops_ordered_in_sectors_.begin(),     c_ops_ordered_in_sectors_.end(),     CompareOverSectors<COp>());
      const int perm_recomputed =
        detail::comb_sort(cdagg_ops_ordered_in_sectors_.begin(), cdagg_ops_ordered_in_sectors_.end(), CompareWithinSectors<CdaggerOp>())*
        detail::comb_sort(c_ops_ordered_in_sectors_.begin(),     c_ops_ordered_in_sectors_.end(),     CompareWithinSectors<COp>());
      assert(permutation_ == perm_recomputed);
#endif
    }
  }
}
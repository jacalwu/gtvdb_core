//! Greedy CRM (credit-risk-mitigation) allocation kernels.
//!
//! Implements the allocator specified in `crm-allocation.md` (repo root):
//!
//! * one loan may carry many collaterals **and** many guarantors; one
//!   collateral / guarantor may cover many loans (multi-to-multi graph);
//! * **specified** edges (`allocation_mode = 'specified'`) are contractually
//!   locked and are allocated **first** — they never enter optimisation;
//! * remaining **optimizable** edges are allocated by a deterministic greedy /
//!   priority allocator (sources & loans ranked by priority, ties broken by
//!   ascending id) so runs are auditable, explainable and reproducible;
//! * collateral capacity is haircut-adjusted up-front via
//!   [`adjusted_collateral_value_ra`] (RiskAuthority / Fermat
//!   `get_collateral_adjusted_rv`); the legacy
//!   [`adjusted_collateral_value`] remains for docs / simple callers;
//! * guarantee coverage respects `CRMg = min(G, E − CRMc)` because guarantee
//!   phases always run after the collateral phases have reduced each loan's
//!   remaining exposure;
//! * every allocation is recorded in an audit trail
//!   ([`CrmResult::allocations`]) for later validation / replay.
//!
//! Entities are identified by caller-supplied `u64` ids, so sparse or
//! non-contiguous database ids are fine — the kernel remaps to dense
//! work arrays internally and returns results keyed by the original ids.

use std::collections::HashMap;
use std::error::Error;
use std::fmt;

use rayon::prelude::*;

/// Whether an edge participates in the greedy optimisation pool.
///
/// `Specified` edges mirror contractually locked CRM arrangements — they are
/// allocated first (in edge-list order) and never optimised.
/// `Optimizable` edges are available to the greedy / priority allocator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AllocationMode {
    Specified,
    Optimizable,
}

/// How a CRM source repartitions capacity across several targeted loans
/// (RAY8 §2.7.3 `repartition_field`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum RepartitionMode {
    /// Walk loans in EXPO formula order; each takes `min(src, loan)`.
    #[default]
    Sequential,
    /// Split remaining capacity by residual EAD weights in one pass.
    ProRata,
}

/// Sort direction for one formula column (RAY8 `BIS_CRM_COLUMN`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SortDirection {
    Asc,
    Desc,
}

/// Named indicator used in CRM / EXPO ordering formulas.
///
/// Values are read from the corresponding fields on [`Loan`] / CRM sources;
/// missing inputs should be filled with neutral defaults by the caller.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CrmIndicator {
    /// Legacy / `CONTRACT_GUARANTEE.priority` / type-quality map score.
    Priority,
    /// Eligibility level score (higher = earlier when Desc).
    EligibilityLevel,
    Pd,
    Lgd,
    ResidualMaturity,
    /// Market value if populated, else nominal / capacity.
    MarketValue,
    /// Precomputed ordinal for contract reference (lexicographic rank).
    ContractRefOrd,
    RiskWeight,
    FirstLoss,
    /// On-balance (0) before off-balance (1) when Asc.
    BsType,
    RiskTypeOrd,
    SubRefOrd,
    /// Product / collateral type ordinal.
    ProductTypeOrd,
}

/// One column of a CRM or EXPO ordering formula.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FormulaCol {
    pub indicator: CrmIndicator,
    pub direction: SortDirection,
}

/// A loan that needs credit-risk-mitigation cover.
///
/// `exposure` is the gross exposure to mitigate (EAD / PV); `priority` ranks
/// which loans are covered first by a shared CRM source (larger = earlier)
/// when the active EXPO formula uses [`CrmIndicator::Priority`].
#[derive(Debug, Clone, Copy)]
pub struct Loan {
    pub id: u64,
    pub exposure: f64,
    pub priority: f64,
    pub risk_weight: f64,
    pub first_loss: f64,
    pub residual_maturity: f64,
    pub currency_ord: f64,
    pub bs_type: f64,
    pub contract_ref_ord: f64,
    pub risk_type_ord: f64,
    pub subref_ord: f64,
}

impl Default for Loan {
    fn default() -> Self {
        Self {
            id: 0,
            exposure: 0.0,
            priority: 0.0,
            risk_weight: 0.0,
            first_loss: 0.0,
            residual_maturity: 0.0,
            currency_ord: 0.0,
            bs_type: 0.0,
            contract_ref_ord: 0.0,
            risk_type_ord: 0.0,
            subref_ord: 0.0,
        }
    }
}

impl Loan {
    pub fn new(id: u64, exposure: f64, priority: f64) -> Self {
        Self {
            id,
            exposure,
            priority,
            ..Self::default()
        }
    }

    fn indicator(&self, ind: CrmIndicator) -> f64 {
        match ind {
            CrmIndicator::Priority => self.priority,
            CrmIndicator::RiskWeight => self.risk_weight,
            CrmIndicator::FirstLoss => self.first_loss,
            CrmIndicator::ResidualMaturity => self.residual_maturity,
            CrmIndicator::BsType => self.bs_type,
            CrmIndicator::ContractRefOrd => self.contract_ref_ord,
            CrmIndicator::RiskTypeOrd => self.risk_type_ord,
            CrmIndicator::SubRefOrd => self.subref_ord,
            CrmIndicator::MarketValue => self.exposure,
            CrmIndicator::EligibilityLevel
            | CrmIndicator::Pd
            | CrmIndicator::Lgd
            | CrmIndicator::ProductTypeOrd => 0.0,
        }
    }
}

/// Shared RAY-style ordering / audit metadata for a CRM source.
#[derive(Debug, Clone, Copy)]
pub struct CrmSourceMeta {
    /// Step1 rule rank (lower = earlier). Preset: collateral 10, guarantee 20, CDS 30.
    pub rule_rank: i32,
    pub eligibility_level: f64,
    pub pd: f64,
    pub lgd: f64,
    pub residual_maturity: f64,
    pub market_value: f64,
    pub contract_ref_ord: f64,
    pub product_type_ord: f64,
    pub repartition: RepartitionMode,
    pub crm_formula_id: u32,
    pub expo_formula_id: u32,
}

impl Default for CrmSourceMeta {
    fn default() -> Self {
        Self {
            rule_rank: 0,
            eligibility_level: 0.0,
            pd: 0.0,
            lgd: 0.0,
            residual_maturity: 0.0,
            market_value: 0.0,
            contract_ref_ord: 0.0,
            product_type_ord: 0.0,
            repartition: RepartitionMode::Sequential,
            crm_formula_id: 0,
            expo_formula_id: 0,
        }
    }
}

impl CrmSourceMeta {
    fn indicator(&self, ind: CrmIndicator, priority: f64, capacity: f64) -> f64 {
        match ind {
            CrmIndicator::Priority => priority,
            CrmIndicator::EligibilityLevel => self.eligibility_level,
            CrmIndicator::Pd => self.pd,
            CrmIndicator::Lgd => self.lgd,
            CrmIndicator::ResidualMaturity => self.residual_maturity,
            CrmIndicator::MarketValue => {
                if self.market_value != 0.0 {
                    self.market_value
                } else {
                    capacity
                }
            }
            CrmIndicator::ContractRefOrd => self.contract_ref_ord,
            CrmIndicator::ProductTypeOrd => self.product_type_ord,
            CrmIndicator::RiskWeight
            | CrmIndicator::FirstLoss
            | CrmIndicator::BsType
            | CrmIndicator::RiskTypeOrd
            | CrmIndicator::SubRefOrd => 0.0,
        }
    }
}

/// A collateral source.
///
/// `capacity` must already be haircut-adjusted (see
/// [`adjusted_collateral_value`]); `priority` ranks which sources are consumed
/// first (larger = earlier) when the CRM formula uses [`CrmIndicator::Priority`].
#[derive(Debug, Clone, Copy)]
pub struct Collateral {
    pub id: u64,
    pub capacity: f64,
    pub priority: f64,
    pub meta: CrmSourceMeta,
}

impl Default for Collateral {
    fn default() -> Self {
        Self {
            id: 0,
            capacity: 0.0,
            priority: 0.0,
            meta: CrmSourceMeta {
                rule_rank: 10,
                ..CrmSourceMeta::default()
            },
        }
    }
}

impl Collateral {
    pub fn new(id: u64, capacity: f64, priority: f64) -> Self {
        Self {
            id,
            capacity,
            priority,
            meta: CrmSourceMeta {
                rule_rank: 10,
                market_value: capacity,
                ..CrmSourceMeta::default()
            },
        }
    }
}

/// A guarantor. `capacity` is the guarantee amount, `priority` ranks it.
#[derive(Debug, Clone, Copy)]
pub struct Guarantor {
    pub id: u64,
    pub capacity: f64,
    pub priority: f64,
    pub meta: CrmSourceMeta,
}

impl Default for Guarantor {
    fn default() -> Self {
        Self {
            id: 0,
            capacity: 0.0,
            priority: 0.0,
            meta: CrmSourceMeta {
                rule_rank: 20,
                ..CrmSourceMeta::default()
            },
        }
    }
}

impl Guarantor {
    pub fn new(id: u64, capacity: f64, priority: f64) -> Self {
        Self {
            id,
            capacity,
            priority,
            meta: CrmSourceMeta {
                rule_rank: 20,
                market_value: capacity,
                ..CrmSourceMeta::default()
            },
        }
    }
}

/// Pledge edge from one collateral to one loan.
///
/// `ratio` is the contractual share of the collateral earmarked for the loan.
/// It is retained for proration weighting and for the future LP solver; the
/// greedy allocator itself treats every edge as giving access to the source's
/// remaining capacity (exactly the semantics of `crm-allocation.md` §5/§6).
#[derive(Debug, Clone, Copy)]
pub struct CollateralEdge {
    pub col_id: u64,
    pub loan_id: u64,
    pub ratio: f64,
    pub mode: AllocationMode,
    pub repartition: RepartitionMode,
}

/// Guarantee edge from one guarantor to one loan.
///
/// `amount` is the contractual guarantee line for that loan; it is retained for
/// the future LP solver while the greedy allocator consumes the guarantor's
/// remaining capacity (see [`crm_alloc_greedy`]).
#[derive(Debug, Clone, Copy)]
pub struct GuaranteeEdge {
    pub guarantor_id: u64,
    pub loan_id: u64,
    pub amount: f64,
    pub mode: AllocationMode,
    pub repartition: RepartitionMode,
}

/// A CDS-style protection source (Phase 4). `capacity` is the recognised
/// protection amount, `priority` ranks it.
#[derive(Debug, Clone, Copy)]
pub struct Cds {
    pub id: u64,
    pub capacity: f64,
    pub priority: f64,
    pub meta: CrmSourceMeta,
}

impl Default for Cds {
    fn default() -> Self {
        Self {
            id: 0,
            capacity: 0.0,
            priority: 0.0,
            meta: CrmSourceMeta {
                rule_rank: 30,
                ..CrmSourceMeta::default()
            },
        }
    }
}

impl Cds {
    pub fn new(id: u64, capacity: f64, priority: f64) -> Self {
        Self {
            id,
            capacity,
            priority,
            meta: CrmSourceMeta {
                rule_rank: 30,
                market_value: capacity,
                ..CrmSourceMeta::default()
            },
        }
    }
}

/// Protection edge from one CDS to one loan. `amount` is the contractual
/// protection line (LP cap); the greedy allocator consumes the CDS's remaining
/// capacity.
#[derive(Debug, Clone, Copy)]
pub struct CdsEdge {
    pub cds_id: u64,
    pub loan_id: u64,
    pub amount: f64,
    pub mode: AllocationMode,
    pub repartition: RepartitionMode,
}

/// Which kind of CRM source produced an audit record.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CrmSourceKind {
    Collateral,
    Guarantee,
    Cds,
}

/// Options for Phase-2 greedy allocation (RAY8 Normal Allocation subset).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CrmAllocOptions {
    /// Pool binding order (Step1 category ranks). Default: Collateral → Guarantee → CDS.
    pub pool_order: Vec<CrmSourceKind>,
    /// Step2: CRM source indicators (within a pool).
    pub crm_formula: Vec<FormulaCol>,
    /// Step3: deal / exposure indicators.
    pub expo_formula: Vec<FormulaCol>,
    /// Default repartition when source/edge do not override.
    pub repartition_default: RepartitionMode,
}

impl Default for CrmAllocOptions {
    fn default() -> Self {
        Self::legacy()
    }
}

impl CrmAllocOptions {
    /// Preserve historical greedy behaviour: priority desc within each pool,
    /// pools in Collateral → Guarantee → CDS order, sequential repartition.
    pub fn legacy() -> Self {
        Self {
            pool_order: vec![
                CrmSourceKind::Collateral,
                CrmSourceKind::Guarantee,
                CrmSourceKind::Cds,
            ],
            crm_formula: vec![FormulaCol {
                indicator: CrmIndicator::Priority,
                direction: SortDirection::Desc,
            }],
            expo_formula: vec![FormulaCol {
                indicator: CrmIndicator::Priority,
                direction: SortDirection::Desc,
            }],
            repartition_default: RepartitionMode::Sequential,
        }
    }
}

/// Which allocation phase produced an audit record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrmStage {
    /// Contract-locked edges (phase 1) — allocated first, in edge-list order.
    Specified,
    /// Greedy / priority allocator over optimizable edges (phase 2).
    Greedy,
    /// Exact LP optimizer over optimizable edges (phase 3, optional `crm-lp`
    /// feature) — may enforce per-edge caps / a different objective.
    Lp,
}

/// One auditable allocation step.
///
/// Every non-zero cover writes one record so results can be replayed and
/// checked: `Σ amount` per source must not exceed its capacity, `Σ amount` per
/// loan must not exceed its exposure, and `net_exposure = exposure − Σ covers`.
#[derive(Debug, Clone, Copy)]
pub struct CrmAllocation {
    pub source_kind: CrmSourceKind,
    pub stage: CrmStage,
    pub source_id: u64,
    pub loan_id: u64,
    pub amount: f64,
    /// Remaining source capacity **after** this allocation.
    pub source_remaining: f64,
    /// Remaining loan exposure **after** this allocation.
    pub loan_remaining: f64,
    /// Step1 rule rank of the source (0 when unspecified / specified phase).
    pub rule_rank: i32,
    pub crm_formula_id: u32,
    pub expo_formula_id: u32,
    pub repartition: RepartitionMode,
}

/// Per-loan allocation outcome, aligned with the input `loans` order.
#[derive(Debug, Clone)]
pub struct CrmResult {
    pub loan_id: Vec<u64>,
    /// Original exposure (== `Loan.exposure` of the input row).
    pub exposure: Vec<f64>,
    /// Total collateral cover per loan (specified + greedy).
    pub collateral_cover: Vec<f64>,
    /// Total guarantee cover per loan (specified + greedy).
    pub guarantee_cover: Vec<f64>,
    /// Total CDS protection cover per loan (Phase 4).
    pub cds_cover: Vec<f64>,
    /// Audited residual exposure after all covers (≥ 0; equals the
    /// `loan_remaining` of the last audit record for the loan).
    pub net_exposure: Vec<f64>,
    /// Full audit trail of every non-zero allocation, in allocation order.
    pub allocations: Vec<CrmAllocation>,
}

/// Input-validation errors from [`crm_alloc_greedy`].
#[derive(Debug, Clone, PartialEq)]
pub enum CrmError {
    DuplicateLoan(u64),
    DuplicateCollateral(u64),
    DuplicateGuarantor(u64),
    DuplicateCds(u64),
    DanglingCollateralEdge {
        col_id: u64,
        loan_id: u64,
    },
    DanglingGuaranteeEdge {
        guarantor_id: u64,
        loan_id: u64,
    },
    DanglingCdsEdge {
        cds_id: u64,
        loan_id: u64,
    },
    /// The optional LP solver could not solve the model (message from the
    /// solver backend).
    LpSolver(String),
}

impl fmt::Display for CrmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CrmError::DuplicateLoan(id) => write!(f, "duplicate loan_id {id}"),
            CrmError::DuplicateCollateral(id) => write!(f, "duplicate col_id {id}"),
            CrmError::DuplicateGuarantor(id) => write!(f, "duplicate guarantor_id {id}"),
            CrmError::DuplicateCds(id) => write!(f, "duplicate cds_id {id}"),
            CrmError::DanglingCollateralEdge { col_id, loan_id } => {
                write!(
                    f,
                    "collateral edge ({col_id} -> {loan_id}) references an unknown node"
                )
            }
            CrmError::DanglingGuaranteeEdge {
                guarantor_id,
                loan_id,
            } => {
                write!(
                    f,
                    "guarantee edge ({guarantor_id} -> {loan_id}) references an unknown node"
                )
            }
            CrmError::DanglingCdsEdge { cds_id, loan_id } => {
                write!(
                    f,
                    "CDS edge ({cds_id} -> {loan_id}) references an unknown node"
                )
            }
            CrmError::LpSolver(msg) => write!(f, "lp solver failed: {msg}"),
        }
    }
}

impl Error for CrmError {}

/// Haircut-adjusted collateral value (simplified / legacy).
///
/// `C_adj = max(0, C × (1 − Hc − Hfx − Hmm))`. Prefer
/// [`adjusted_collateral_value_ra`] when aligning to RiskAuthority /
/// Fermat `get_collateral_adjusted_rv` (Ray8): that form applies
/// `(1−Hmm)` **outside** and supports `hr` / `hcr` / RE-skip / pledge.
///
/// When `hr = hcr = 0`, no RE-skip, and `Hmm = 0`, both formulas agree.
pub fn adjusted_collateral_value(
    value: f64,
    haircut: f64,
    fx_haircut: f64,
    maturity_haircut: f64,
) -> f64 {
    if !(value > 0.0) {
        return 0.0;
    }
    let d = 1.0 - haircut - fx_haircut - maturity_haircut;
    if d <= 0.0 {
        0.0
    } else {
        value * d
    }
}

/// Parameters for the RiskAuthority / Fermat collateral RV adjustment
/// (`b2_haircut.cpp` `get_collateral_adjusted_rv`).
#[derive(Clone, Copy, Debug, Default)]
pub struct AdjustedCollateralParams {
    /// Gross recovery / market value (`rv`).
    pub rv: f64,
    /// Supervisory collateral haircut `hc`.
    pub hc: f64,
    /// FX mismatch haircut `hfx`.
    pub hfx: f64,
    /// Holding-period / volatility add-on `hr` (applied as `hc * (1 + hr)`).
    pub hr: f64,
    /// Extra haircut `hcr`.
    pub hcr: f64,
    /// Maturity-mismatch factor in `[0, 1]` (RA applies `(1 - hmm)` outside).
    pub maturity_mismatch: f64,
    /// When true, maturity is already folded into the covered-amount path and
    /// `maturity_mismatch` is **not** applied here (RA
    /// `ab_MaturityAdjustmentInCoveredAmount`).
    pub maturity_in_covered_amount: bool,
    /// When true, skip `hc` (RA: `nominal_vs_mv == 'N' && collateral_re`).
    pub skip_hc: bool,
    /// Pledge-portfolio path: divide by `(1 - hc)` when `0 < hc < 1`.
    pub pledge_portfolio: bool,
}

/// RiskAuthority-aligned adjusted collateral capacity.
///
/// ```text
/// hc_eff = skip_hc ? 0 : hc
/// inner  = 1 - hc_eff*(1+hr) - hfx - hcr
/// rv_ca  = maturity_in_covered_amount
///            ? rv * inner
///            : (1 - hmm) * rv * inner
/// if pledge_portfolio && 0 < hc < 1: rv_ca /= (1 - hc)
/// rv_ca  = max(0, rv_ca)
/// ```
pub fn adjusted_collateral_value_ra(p: AdjustedCollateralParams) -> f64 {
    if !(p.rv > 0.0) {
        return 0.0;
    }
    let hc_eff = if p.skip_hc { 0.0 } else { p.hc };
    let inner = 1.0 - hc_eff * (1.0 + p.hr) - p.hfx - p.hcr;
    let mut rv_ca = if p.maturity_in_covered_amount {
        p.rv * inner
    } else {
        (1.0 - p.maturity_mismatch) * p.rv * inner
    };
    if p.pledge_portfolio && p.hc > 0.0 && p.hc != 1.0 {
        let denom = 1.0 - p.hc;
        if denom != 0.0 {
            rv_ca /= denom;
        }
    }
    if rv_ca < 0.0 {
        0.0
    } else {
        rv_ca
    }
}

/// Build RA params from the common CSV columns (optional extras default off).
#[inline]
pub fn adjusted_collateral_value_ra_from_parts(
    rv: f64,
    hc: f64,
    hfx: f64,
    hmm: f64,
    hr: f64,
    hcr: f64,
    skip_hc: bool,
    maturity_in_covered_amount: bool,
    pledge_portfolio: bool,
) -> f64 {
    adjusted_collateral_value_ra(AdjustedCollateralParams {
        rv,
        hc,
        hfx,
        hr,
        hcr,
        maturity_mismatch: hmm,
        maturity_in_covered_amount,
        skip_hc,
        pledge_portfolio,
    })
}

/// Run the two-phase greedy CRM allocator (Phase 1 + Phase 2).
///
/// Phase 1 — *specified* edges, in edge-list order:
/// each edge covers `min(source.remaining, loan.remaining)`.
///
/// Phase 2 — *optimizable* edges, greedy / priority:
/// sources are consumed in (priority desc, id asc) order and each source walks
/// its optimizable edges in loan (priority desc, id asc) order, allocating
/// `min(source.remaining, loan.remaining)` per edge until the source or the
/// loan is exhausted.
///
/// Guarantee phases always follow their collateral counterparts, which realises
/// `CRMg = min(G, E − CRMc)` on the *residual* exposure after collateral.
///
/// Returns an error for duplicate ids or edges that reference unknown nodes —
/// a CRM allocator must never silently drop a contractual edge.
///
/// The optional `crm-lp` feature adds [`crate::crm_lp::crm_alloc_lp`] (Phase 3),
/// which shares this Phase-1 front end and replaces the greedy Phase 2 with an
/// exact LP optimizer.
pub fn crm_alloc_greedy(
    loans: &[Loan],
    collaterals: &[Collateral],
    guarantors: &[Guarantor],
    coll_edges: &[CollateralEdge],
    guar_edges: &[GuaranteeEdge],
) -> Result<CrmResult, CrmError> {
    crm_alloc_greedy_with_options(
        loans,
        collaterals,
        guarantors,
        coll_edges,
        guar_edges,
        &[],
        &[],
        &CrmAllocOptions::legacy(),
    )
}

/// Greedy allocator with a CDS protection pool (Phase 4).
///
/// Pools bind in the Basel interaction order `Collateral → Guarantee → CDS`, so
/// `CRMg = min(G, E − CRMc)` and `CRMcds = min(CDS, E − CRMc − CRMg)`.
///
/// When the CRM graph has multiple connected components (no shared loan /
/// source capacity across components) and `GTV_CRM_PARALLEL` is not `0`, each
/// component is allocated on a rayon worker and the audit trail is merged with
/// a stable `(stage, kind, source_id, loan_id)` order and recomputed remainings.
pub fn crm_alloc_greedy_cds(
    loans: &[Loan],
    collaterals: &[Collateral],
    guarantors: &[Guarantor],
    coll_edges: &[CollateralEdge],
    guar_edges: &[GuaranteeEdge],
    cds: &[Cds],
    cds_edges: &[CdsEdge],
) -> Result<CrmResult, CrmError> {
    crm_alloc_greedy_with_options(
        loans,
        collaterals,
        guarantors,
        coll_edges,
        guar_edges,
        cds,
        cds_edges,
        &CrmAllocOptions::legacy(),
    )
}

/// Greedy allocator with explicit RAY-style ordering / repartition options.
#[allow(clippy::too_many_arguments)]
pub fn crm_alloc_greedy_with_options(
    loans: &[Loan],
    collaterals: &[Collateral],
    guarantors: &[Guarantor],
    coll_edges: &[CollateralEdge],
    guar_edges: &[GuaranteeEdge],
    cds: &[Cds],
    cds_edges: &[CdsEdge],
    options: &CrmAllocOptions,
) -> Result<CrmResult, CrmError> {
    validate_crm_inputs(
        loans,
        collaterals,
        guarantors,
        coll_edges,
        guar_edges,
        cds,
        cds_edges,
    )?;

    if !crm_parallel_enabled() || loans.is_empty() {
        return crm_alloc_greedy_cds_serial(
            loans,
            collaterals,
            guarantors,
            coll_edges,
            guar_edges,
            cds,
            cds_edges,
            options,
        );
    }

    let components = crm_connected_components(
        loans,
        collaterals,
        guarantors,
        coll_edges,
        guar_edges,
        cds,
        cds_edges,
    );
    // Need ≥2 non-empty loan components to benefit; also skip tiny graphs
    // where rayon overhead dominates.
    let multi = components.iter().filter(|c| !c.loan_ids.is_empty()).count() >= 2;
    if !multi || loans.len() < 64 {
        return crm_alloc_greedy_cds_serial(
            loans,
            collaterals,
            guarantors,
            coll_edges,
            guar_edges,
            cds,
            cds_edges,
            options,
        );
    }

    let loan_pos: HashMap<u64, usize> = loans.iter().enumerate().map(|(i, l)| (l.id, i)).collect();
    let col_by_id: HashMap<u64, &Collateral> =
        collaterals.iter().map(|c| (c.id, c)).collect();
    let guar_by_id: HashMap<u64, &Guarantor> =
        guarantors.iter().map(|g| (g.id, g)).collect();
    let cds_by_id: HashMap<u64, &Cds> = cds.iter().map(|c| (c.id, c)).collect();

    let partials: Result<Vec<(Vec<usize>, CrmResult)>, CrmError> = components
        .par_iter()
        .filter(|c| !c.loan_ids.is_empty())
        .map(|comp| {
            let idxs: Vec<usize> = comp
                .loan_ids
                .iter()
                .map(|id| loan_pos[id])
                .collect();
            let sub_loans: Vec<Loan> = idxs.iter().map(|&i| loans[i]).collect();
            let sub_cols: Vec<Collateral> = comp
                .col_ids
                .iter()
                .filter_map(|id| col_by_id.get(id).copied().copied())
                .collect();
            let sub_guars: Vec<Guarantor> = comp
                .guar_ids
                .iter()
                .filter_map(|id| guar_by_id.get(id).copied().copied())
                .collect();
            let sub_cds: Vec<Cds> = comp
                .cds_ids
                .iter()
                .filter_map(|id| cds_by_id.get(id).copied().copied())
                .collect();
            let loan_set: std::collections::HashSet<u64> =
                comp.loan_ids.iter().copied().collect();
            let col_set: std::collections::HashSet<u64> =
                comp.col_ids.iter().copied().collect();
            let guar_set: std::collections::HashSet<u64> =
                comp.guar_ids.iter().copied().collect();
            let cds_set: std::collections::HashSet<u64> =
                comp.cds_ids.iter().copied().collect();
            let sub_ce: Vec<CollateralEdge> = coll_edges
                .iter()
                .copied()
                .filter(|e| loan_set.contains(&e.loan_id) && col_set.contains(&e.col_id))
                .collect();
            let sub_ge: Vec<GuaranteeEdge> = guar_edges
                .iter()
                .copied()
                .filter(|e| loan_set.contains(&e.loan_id) && guar_set.contains(&e.guarantor_id))
                .collect();
            let sub_cds_e: Vec<CdsEdge> = cds_edges
                .iter()
                .copied()
                .filter(|e| loan_set.contains(&e.loan_id) && cds_set.contains(&e.cds_id))
                .collect();
            let r = crm_alloc_greedy_cds_serial(
                &sub_loans,
                &sub_cols,
                &sub_guars,
                &sub_ce,
                &sub_ge,
                &sub_cds,
                &sub_cds_e,
                options,
            )?;
            Ok((idxs, r))
        })
        .collect();

    let partials = partials?;
    Ok(merge_component_results(
        loans,
        collaterals,
        guarantors,
        cds,
        partials,
    ))
}

#[allow(clippy::too_many_arguments)]
fn crm_alloc_greedy_cds_serial(
    loans: &[Loan],
    collaterals: &[Collateral],
    guarantors: &[Guarantor],
    coll_edges: &[CollateralEdge],
    guar_edges: &[GuaranteeEdge],
    cds: &[Cds],
    cds_edges: &[CdsEdge],
    options: &CrmAllocOptions,
) -> Result<CrmResult, CrmError> {
    let mut state = specified_phase(
        loans,
        collaterals,
        guarantors,
        coll_edges,
        guar_edges,
        cds,
        cds_edges,
    )?;
    greedy_phase(&mut state, options);
    Ok(state.finish())
}

/// `GTV_CRM_PARALLEL=0` disables component parallelism (default: enabled).
fn crm_parallel_enabled() -> bool {
    match std::env::var("GTV_CRM_PARALLEL") {
        Ok(v) => {
            let v = v.trim();
            !(v == "0" || v.eq_ignore_ascii_case("false") || v.eq_ignore_ascii_case("off"))
        }
        Err(_) => true,
    }
}

#[derive(Default, Clone)]
struct CrmComponent {
    loan_ids: Vec<u64>,
    col_ids: Vec<u64>,
    guar_ids: Vec<u64>,
    cds_ids: Vec<u64>,
}

/// Tagged node id for bipartite union-find (loans ↔ CRM sources).
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum CrmNode {
    Loan(u64),
    Coll(u64),
    Guar(u64),
    Cds(u64),
}

fn crm_connected_components(
    loans: &[Loan],
    collaterals: &[Collateral],
    guarantors: &[Guarantor],
    coll_edges: &[CollateralEdge],
    guar_edges: &[GuaranteeEdge],
    cds: &[Cds],
    cds_edges: &[CdsEdge],
) -> Vec<CrmComponent> {
    struct Uf {
        parent: HashMap<CrmNode, CrmNode>,
    }
    impl Uf {
        fn ensure(&mut self, x: CrmNode) {
            self.parent.entry(x).or_insert(x);
        }
        fn find(&mut self, x: CrmNode) -> CrmNode {
            self.ensure(x);
            let mut root = x;
            loop {
                let p = self.parent[&root];
                if p == root {
                    break;
                }
                root = p;
            }
            let mut cur = x;
            while cur != root {
                let next = self.parent[&cur];
                self.parent.insert(cur, root);
                cur = next;
            }
            root
        }
        fn union(&mut self, a: CrmNode, b: CrmNode) {
            let ra = self.find(a);
            let rb = self.find(b);
            if ra != rb {
                self.parent.insert(ra, rb);
            }
        }
    }

    let mut uf = Uf {
        parent: HashMap::new(),
    };
    for l in loans {
        uf.ensure(CrmNode::Loan(l.id));
    }
    for c in collaterals {
        uf.ensure(CrmNode::Coll(c.id));
    }
    for g in guarantors {
        uf.ensure(CrmNode::Guar(g.id));
    }
    for c in cds {
        uf.ensure(CrmNode::Cds(c.id));
    }
    for e in coll_edges {
        uf.union(CrmNode::Coll(e.col_id), CrmNode::Loan(e.loan_id));
    }
    for e in guar_edges {
        uf.union(CrmNode::Guar(e.guarantor_id), CrmNode::Loan(e.loan_id));
    }
    for e in cds_edges {
        uf.union(CrmNode::Cds(e.cds_id), CrmNode::Loan(e.loan_id));
    }

    let mut groups: HashMap<CrmNode, CrmComponent> = HashMap::new();
    let nodes: Vec<CrmNode> = uf.parent.keys().copied().collect();
    for n in nodes {
        let r = uf.find(n);
        let g = groups.entry(r).or_default();
        match n {
            CrmNode::Loan(id) => g.loan_ids.push(id),
            CrmNode::Coll(id) => g.col_ids.push(id),
            CrmNode::Guar(id) => g.guar_ids.push(id),
            CrmNode::Cds(id) => g.cds_ids.push(id),
        }
    }
    let mut out: Vec<CrmComponent> = groups.into_values().collect();
    for c in &mut out {
        c.loan_ids.sort_unstable();
        c.col_ids.sort_unstable();
        c.guar_ids.sort_unstable();
        c.cds_ids.sort_unstable();
    }
    out.sort_by(|a, b| {
        let ka = a.loan_ids.first().copied().unwrap_or(u64::MAX);
        let kb = b.loan_ids.first().copied().unwrap_or(u64::MAX);
        ka.cmp(&kb)
    });
    out
}

fn validate_crm_inputs(
    loans: &[Loan],
    collaterals: &[Collateral],
    guarantors: &[Guarantor],
    coll_edges: &[CollateralEdge],
    guar_edges: &[GuaranteeEdge],
    cds: &[Cds],
    cds_edges: &[CdsEdge],
) -> Result<(), CrmError> {
    let mut loan_pos: HashMap<u64, usize> = HashMap::with_capacity(loans.len());
    for (i, l) in loans.iter().enumerate() {
        if loan_pos.insert(l.id, i).is_some() {
            return Err(CrmError::DuplicateLoan(l.id));
        }
    }
    let mut col_pos: HashMap<u64, usize> = HashMap::with_capacity(collaterals.len());
    for (i, c) in collaterals.iter().enumerate() {
        if col_pos.insert(c.id, i).is_some() {
            return Err(CrmError::DuplicateCollateral(c.id));
        }
    }
    let mut guar_pos: HashMap<u64, usize> = HashMap::with_capacity(guarantors.len());
    for (i, g) in guarantors.iter().enumerate() {
        if guar_pos.insert(g.id, i).is_some() {
            return Err(CrmError::DuplicateGuarantor(g.id));
        }
    }
    let mut cds_pos: HashMap<u64, usize> = HashMap::with_capacity(cds.len());
    for (i, c) in cds.iter().enumerate() {
        if cds_pos.insert(c.id, i).is_some() {
            return Err(CrmError::DuplicateCds(c.id));
        }
    }
    for e in coll_edges {
        if !col_pos.contains_key(&e.col_id) || !loan_pos.contains_key(&e.loan_id) {
            return Err(CrmError::DanglingCollateralEdge {
                col_id: e.col_id,
                loan_id: e.loan_id,
            });
        }
    }
    for e in guar_edges {
        if !guar_pos.contains_key(&e.guarantor_id) || !loan_pos.contains_key(&e.loan_id) {
            return Err(CrmError::DanglingGuaranteeEdge {
                guarantor_id: e.guarantor_id,
                loan_id: e.loan_id,
            });
        }
    }
    for e in cds_edges {
        if !cds_pos.contains_key(&e.cds_id) || !loan_pos.contains_key(&e.loan_id) {
            return Err(CrmError::DanglingCdsEdge {
                cds_id: e.cds_id,
                loan_id: e.loan_id,
            });
        }
    }
    Ok(())
}

fn merge_component_results(
    loans: &[Loan],
    collaterals: &[Collateral],
    guarantors: &[Guarantor],
    cds: &[Cds],
    partials: Vec<(Vec<usize>, CrmResult)>,
) -> CrmResult {
    let n = loans.len();
    let mut collateral_cover = vec![0.0f64; n];
    let mut guarantee_cover = vec![0.0f64; n];
    let mut cds_cover = vec![0.0f64; n];
    let mut net_exposure: Vec<f64> = loans.iter().map(|l| l.exposure).collect();
    let mut allocations = Vec::new();

    for (idxs, r) in partials {
        for (local, &global) in idxs.iter().enumerate() {
            collateral_cover[global] = r.collateral_cover[local];
            guarantee_cover[global] = r.guarantee_cover[local];
            cds_cover[global] = r.cds_cover[local];
            net_exposure[global] = r.net_exposure[local];
        }
        allocations.extend(r.allocations);
    }

    // Stable audit order across components, then refresh remainings so `seq`
    // from the engine is reproducible regardless of rayon completion order.
    allocations.sort_by(|a, b| {
        stage_ord(a.stage)
            .cmp(&stage_ord(b.stage))
            .then(kind_ord(a.source_kind).cmp(&kind_ord(b.source_kind)))
            .then(a.source_id.cmp(&b.source_id))
            .then(a.loan_id.cmp(&b.loan_id))
    });
    recompute_audit_remainings(
        &mut allocations,
        loans,
        collaterals,
        guarantors,
        cds,
    );

    CrmResult {
        loan_id: loans.iter().map(|l| l.id).collect(),
        exposure: loans.iter().map(|l| l.exposure).collect(),
        collateral_cover,
        guarantee_cover,
        cds_cover,
        net_exposure,
        allocations,
    }
}

fn stage_ord(s: CrmStage) -> u8 {
    match s {
        CrmStage::Specified => 0,
        CrmStage::Greedy => 1,
        CrmStage::Lp => 2,
    }
}

fn kind_ord(k: CrmSourceKind) -> u8 {
    match k {
        CrmSourceKind::Collateral => 0,
        CrmSourceKind::Guarantee => 1,
        CrmSourceKind::Cds => 2,
    }
}

fn recompute_audit_remainings(
    allocs: &mut [CrmAllocation],
    loans: &[Loan],
    collaterals: &[Collateral],
    guarantors: &[Guarantor],
    cds: &[Cds],
) {
    let mut loan_rem: HashMap<u64, f64> = loans.iter().map(|l| (l.id, l.exposure)).collect();
    let mut col_rem: HashMap<u64, f64> = collaterals.iter().map(|c| (c.id, c.capacity)).collect();
    let mut guar_rem: HashMap<u64, f64> = guarantors.iter().map(|g| (g.id, g.capacity)).collect();
    let mut cds_rem: HashMap<u64, f64> = cds.iter().map(|c| (c.id, c.capacity)).collect();
    for a in allocs.iter_mut() {
        let src_map = match a.source_kind {
            CrmSourceKind::Collateral => &mut col_rem,
            CrmSourceKind::Guarantee => &mut guar_rem,
            CrmSourceKind::Cds => &mut cds_rem,
        };
        let sr = src_map.entry(a.source_id).or_insert(0.0);
        *sr = (*sr - a.amount).max(0.0);
        a.source_remaining = *sr;
        let lr = loan_rem.entry(a.loan_id).or_insert(0.0);
        *lr = (*lr - a.amount).max(0.0);
        a.loan_remaining = *lr;
    }
}

// ---------------------------------------------------------------------------
// Shared Phase-1 front end + Phase-2 greedy engine.
//
// Both `crm_alloc_greedy` (phase 2) and the optional LP solver (phase 3,
// `crm_lp`) consume [`SpecifiedState`]: validation, densification and the
// contract-locked specified allocation are byte-identical regardless of which
// allocator runs afterwards.
// ---------------------------------------------------------------------------

/// A dense optimizable-pool edge (positions into the node arrays of a
/// [`SpecifiedState`]).
///
/// `cap` is the per-edge contractual upper bound used by the LP solver:
/// * collateral edges — `ratio × haircut-adjusted capacity`,
/// * guarantee edges — the guarantee-line `amount`,
/// * `f64::INFINITY` when the edge carries no cap (no ratio / no amount).
///
/// The greedy phase 2 ignores caps (it allocates `min(source, loan)` per edge,
/// matching `crm-allocation.md` §5/§6).
#[derive(Debug, Clone, Copy)]
pub(crate) struct OptEdge {
    pub src: usize,
    pub loan: usize,
    /// Per-edge cap consumed by the LP solver (`crm-lp` feature); the greedy
    /// engine never reads it, so it is dead code in feature-off builds.
    #[allow(dead_code)]
    pub cap: f64,
    pub repartition: RepartitionMode,
}

/// Shared trait over collateral and guarantor sources (id / priority used for
/// the deterministic greedy ordering and for the LP objective weights).
pub(crate) trait SourceNode {
    fn source_id(&self) -> u64;
    fn source_priority(&self) -> f64;
    fn source_meta(&self) -> &CrmSourceMeta;
    fn source_capacity(&self) -> f64;
}

impl SourceNode for Collateral {
    fn source_id(&self) -> u64 {
        self.id
    }
    fn source_priority(&self) -> f64 {
        self.priority
    }
    fn source_meta(&self) -> &CrmSourceMeta {
        &self.meta
    }
    fn source_capacity(&self) -> f64 {
        self.capacity
    }
}

impl SourceNode for Guarantor {
    fn source_id(&self) -> u64 {
        self.id
    }
    fn source_priority(&self) -> f64 {
        self.priority
    }
    fn source_meta(&self) -> &CrmSourceMeta {
        &self.meta
    }
    fn source_capacity(&self) -> f64 {
        self.capacity
    }
}

impl SourceNode for Cds {
    fn source_id(&self) -> u64 {
        self.id
    }
    fn source_priority(&self) -> f64 {
        self.priority
    }
    fn source_meta(&self) -> &CrmSourceMeta {
        &self.meta
    }
    fn source_capacity(&self) -> f64 {
        self.capacity
    }
}

fn cmp_f64_dir(a: f64, b: f64, dir: SortDirection) -> std::cmp::Ordering {
    let ord = match dir {
        SortDirection::Desc => b.partial_cmp(&a),
        SortDirection::Asc => a.partial_cmp(&b),
    };
    ord.unwrap_or(std::cmp::Ordering::Equal)
}

fn cmp_formula_keys(
    a_vals: impl Fn(CrmIndicator) -> f64,
    b_vals: impl Fn(CrmIndicator) -> f64,
    formula: &[FormulaCol],
) -> std::cmp::Ordering {
    for col in formula {
        let c = cmp_f64_dir(a_vals(col.indicator), b_vals(col.indicator), col.direction);
        if c != std::cmp::Ordering::Equal {
            return c;
        }
    }
    std::cmp::Ordering::Equal
}

/// Deterministic source ordering: rule_rank asc, then CRM formula, then id asc.
pub(crate) fn source_order<S: SourceNode>(sources: &[S], formula: &[FormulaCol]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..sources.len()).collect();
    order.sort_by(|&a, &b| {
        sources[a]
            .source_meta()
            .rule_rank
            .cmp(&sources[b].source_meta().rule_rank)
            .then_with(|| {
                cmp_formula_keys(
                    |ind| {
                        sources[a]
                            .source_meta()
                            .indicator(ind, sources[a].source_priority(), sources[a].source_capacity())
                    },
                    |ind| {
                        sources[b]
                            .source_meta()
                            .indicator(ind, sources[b].source_priority(), sources[b].source_capacity())
                    },
                    formula,
                )
            })
            .then_with(|| sources[a].source_id().cmp(&sources[b].source_id()))
    });
    order
}

/// Position of each loan in the EXPO-formula ordering (then id asc).
pub(crate) fn loan_order_rank(loans: &[Loan], formula: &[FormulaCol]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..loans.len()).collect();
    order.sort_by(|&a, &b| {
        cmp_formula_keys(
            |ind| loans[a].indicator(ind),
            |ind| loans[b].indicator(ind),
            formula,
        )
        .then_with(|| loans[a].id.cmp(&loans[b].id))
    });
    let mut rank = vec![0usize; loans.len()];
    for (i, &l) in order.iter().enumerate() {
        rank[l] = i;
    }
    rank
}

/// State after validation + densification + Phase 1 (specified edges).
///
/// `*_remaining` are residual capacities / exposures *after* the specified
/// phase; the cover and audit fields hold the Phase-1 results. Optimizable
/// edges are dense (`src` / `loan` are positions) so the greedy engine and the
/// LP solver both operate without re-hashing ids.
#[derive(Debug, Clone)]
pub(crate) struct SpecifiedState {
    pub loans: Vec<Loan>,
    pub loan_remaining: Vec<f64>,
    pub collaterals: Vec<Collateral>,
    pub col_remaining: Vec<f64>,
    pub guarantors: Vec<Guarantor>,
    pub guar_remaining: Vec<f64>,
    pub cds: Vec<Cds>,
    pub cds_remaining: Vec<f64>,
    pub collateral_cover: Vec<f64>,
    pub guarantee_cover: Vec<f64>,
    pub cds_cover: Vec<f64>,
    pub audit: Vec<CrmAllocation>,
    pub opt_coll_edges: Vec<OptEdge>,
    pub opt_guar_edges: Vec<OptEdge>,
    pub opt_cds_edges: Vec<OptEdge>,
}

/// Validate + densify inputs, then run Phase 1 (specified edges, edge-list
/// order). The returned state is consumed by the greedy Phase 2 or the LP
/// Phase 3.
pub(crate) fn specified_phase(
    loans: &[Loan],
    collaterals: &[Collateral],
    guarantors: &[Guarantor],
    coll_edges: &[CollateralEdge],
    guar_edges: &[GuaranteeEdge],
    cds: &[Cds],
    cds_edges: &[CdsEdge],
) -> Result<SpecifiedState, CrmError> {
    // dense position maps + duplicate / dangling validation
    let mut loan_pos: HashMap<u64, usize> = HashMap::with_capacity(loans.len());
    for (i, l) in loans.iter().enumerate() {
        if loan_pos.insert(l.id, i).is_some() {
            return Err(CrmError::DuplicateLoan(l.id));
        }
    }
    let mut col_pos: HashMap<u64, usize> = HashMap::with_capacity(collaterals.len());
    for (i, c) in collaterals.iter().enumerate() {
        if col_pos.insert(c.id, i).is_some() {
            return Err(CrmError::DuplicateCollateral(c.id));
        }
    }
    let mut guar_pos: HashMap<u64, usize> = HashMap::with_capacity(guarantors.len());
    for (i, g) in guarantors.iter().enumerate() {
        if guar_pos.insert(g.id, i).is_some() {
            return Err(CrmError::DuplicateGuarantor(g.id));
        }
    }
    let mut cds_pos: HashMap<u64, usize> = HashMap::with_capacity(cds.len());
    for (i, c) in cds.iter().enumerate() {
        if cds_pos.insert(c.id, i).is_some() {
            return Err(CrmError::DuplicateCds(c.id));
        }
    }
    for e in coll_edges {
        if !col_pos.contains_key(&e.col_id) || !loan_pos.contains_key(&e.loan_id) {
            return Err(CrmError::DanglingCollateralEdge {
                col_id: e.col_id,
                loan_id: e.loan_id,
            });
        }
    }
    for e in guar_edges {
        if !guar_pos.contains_key(&e.guarantor_id) || !loan_pos.contains_key(&e.loan_id) {
            return Err(CrmError::DanglingGuaranteeEdge {
                guarantor_id: e.guarantor_id,
                loan_id: e.loan_id,
            });
        }
    }
    for e in cds_edges {
        if !cds_pos.contains_key(&e.cds_id) || !loan_pos.contains_key(&e.loan_id) {
            return Err(CrmError::DanglingCdsEdge {
                cds_id: e.cds_id,
                loan_id: e.loan_id,
            });
        }
    }

    let mut state = SpecifiedState {
        loans: loans.to_vec(),
        loan_remaining: loans.iter().map(|l| l.exposure).collect(),
        collaterals: collaterals.to_vec(),
        col_remaining: collaterals.iter().map(|c| c.capacity).collect(),
        guarantors: guarantors.to_vec(),
        guar_remaining: guarantors.iter().map(|g| g.capacity).collect(),
        cds: cds.to_vec(),
        cds_remaining: cds.iter().map(|c| c.capacity).collect(),
        collateral_cover: vec![0.0f64; loans.len()],
        guarantee_cover: vec![0.0f64; loans.len()],
        cds_cover: vec![0.0f64; loans.len()],
        audit: Vec::new(),
        opt_coll_edges: Vec::new(),
        opt_guar_edges: Vec::new(),
        opt_cds_edges: Vec::new(),
    };

    // Phase 1a: specified collateral, in edge-list order.
    for e in coll_edges {
        if e.mode != AllocationMode::Specified {
            continue;
        }
        let (c, l) = (col_pos[&e.col_id], loan_pos[&e.loan_id]);
        if state.loan_remaining[l] <= 0.0 || state.col_remaining[c] <= 0.0 {
            continue;
        }
        let cover = state.col_remaining[c].min(state.loan_remaining[l]);
        state.col_remaining[c] -= cover;
        state.loan_remaining[l] -= cover;
        state.collateral_cover[l] += cover;
        state.audit.push(CrmAllocation {
            source_kind: CrmSourceKind::Collateral,
            stage: CrmStage::Specified,
            source_id: e.col_id,
            loan_id: e.loan_id,
            amount: cover,
            source_remaining: state.col_remaining[c],
            loan_remaining: state.loan_remaining[l],
            rule_rank: state.collaterals[c].meta.rule_rank,
            crm_formula_id: state.collaterals[c].meta.crm_formula_id,
            expo_formula_id: state.collaterals[c].meta.expo_formula_id,
            repartition: e.repartition,
        });
    }

    // Phase 1b: specified guarantee, in edge-list order.
    for e in guar_edges {
        if e.mode != AllocationMode::Specified {
            continue;
        }
        let (g, l) = (guar_pos[&e.guarantor_id], loan_pos[&e.loan_id]);
        if state.loan_remaining[l] <= 0.0 || state.guar_remaining[g] <= 0.0 {
            continue;
        }
        let cover = state.guar_remaining[g].min(state.loan_remaining[l]);
        state.guar_remaining[g] -= cover;
        state.loan_remaining[l] -= cover;
        state.guarantee_cover[l] += cover;
        state.audit.push(CrmAllocation {
            source_kind: CrmSourceKind::Guarantee,
            stage: CrmStage::Specified,
            source_id: e.guarantor_id,
            loan_id: e.loan_id,
            amount: cover,
            source_remaining: state.guar_remaining[g],
            loan_remaining: state.loan_remaining[l],
            rule_rank: state.guarantors[g].meta.rule_rank,
            crm_formula_id: state.guarantors[g].meta.crm_formula_id,
            expo_formula_id: state.guarantors[g].meta.expo_formula_id,
            repartition: e.repartition,
        });
    }

    // Phase 1c: specified CDS protection, in edge-list order.
    for e in cds_edges {
        if e.mode != AllocationMode::Specified {
            continue;
        }
        let (c, l) = (cds_pos[&e.cds_id], loan_pos[&e.loan_id]);
        if state.loan_remaining[l] <= 0.0 || state.cds_remaining[c] <= 0.0 {
            continue;
        }
        let cover = state.cds_remaining[c].min(state.loan_remaining[l]);
        state.cds_remaining[c] -= cover;
        state.loan_remaining[l] -= cover;
        state.cds_cover[l] += cover;
        state.audit.push(CrmAllocation {
            source_kind: CrmSourceKind::Cds,
            stage: CrmStage::Specified,
            source_id: e.cds_id,
            loan_id: e.loan_id,
            amount: cover,
            source_remaining: state.cds_remaining[c],
            loan_remaining: state.loan_remaining[l],
            rule_rank: state.cds[c].meta.rule_rank,
            crm_formula_id: state.cds[c].meta.crm_formula_id,
            expo_formula_id: state.cds[c].meta.expo_formula_id,
            repartition: e.repartition,
        });
    }

    // Dense optimizable-pool edges (+ per-edge caps for the LP solver).
    for e in coll_edges {
        if e.mode != AllocationMode::Optimizable {
            continue;
        }
        let (c, l) = (col_pos[&e.col_id], loan_pos[&e.loan_id]);
        let cap = if e.ratio > 0.0 {
            e.ratio * collaterals[c].capacity
        } else {
            f64::INFINITY
        };
        state.opt_coll_edges.push(OptEdge {
            src: c,
            loan: l,
            cap,
            repartition: e.repartition,
        });
    }
    for e in guar_edges {
        if e.mode != AllocationMode::Optimizable {
            continue;
        }
        let (g, l) = (guar_pos[&e.guarantor_id], loan_pos[&e.loan_id]);
        let cap = if e.amount > 0.0 {
            e.amount
        } else {
            f64::INFINITY
        };
        state.opt_guar_edges.push(OptEdge {
            src: g,
            loan: l,
            cap,
            repartition: e.repartition,
        });
    }
    for e in cds_edges {
        if e.mode != AllocationMode::Optimizable {
            continue;
        }
        let (c, l) = (cds_pos[&e.cds_id], loan_pos[&e.loan_id]);
        let cap = if e.amount > 0.0 {
            e.amount
        } else {
            f64::INFINITY
        };
        state.opt_cds_edges.push(OptEdge {
            src: c,
            loan: l,
            cap,
            repartition: e.repartition,
        });
    }

    Ok(state)
}

/// Phase 2: greedy / priority allocation over the optimizable pool, in place.
///
/// Pool order follows [`CrmAllocOptions::pool_order`] (Step1). Within a pool,
/// sources follow rule_rank + CRM formula (Step2) and each source walks loans
/// by EXPO formula (Step3), using Sequential or ProRata repartition.
fn greedy_phase(state: &mut SpecifiedState, options: &CrmAllocOptions) {
    let loan_rank = loan_order_rank(&state.loans, &options.expo_formula);
    for kind in &options.pool_order {
        match kind {
            CrmSourceKind::Collateral => apply_greedy_pool(
                &state.loans,
                &mut state.loan_remaining,
                &mut state.collateral_cover,
                &state.collaterals,
                &mut state.col_remaining,
                &state.opt_coll_edges,
                &loan_rank,
                CrmSourceKind::Collateral,
                options,
                &mut state.audit,
            ),
            CrmSourceKind::Guarantee => apply_greedy_pool(
                &state.loans,
                &mut state.loan_remaining,
                &mut state.guarantee_cover,
                &state.guarantors,
                &mut state.guar_remaining,
                &state.opt_guar_edges,
                &loan_rank,
                CrmSourceKind::Guarantee,
                options,
                &mut state.audit,
            ),
            CrmSourceKind::Cds => apply_greedy_pool(
                &state.loans,
                &mut state.loan_remaining,
                &mut state.cds_cover,
                &state.cds,
                &mut state.cds_remaining,
                &state.opt_cds_edges,
                &loan_rank,
                CrmSourceKind::Cds,
                options,
                &mut state.audit,
            ),
        }
    }
}

fn effective_repartition(
    source: &CrmSourceMeta,
    edge_modes: &[RepartitionMode],
    default: RepartitionMode,
) -> RepartitionMode {
    if source.repartition == RepartitionMode::ProRata {
        return RepartitionMode::ProRata;
    }
    if edge_modes
        .iter()
        .any(|m| *m == RepartitionMode::ProRata)
    {
        return RepartitionMode::ProRata;
    }
    if default == RepartitionMode::ProRata {
        return RepartitionMode::ProRata;
    }
    RepartitionMode::Sequential
}

/// Greedy allocation over one source pool (collateral or guarantee). Only the
/// optimizable edges of [`SpecifiedState`] participate, so it never re-uses
/// capacity already locked by the specified phase.
#[allow(clippy::too_many_arguments)]
fn apply_greedy_pool<S: SourceNode>(
    loans: &[Loan],
    loan_remaining: &mut [f64],
    cover: &mut [f64],
    sources: &[S],
    source_remaining: &mut [f64],
    edges: &[OptEdge],
    loan_rank: &[usize],
    kind: CrmSourceKind,
    options: &CrmAllocOptions,
    audit: &mut Vec<CrmAllocation>,
) {
    // per-source adjacency of optimizable loans, walked in loan-priority order
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); sources.len()];
    let mut edge_rep: Vec<Vec<RepartitionMode>> = vec![Vec::new(); sources.len()];
    for e in edges {
        adj[e.src].push(e.loan);
        edge_rep[e.src].push(e.repartition);
    }
    for list in adj.iter_mut() {
        list.sort_unstable_by_key(|&l| loan_rank[l]);
        list.dedup();
    }
    let src_order = source_order(sources, &options.crm_formula);
    for s in src_order {
        if source_remaining[s] <= 0.0 {
            continue;
        }
        let meta = *sources[s].source_meta();
        let mode = effective_repartition(&meta, &edge_rep[s], options.repartition_default);
        match mode {
            RepartitionMode::Sequential => {
                for &l in &adj[s] {
                    if loan_remaining[l] <= 0.0 {
                        continue;
                    }
                    let cover_amount = source_remaining[s].min(loan_remaining[l]);
                    if cover_amount <= 0.0 {
                        continue;
                    }
                    source_remaining[s] -= cover_amount;
                    loan_remaining[l] -= cover_amount;
                    cover[l] += cover_amount;
                    audit.push(CrmAllocation {
                        source_kind: kind,
                        stage: CrmStage::Greedy,
                        source_id: sources[s].source_id(),
                        loan_id: loans[l].id,
                        amount: cover_amount,
                        source_remaining: source_remaining[s],
                        loan_remaining: loan_remaining[l],
                        rule_rank: meta.rule_rank,
                        crm_formula_id: meta.crm_formula_id,
                        expo_formula_id: meta.expo_formula_id,
                        repartition: mode,
                    });
                    if source_remaining[s] <= 0.0 {
                        break;
                    }
                }
            }
            RepartitionMode::ProRata => {
                let targets: Vec<usize> = adj[s]
                    .iter()
                    .copied()
                    .filter(|&l| loan_remaining[l] > 0.0)
                    .collect();
                if targets.is_empty() {
                    continue;
                }
                let total_ead: f64 = targets.iter().map(|&l| loan_remaining[l]).sum();
                if total_ead <= 0.0 {
                    continue;
                }
                let available = source_remaining[s];
                let mut allocated = 0.0;
                for (i, &l) in targets.iter().enumerate() {
                    let raw = if i + 1 == targets.len() {
                        (available - allocated).max(0.0)
                    } else {
                        available * (loan_remaining[l] / total_ead)
                    };
                    let cover_amount = raw.min(loan_remaining[l]).min(source_remaining[s]);
                    if cover_amount <= 0.0 {
                        continue;
                    }
                    source_remaining[s] -= cover_amount;
                    loan_remaining[l] -= cover_amount;
                    cover[l] += cover_amount;
                    allocated += cover_amount;
                    audit.push(CrmAllocation {
                        source_kind: kind,
                        stage: CrmStage::Greedy,
                        source_id: sources[s].source_id(),
                        loan_id: loans[l].id,
                        amount: cover_amount,
                        source_remaining: source_remaining[s],
                        loan_remaining: loan_remaining[l],
                        rule_rank: meta.rule_rank,
                        crm_formula_id: meta.crm_formula_id,
                        expo_formula_id: meta.expo_formula_id,
                        repartition: mode,
                    });
                }
            }
        }
    }
}

impl SpecifiedState {
    /// Consume the (fully allocated) state into a per-loan result. Net exposure
    /// is the audited residual remaining per loan, kept ≥ 0.
    pub(crate) fn finish(self) -> CrmResult {
        let loan_id: Vec<u64> = self.loans.iter().map(|l| l.id).collect();
        let exposure: Vec<f64> = self.loans.iter().map(|l| l.exposure).collect();
        let net_exposure: Vec<f64> = self.loan_remaining.iter().map(|r| r.max(0.0)).collect();
        CrmResult {
            loan_id,
            exposure,
            collateral_cover: self.collateral_cover,
            guarantee_cover: self.guarantee_cover,
            cds_cover: self.cds_cover,
            net_exposure,
            allocations: self.audit,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn loan(id: u64, exposure: f64) -> Loan {
        Loan::new(id, exposure, 0.0)
    }
    fn col(id: u64, capacity: f64) -> Collateral {
        Collateral::new(id, capacity, 0.0)
    }
    fn guar(id: u64, capacity: f64) -> Guarantor {
        Guarantor::new(id, capacity, 0.0)
    }

    #[test]
    fn haircut_adjusted_value() {
        // C_adj = C x (1 - Hc - Hfx - Hmm)
        assert!((adjusted_collateral_value(1000.0, 0.1, 0.05, 0.05) - 800.0).abs() < 1e-9);
        assert_eq!(adjusted_collateral_value(1000.0, 0.6, 0.6, 0.0), 0.0); // floored
        assert_eq!(adjusted_collateral_value(-5.0, 0.0, 0.0, 0.0), 0.0);
        assert_eq!(adjusted_collateral_value(0.0, 0.0, 0.0, 0.0), 0.0);
        assert_eq!(adjusted_collateral_value(100.0, 0.0, 0.0, 0.0), 100.0);
    }

    #[test]
    fn ra_haircut_matches_fermat_branches() {
        // Baseline: hmm outside, hr=hcr=0, no skip → (1-hmm)*rv*(1-hc-hfx)
        let base = adjusted_collateral_value_ra(AdjustedCollateralParams {
            rv: 5000.0,
            hc: 0.4,
            hfx: 0.0,
            hr: 0.0,
            hcr: 0.0,
            maturity_mismatch: 0.0,
            ..AdjustedCollateralParams::default()
        });
        assert!((base - 3000.0).abs() < 1e-9);

        // RE / nominal skip zeroes hc
        let skipped = adjusted_collateral_value_ra(AdjustedCollateralParams {
            rv: 5000.0,
            hc: 0.4,
            skip_hc: true,
            ..AdjustedCollateralParams::default()
        });
        assert!((skipped - 5000.0).abs() < 1e-9);

        // hr scales hc: hc*(1+hr)
        let with_hr = adjusted_collateral_value_ra(AdjustedCollateralParams {
            rv: 1000.0,
            hc: 0.10,
            hr: 0.5, // → 0.15
            ..AdjustedCollateralParams::default()
        });
        assert!((with_hr - 850.0).abs() < 1e-9);

        // hcr subtracts directly
        let with_hcr = adjusted_collateral_value_ra(AdjustedCollateralParams {
            rv: 1000.0,
            hc: 0.10,
            hcr: 0.05,
            ..AdjustedCollateralParams::default()
        });
        assert!((with_hcr - 850.0).abs() < 1e-9);

        // maturity outside: (1-0.2)*1000*(1-0.1) = 720
        let with_mm = adjusted_collateral_value_ra(AdjustedCollateralParams {
            rv: 1000.0,
            hc: 0.10,
            maturity_mismatch: 0.2,
            ..AdjustedCollateralParams::default()
        });
        assert!((with_mm - 720.0).abs() < 1e-9);

        // maturity already in covered amount: ignore hmm
        let mm_in_cover = adjusted_collateral_value_ra(AdjustedCollateralParams {
            rv: 1000.0,
            hc: 0.10,
            maturity_mismatch: 0.2,
            maturity_in_covered_amount: true,
            ..AdjustedCollateralParams::default()
        });
        assert!((mm_in_cover - 900.0).abs() < 1e-9);

        // pledge portfolio: rv*(1-hc-hfx) / (1-hc)
        let pledge = adjusted_collateral_value_ra(AdjustedCollateralParams {
            rv: 1000.0,
            hc: 0.20,
            hfx: 0.10,
            pledge_portfolio: true,
            ..AdjustedCollateralParams::default()
        });
        // (1-0.2-0.1)=0.7 → 700 / 0.8 = 875
        assert!((pledge - 875.0).abs() < 1e-9);

        // floor at 0
        assert_eq!(
            adjusted_collateral_value_ra(AdjustedCollateralParams {
                rv: 100.0,
                hc: 0.9,
                hfx: 0.2,
                ..AdjustedCollateralParams::default()
            }),
            0.0
        );
    }

    #[test]
    fn ra_agrees_with_simple_when_hmm_and_addons_zero() {
        let v = 10000.0;
        let hc = 0.0424264068711929;
        let simple = adjusted_collateral_value(v, hc, 0.0, 0.0);
        let ra = adjusted_collateral_value_ra_from_parts(
            v, hc, 0.0, 0.0, 0.0, 0.0, false, false, false,
        );
        assert!((simple - ra).abs() < 1e-9);
        assert!((ra - 9575.73593128807).abs() < 1e-6);
    }

    #[test]
    fn specified_collateral_is_allocated_before_optimizable() {
        // L1: specified C1(60) + optimizable C2(100). L2: optimizable C2(100).
        // C1 must be consumed first by contract, only then C2 is optimised.
        let loans = [loan(1, 100.0), loan(2, 100.0)];
        let cols = [col(10, 60.0), col(20, 100.0)];
        let ce = [
            CollateralEdge {
                col_id: 10,
                loan_id: 1,
                ratio: 1.0,
                mode: AllocationMode::Specified,
            repartition: RepartitionMode::Sequential,
        },
            CollateralEdge {
                col_id: 20,
                loan_id: 1,
                ratio: 1.0,
                mode: AllocationMode::Optimizable,
            repartition: RepartitionMode::Sequential,
        },
            CollateralEdge {
                col_id: 20,
                loan_id: 2,
                ratio: 1.0,
                mode: AllocationMode::Optimizable,
            repartition: RepartitionMode::Sequential,
        },
        ];
        let r = crm_alloc_greedy(&loans, &cols, &[], &ce, &[]).unwrap();
        assert_eq!(r.loan_id, vec![1, 2]);
        assert!((r.collateral_cover[0] - 100.0).abs() < 1e-9); // 60 specified + 40 greedy
        assert!((r.collateral_cover[1] - 60.0).abs() < 1e-9);
        assert!((r.net_exposure[0]).abs() < 1e-9);
        assert!((r.net_exposure[1] - 40.0).abs() < 1e-9);
        assert_eq!(r.allocations.len(), 3);
        assert_eq!(r.allocations[0].stage, CrmStage::Specified);
        assert_eq!(r.allocations[0].source_id, 10);
        assert_eq!(r.allocations[0].amount, 60.0);
        // first greedy alloc is still the residual on L1, not L2
        assert_eq!(r.allocations[1].stage, CrmStage::Greedy);
        assert_eq!(r.allocations[1].loan_id, 1);
        assert_eq!(r.allocations[2].loan_id, 2);
    }

    #[test]
    fn guarantee_netting_formula_min_g_e_minus_crmc() {
        // L1 exp 100 with specified collateral 70 -> residual 30; guarantee
        // (capacity 100) must cover exactly min(G, 30) = 30.
        let loans = [loan(1, 100.0)];
        let cols = [col(10, 70.0)];
        let guars = [guar(100, 100.0)];
        let ce = [CollateralEdge {
            col_id: 10,
            loan_id: 1,
            ratio: 1.0,
            mode: AllocationMode::Specified,
            repartition: RepartitionMode::Sequential,
        }];
        let ge = [GuaranteeEdge {
            guarantor_id: 100,
            loan_id: 1,
            amount: 100.0,
            mode: AllocationMode::Optimizable,
            repartition: RepartitionMode::Sequential,
        }];
        let r = crm_alloc_greedy(&loans, &cols, &guars, &ce, &ge).unwrap();
        assert!((r.collateral_cover[0] - 70.0).abs() < 1e-9);
        assert!((r.guarantee_cover[0] - 30.0).abs() < 1e-9);
        assert!(r.net_exposure[0].abs() < 1e-9);
        assert_eq!(r.allocations.len(), 2);
        // guarantee record shows the residual loan state
        assert_eq!(r.allocations[1].source_kind, CrmSourceKind::Guarantee);
        assert_eq!(r.allocations[1].amount, 30.0);
        assert_eq!(r.allocations[1].source_remaining, 70.0);
        assert_eq!(r.allocations[1].loan_remaining, 0.0);
    }

    #[test]
    fn riskier_loan_is_covered_first() {
        // Single collateral C(100) shared by L2 (priority 1.0, riskier) and
        // L1 (priority 0.0). Loan priority must beat id order.
        let loans = [
            Loan::new(1, 100.0, 0.0),
            Loan::new(2, 100.0, 1.0),
        ];
        let cols = [col(10, 100.0)];
        let ce = [
            CollateralEdge {
                col_id: 10,
                loan_id: 1,
                ratio: 1.0,
                mode: AllocationMode::Optimizable,
            repartition: RepartitionMode::Sequential,
        },
            CollateralEdge {
                col_id: 10,
                loan_id: 2,
                ratio: 1.0,
                mode: AllocationMode::Optimizable,
            repartition: RepartitionMode::Sequential,
        },
        ];
        let r = crm_alloc_greedy(&loans, &cols, &[], &ce, &[]).unwrap();
        assert!(
            (r.collateral_cover[0]).abs() < 1e-9,
            "low-risk loan uncovered"
        );
        assert!(
            (r.collateral_cover[1] - 100.0).abs() < 1e-9,
            "risky loan fully covered"
        );
        assert_eq!(r.allocations[0].loan_id, 2);
    }

    #[test]
    fn source_priority_controls_consumption_order() {
        // Two collaterals (id 10 low priority 0, id 20 high priority 1) both
        // optimizable to L(150). Higher-priority source is consumed first and
        // fully drains before the next source is touched.
        let loans = [loan(1, 150.0)];
        let cols = [
            Collateral::new(10, 100.0, 0.0),
            Collateral::new(20, 100.0, 1.0),
        ];
        let ce = [
            CollateralEdge {
                col_id: 10,
                loan_id: 1,
                ratio: 1.0,
                mode: AllocationMode::Optimizable,
            repartition: RepartitionMode::Sequential,
        },
            CollateralEdge {
                col_id: 20,
                loan_id: 1,
                ratio: 1.0,
                mode: AllocationMode::Optimizable,
            repartition: RepartitionMode::Sequential,
        },
        ];
        let r = crm_alloc_greedy(&loans, &cols, &[], &ce, &[]).unwrap();
        assert_eq!(r.allocations[0].source_id, 20);
        assert_eq!(r.allocations[0].amount, 100.0);
        assert_eq!(r.allocations[1].source_id, 10);
        assert_eq!(r.allocations[1].amount, 50.0);
        assert!(r.net_exposure[0].abs() < 1e-9);
    }

    #[test]
    fn one_collateral_many_loans_shares_capacity() {
        // C(100) optimizable to three equal-priority loans of 60 each:
        // deterministic id-ascending walk covers 60 + 40 then stops.
        let loans = [loan(1, 60.0), loan(2, 60.0), loan(3, 60.0)];
        let cols = [col(10, 100.0)];
        let ce: Vec<CollateralEdge> = [1, 2, 3]
            .iter()
            .map(|&lid| CollateralEdge {
                col_id: 10,
                loan_id: lid,
                ratio: 1.0,
                mode: AllocationMode::Optimizable,
            repartition: RepartitionMode::Sequential,
        })
            .collect();
        let r = crm_alloc_greedy(&loans, &cols, &[], &ce, &[]).unwrap();
        assert_eq!(r.collateral_cover, vec![60.0, 40.0, 0.0]);
        assert_eq!(r.net_exposure, vec![0.0, 20.0, 60.0]);
    }

    #[test]
    fn specified_guarantee_phase_runs_before_greedy_collateral() {
        // crm-allocation.md phase order: specified col -> specified guar ->
        // greedy col -> greedy guar. A specified guarantee(80) locks first, the
        // optimizable collateral(100) then only covers the residual 20.
        let loans = [loan(1, 100.0)];
        let cols = [col(10, 100.0)];
        let guars = [guar(100, 80.0)];
        let ce = [CollateralEdge {
            col_id: 10,
            loan_id: 1,
            ratio: 1.0,
            mode: AllocationMode::Optimizable,
            repartition: RepartitionMode::Sequential,
        }];
        let ge = [GuaranteeEdge {
            guarantor_id: 100,
            loan_id: 1,
            amount: 80.0,
            mode: AllocationMode::Specified,
            repartition: RepartitionMode::Sequential,
        }];
        let r = crm_alloc_greedy(&loans, &cols, &guars, &ce, &ge).unwrap();
        assert_eq!(r.allocations.len(), 2);
        assert_eq!(r.allocations[0].source_kind, CrmSourceKind::Guarantee);
        assert_eq!(r.allocations[0].stage, CrmStage::Specified);
        assert_eq!(r.allocations[0].amount, 80.0);
        assert_eq!(r.allocations[1].source_kind, CrmSourceKind::Collateral);
        assert_eq!(r.allocations[1].stage, CrmStage::Greedy);
        assert_eq!(r.allocations[1].amount, 20.0);
        assert!(r.net_exposure[0].abs() < 1e-9);
        assert!((r.guarantee_cover[0] - 80.0).abs() < 1e-9);
        assert!((r.collateral_cover[0] - 20.0).abs() < 1e-9);
    }

    #[test]
    fn sparse_ids_are_supported_and_results_aligned_to_input() {
        let loans = [loan(1000, 80.0), loan(2000, 50.0)];
        let cols = [col(9000, 120.0)];
        let ce = [
            CollateralEdge {
                col_id: 9000,
                loan_id: 2000,
                ratio: 1.0,
                mode: AllocationMode::Optimizable,
            repartition: RepartitionMode::Sequential,
        },
            CollateralEdge {
                col_id: 9000,
                loan_id: 1000,
                ratio: 1.0,
                mode: AllocationMode::Optimizable,
            repartition: RepartitionMode::Sequential,
        },
        ];
        // loan 2000 (id 2000) is riskier? equal priority -> id ascending:
        // 1000 covered first (80), then 2000 (40 of remaining 40).
        let r = crm_alloc_greedy(&loans, &cols, &[], &ce, &[]).unwrap();
        assert_eq!(r.loan_id, vec![1000, 2000]);
        assert_eq!(r.collateral_cover, vec![80.0, 40.0]);
        assert_eq!(r.net_exposure, vec![0.0, 10.0]);
    }

    #[test]
    fn deterministic_across_reruns() {
        let loans = [
            Loan::new(3, 90.0, 0.4),
            Loan::new(1, 70.0, 0.9),
            Loan::new(2, 50.0, 0.9),
        ];
        let cols = [
            Collateral::new(30, 80.0, 0.5),
            Collateral::new(20, 80.0, 0.5),
        ];
        let guars = [Guarantor::new(5, 100.0, 2.0)];
        let ce: Vec<CollateralEdge> = (1..=3)
            .flat_map(|lid| {
                [20, 30].iter().map(move |&cid| CollateralEdge {
                    col_id: cid,
                    loan_id: lid,
                    ratio: 1.0,
                    mode: if lid % 2 == 0 {
                        AllocationMode::Specified
                    } else {
                        AllocationMode::Optimizable
                    },
                            repartition: RepartitionMode::Sequential,
        })
            })
            .collect();
        let ge: Vec<GuaranteeEdge> = (1..=3)
            .map(|lid| GuaranteeEdge {
                guarantor_id: 5,
                loan_id: lid,
                amount: 40.0,
                mode: AllocationMode::Optimizable,
            repartition: RepartitionMode::Sequential,
        })
            .collect();

        let a = crm_alloc_greedy(&loans, &cols, &guars, &ce, &ge).unwrap();
        let b = crm_alloc_greedy(&loans, &cols, &guars, &ce, &ge).unwrap();
        assert_eq!(a.loan_id, b.loan_id);
        assert_eq!(a.collateral_cover, b.collateral_cover);
        assert_eq!(a.guarantee_cover, b.guarantee_cover);
        assert_eq!(a.net_exposure, b.net_exposure);
        assert_eq!(a.allocations.len(), b.allocations.len());
        for (x, y) in a.allocations.iter().zip(b.allocations.iter()) {
            assert_eq!(x.source_id, y.source_id);
            assert_eq!(x.loan_id, y.loan_id);
            assert_eq!(x.amount, y.amount);
            assert_eq!(x.stage, y.stage);
            assert_eq!(x.source_kind, y.source_kind);
        }
    }

    #[test]
    fn audit_invariants_no_source_overallocated_no_loan_overcovered() {
        // Random-ish connected instance: after allocation, sum per source in
        // the audit must never exceed capacity and per-loan total covers must
        // never exceed exposure.
        let loans: Vec<Loan> = (1..=8).map(|i| loan(i, 10.0 * i as f64)).collect();
        let cols: Vec<Collateral> = (1..=4).map(|i| col(100 + i, 25.0 * i as f64)).collect();
        let guars: Vec<Guarantor> = (1..=3).map(|i| guar(200 + i, 20.0 * i as f64)).collect();
        let ce: Vec<CollateralEdge> = (1..=4)
            .flat_map(move |ci| {
                (1..=8)
                    .filter(move |li| (ci + li) % 2 == 0)
                    .map(move |li| CollateralEdge {
                        col_id: 100 + ci,
                        loan_id: li,
                        ratio: 0.5,
                        mode: if li % 3 == 0 {
                            AllocationMode::Specified
                        } else {
                            AllocationMode::Optimizable
                        },
                                repartition: RepartitionMode::Sequential,
        })
            })
            .collect();
        let ge: Vec<GuaranteeEdge> = (1..=3)
            .flat_map(move |gi| {
                (1..=8)
                    .filter(move |li| (gi * li) % 2 == 1)
                    .map(move |li| GuaranteeEdge {
                        guarantor_id: 200 + gi,
                        loan_id: li,
                        amount: 5.0,
                        mode: if li % 2 == 0 {
                            AllocationMode::Specified
                        } else {
                            AllocationMode::Optimizable
                        },
                                repartition: RepartitionMode::Sequential,
        })
            })
            .collect();

        let r = crm_alloc_greedy(&loans, &cols, &guars, &ce, &ge).unwrap();
        let mut per_source: std::collections::HashMap<u64, f64> = std::collections::HashMap::new();
        let mut per_loan: std::collections::HashMap<u64, f64> = std::collections::HashMap::new();
        for a in &r.allocations {
            assert!(a.amount > 0.0);
            *per_source.entry(a.source_id).or_insert(0.0) += a.amount;
            *per_loan.entry(a.loan_id).or_insert(0.0) += a.amount;
        }
        for c in &cols {
            assert!(per_source.get(&c.id).copied().unwrap_or(0.0) <= c.capacity + 1e-9);
        }
        for g in &guars {
            assert!(per_source.get(&g.id).copied().unwrap_or(0.0) <= g.capacity + 1e-9);
        }
        for l in &loans {
            assert!(per_loan.get(&l.id).copied().unwrap_or(0.0) <= l.exposure + 1e-9);
        }
        // net exposure consistency
        for i in 0..r.loan_id.len() {
            let net = r.exposure[i] - r.collateral_cover[i] - r.guarantee_cover[i];
            assert!((net - r.net_exposure[i]).abs() < 1e-9);
            assert!(r.net_exposure[i] > -1e-9);
        }
    }

    #[test]
    fn rejects_duplicate_and_dangling_ids() {
        assert!(matches!(
            crm_alloc_greedy(&[loan(1, 10.0), loan(1, 20.0)], &[], &[], &[], &[]),
            Err(CrmError::DuplicateLoan(1))
        ));
        assert!(matches!(
            crm_alloc_greedy(&[], &[col(5, 10.0), col(5, 9.0)], &[], &[], &[]),
            Err(CrmError::DuplicateCollateral(5))
        ));
        assert!(matches!(
            crm_alloc_greedy(&[], &[], &[guar(7, 1.0), guar(7, 2.0)], &[], &[]),
            Err(CrmError::DuplicateGuarantor(7))
        ));
        // edge to an unknown collateral
        let err = crm_alloc_greedy(
            &[loan(1, 10.0)],
            &[col(5, 10.0)],
            &[],
            &[CollateralEdge {
                col_id: 999,
                loan_id: 1,
                ratio: 1.0,
                mode: AllocationMode::Specified,
            repartition: RepartitionMode::Sequential,
        }],
            &[],
        )
        .unwrap_err();
        assert!(matches!(
            err,
            CrmError::DanglingCollateralEdge {
                col_id: 999,
                loan_id: 1,
            }
        ));
        // edge to an unknown loan
        let err = crm_alloc_greedy(
            &[loan(1, 10.0)],
            &[],
            &[guar(5, 10.0)],
            &[],
            &[GuaranteeEdge {
                guarantor_id: 5,
                loan_id: 777,
                amount: 1.0,
                mode: AllocationMode::Specified,
            repartition: RepartitionMode::Sequential,
        }],
        )
        .unwrap_err();
        assert!(matches!(
            err,
            CrmError::DanglingGuaranteeEdge {
                guarantor_id: 5,
                loan_id: 777,
            }
        ));
    }

    #[test]
    fn parallel_components_match_serial_covers() {
        // 80 disjoint loan↔collateral pairs → many components; covers must
        // match the forced-serial path (audit order may differ / be restabilised).
        let n = 80u64;
        let loans: Vec<Loan> = (1..=n).map(|i| loan(i, 100.0)).collect();
        let cols: Vec<Collateral> = (1..=n).map(|i| col(1000 + i, 40.0)).collect();
        let ce: Vec<CollateralEdge> = (1..=n)
            .map(|i| CollateralEdge {
                col_id: 1000 + i,
                loan_id: i,
                ratio: 1.0,
                mode: AllocationMode::Optimizable,
            repartition: RepartitionMode::Sequential,
        })
            .collect();

        std::env::set_var("GTV_CRM_PARALLEL", "0");
        let serial = crm_alloc_greedy(&loans, &cols, &[], &ce, &[]).unwrap();
        std::env::set_var("GTV_CRM_PARALLEL", "1");
        let parallel = crm_alloc_greedy(&loans, &cols, &[], &ce, &[]).unwrap();
        std::env::remove_var("GTV_CRM_PARALLEL");

        assert_eq!(serial.loan_id, parallel.loan_id);
        assert_eq!(serial.collateral_cover, parallel.collateral_cover);
        assert_eq!(serial.guarantee_cover, parallel.guarantee_cover);
        assert_eq!(serial.cds_cover, parallel.cds_cover);
        assert_eq!(serial.net_exposure, parallel.net_exposure);
        assert_eq!(serial.allocations.len(), parallel.allocations.len());
        // Parallel audit is stably sorted; amounts per (source, loan) match.
        let mut s_keys: Vec<_> = serial
            .allocations
            .iter()
            .map(|a| (a.source_id, a.loan_id, a.amount.to_bits()))
            .collect();
        let mut p_keys: Vec<_> = parallel
            .allocations
            .iter()
            .map(|a| (a.source_id, a.loan_id, a.amount.to_bits()))
            .collect();
        s_keys.sort_unstable();
        p_keys.sort_unstable();
        assert_eq!(s_keys, p_keys);
    }

    #[test]
    fn pro_rata_splits_equal_residual_ead() {
        let loans = [Loan::new(1, 100.0, 1.0), Loan::new(2, 100.0, 1.0)];
        let mut col = Collateral::new(10, 80.0, 1.0);
        col.meta.repartition = RepartitionMode::ProRata;
        let ce = [
            CollateralEdge {
                col_id: 10,
                loan_id: 1,
                ratio: 1.0,
                mode: AllocationMode::Optimizable,
                repartition: RepartitionMode::ProRata,
            },
            CollateralEdge {
                col_id: 10,
                loan_id: 2,
                ratio: 1.0,
                mode: AllocationMode::Optimizable,
                repartition: RepartitionMode::ProRata,
            },
        ];
        let r = crm_alloc_greedy_with_options(
            &loans,
            &[col],
            &[],
            &ce,
            &[],
            &[],
            &[],
            &CrmAllocOptions::legacy(),
        )
        .unwrap();
        assert!((r.collateral_cover[0] - 40.0).abs() < 1e-9);
        assert!((r.collateral_cover[1] - 40.0).abs() < 1e-9);
        assert!(r
            .allocations
            .iter()
            .all(|a| a.repartition == RepartitionMode::ProRata));
    }

    #[test]
    fn expo_formula_risk_weight_overrides_priority_walk() {
        // Same priority; EXPO formula uses risk_weight desc → L1 before L2.
        let mut l1 = Loan::new(1, 50.0, 1.0);
        l1.risk_weight = 1.5;
        let mut l2 = Loan::new(2, 50.0, 1.0);
        l2.risk_weight = 0.2;
        let col = Collateral::new(10, 50.0, 1.0);
        let ce = [
            CollateralEdge {
                col_id: 10,
                loan_id: 1,
                ratio: 1.0,
                mode: AllocationMode::Optimizable,
                repartition: RepartitionMode::Sequential,
            },
            CollateralEdge {
                col_id: 10,
                loan_id: 2,
                ratio: 1.0,
                mode: AllocationMode::Optimizable,
                repartition: RepartitionMode::Sequential,
            },
        ];
        let mut opts = CrmAllocOptions::legacy();
        opts.expo_formula = vec![FormulaCol {
            indicator: CrmIndicator::RiskWeight,
            direction: SortDirection::Desc,
        }];
        let r = crm_alloc_greedy_with_options(
            &[l1, l2],
            &[col],
            &[],
            &ce,
            &[],
            &[],
            &[],
            &opts,
        )
        .unwrap();
        assert_eq!(r.allocations[0].loan_id, 1);
        assert!((r.collateral_cover[0] - 50.0).abs() < 1e-9);
        assert!((r.collateral_cover[1] - 0.0).abs() < 1e-9);
    }

    #[test]
    fn rule_priority_conflict_takes_lowest_priority_rank() {
        // Two collaterals; lower rule_rank consumed first regardless of priority.
        let mut c_hi = Collateral::new(1, 30.0, 99.0);
        c_hi.meta.rule_rank = 20;
        let mut c_lo = Collateral::new(2, 30.0, 1.0);
        c_lo.meta.rule_rank = 5;
        let loan = Loan::new(10, 50.0, 1.0);
        let ce = [
            CollateralEdge {
                col_id: 1,
                loan_id: 10,
                ratio: 1.0,
                mode: AllocationMode::Optimizable,
                repartition: RepartitionMode::Sequential,
            },
            CollateralEdge {
                col_id: 2,
                loan_id: 10,
                ratio: 1.0,
                mode: AllocationMode::Optimizable,
                repartition: RepartitionMode::Sequential,
            },
        ];
        let r = crm_alloc_greedy(&[loan], &[c_hi, c_lo], &[], &ce, &[]).unwrap();
        assert_eq!(r.allocations[0].source_id, 2);
        assert_eq!(r.allocations[0].rule_rank, 5);
    }
}
